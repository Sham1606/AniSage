"""
AniList GraphQL Collector — fetches anime data from graphql.anilist.co

AniList has:
  - Extended tag system (200+ tags like "Found Family", "Unreliable Narrator")
  - Mean score on 0-100 scale
  - Rich season/year data
  - No API key required

Rate limit: 90 requests/minute (we use 60/min to be safe)
Pagination: uses page + perPage (max 50 per page)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase1.schemas.anime_schema import AnimeDocument
from phase1.utils.helpers import (
    AsyncRateLimiter,
    CheckpointManager,
    append_jsonl,
    load_existing_ids,
    log_error,
    log_info,
    log_success,
    log_warning,
    make_progress,
)

ANILIST_URL = "https://graphql.anilist.co"
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

RAW_FILE = RAW_DIR / "anilist_raw.jsonl"
CHECKPOINT_FILE = CHECKPOINT_DIR / "anilist_checkpoint.json"

# ── GraphQL query ─────────────────────────────────────────────────────────────
# Fetches all the data we need in one query per page.

ANIME_QUERY = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      total
      currentPage
      lastPage
      hasNextPage
    }
    media(type: ANIME, sort: ID) {
      id
      idMal
      title {
        romaji
        english
        native
      }
      description(asHtml: false)
      genres
      tags {
        name
        rank
        isMediaSpoiler
      }
      format
      status
      episodes
      duration
      season
      seasonYear
      startDate { year month day }
      studios(isMain: true) {
        nodes { name }
      }
      source
      averageScore
      meanScore
      popularity
      favourites
      coverImage {
        large
        medium
      }
      siteUrl
      isAdult
    }
  }
}
"""


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_anilist_entry(entry: dict) -> Optional[AnimeDocument]:
    """Convert a raw AniList media object into an AnimeDocument."""
    try:
        # Title: prefer English, fall back to romaji
        title_obj = entry.get("title", {}) or {}
        title = title_obj.get("english") or title_obj.get("romaji") or ""
        if not title:
            return None

        # Tags: filter out spoilers, sort by rank (relevance), take top 20
        raw_tags = entry.get("tags", []) or []
        tags = [
            t["name"] for t in sorted(raw_tags, key=lambda x: x.get("rank", 0), reverse=True)
            if not t.get("isMediaSpoiler", False) and t.get("name")
        ][:20]

        # Studios
        studios_data = entry.get("studios", {}).get("nodes", []) or []
        studios = [s["name"] for s in studios_data if s.get("name")]

        # Format mapping: AniList uses ALL_CAPS
        format_map = {
            "TV": "TV", "TV_SHORT": "TV", "MOVIE": "Movie",
            "SPECIAL": "Special", "OVA": "OVA", "ONA": "ONA",
            "MUSIC": "Music",
        }
        media_type = format_map.get(entry.get("format", ""), entry.get("format"))

        # Status mapping
        status_map = {
            "FINISHED": "Finished Airing",
            "RELEASING": "Currently Airing",
            "NOT_YET_RELEASED": "Not yet aired",
            "CANCELLED": "Cancelled",
            "HIATUS": "On Hiatus",
        }
        status = status_map.get(entry.get("status", ""), entry.get("status"))

        # Season
        season_map = {
            "WINTER": "Winter", "SPRING": "Spring",
            "SUMMER": "Summer", "FALL": "Fall",
        }
        season = season_map.get(entry.get("season", ""), None)

        # Cover image
        cover = entry.get("coverImage", {}) or {}
        image_url = cover.get("large") or cover.get("medium")

        # Mean score: AniList uses 0-100
        mean_score = entry.get("meanScore") or entry.get("averageScore")

        doc = AnimeDocument(
            anilist_id=entry.get("id"),
            mal_id=entry.get("idMal"),
            title=title,
            title_japanese=title_obj.get("native"),
            title_synonyms=[title_obj.get("romaji")] if title_obj.get("romaji") and title_obj.get("romaji") != title else [],
            synopsis=entry.get("description") or "",
            genres=entry.get("genres") or [],
            tags=tags,
            media_type=media_type,
            status=status,
            episodes=entry.get("episodes"),
            duration_per_ep_min=entry.get("duration"),
            year=entry.get("seasonYear") or (entry.get("startDate") or {}).get("year"),
            season=season,
            studios=studios,
            source=(entry.get("source") or "").replace("_", " ").title() if entry.get("source") else None,
            mean_score=float(mean_score) if mean_score else None,
            popularity=entry.get("popularity"),
            favorites=entry.get("favourites"),
            image_url=image_url,
            anilist_url=entry.get("siteUrl"),
            is_adult=entry.get("isAdult", False),
            data_sources=["anilist"],
        )
        doc.compute_synopsis_tokens()
        return doc

    except Exception as e:
        log_warning(f"Failed to parse AniList ID {entry.get('id')}: {e}")
        return None


# ── Async fetcher ─────────────────────────────────────────────────────────────

class AniListCollector:
    """
    Async GraphQL collector for AniList.

    Args:
        output_path:    JSONL output file path
        checkpoint_path: Checkpoint file for resume support
        rate_limit:     Requests per second (AniList allows 90/min = 1.5/s)
        max_pages:      Limit pages for testing
        per_page:       Results per page (max 50)
        skip_adult:     Skip adult-rated entries
    """

    def __init__(
        self,
        output_path: Path = RAW_FILE,
        checkpoint_path: Path = CHECKPOINT_FILE,
        rate_limit: float = 1.0,
        max_pages: Optional[int] = None,
        per_page: int = 50,
        skip_adult: bool = True,
    ) -> None:
        self.output_path = output_path
        self.checkpoint_path = checkpoint_path
        self.rate_limiter = AsyncRateLimiter(calls_per_second=rate_limit)
        self.max_pages = max_pages
        self.per_page = per_page
        self.skip_adult = skip_adult

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.checkpoint = CheckpointManager(self.checkpoint_path)
        self.existing_ids: set[int] = set()
        self._load_existing_anilist_ids()

    def _load_existing_anilist_ids(self) -> None:
        """Load AniList IDs already written (not MAL IDs)."""
        if not self.output_path.exists():
            return
        import orjson
        with open(self.output_path, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = orjson.loads(line)
                    if aid := record.get("anilist_id"):
                        self.existing_ids.add(aid)

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        wait=wait_exponential(multiplier=2, min=5, max=120),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _fetch_page(self, session: aiohttp.ClientSession, page: int) -> dict:
        """Execute a GraphQL query for a given page."""
        payload = {
            "query": ANIME_QUERY,
            "variables": {"page": page, "perPage": self.per_page},
        }

        async with self.rate_limiter:
            async with session.post(
                ANILIST_URL,
                json=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:

                # AniList returns 429 with Retry-After header
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", "60"))
                    log_warning(f"Rate limited on page {page}. Waiting {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    raise aiohttp.ClientError("Rate limited")

                if resp.status != 200:
                    raise aiohttp.ClientError(f"HTTP {resp.status} on page {page}")

                result = await resp.json()

                # GraphQL errors come back as 200 with an "errors" key
                if "errors" in result:
                    errors = result["errors"]
                    raise aiohttp.ClientError(f"GraphQL errors: {errors}")

                return result

    async def collect(self) -> int:
        """Main collection loop. Returns total records written."""
        connector = aiohttp.TCPConnector(limit=3)
        async with aiohttp.ClientSession(connector=connector) as session:

            # ── Step 1: Get total pages ───────────────────────────────────────
            log_info("Fetching first page to get total AniList pages...")
            first = await self._fetch_page(session, 1)
            page_info = first["data"]["Page"]["pageInfo"]
            total_pages = page_info["lastPage"]

            if self.max_pages:
                total_pages = min(total_pages, self.max_pages)

            log_info(f"AniList total pages: {total_pages} (~{total_pages * self.per_page:,} anime)")

            records_written = len(self.existing_ids)
            log_info(f"Already have {records_written:,} AniList records. Resuming...")

            # ── Step 2: Paginate ──────────────────────────────────────────────
            with make_progress() as progress:
                task = progress.add_task("Collecting from AniList GraphQL...", total=total_pages)

                for page in range(1, total_pages + 1):
                    if self.checkpoint.is_done(page):
                        progress.advance(task)
                        continue

                    try:
                        data = await self._fetch_page(session, page)
                        media_list = data["data"]["Page"]["media"]

                        page_written = 0
                        for entry in media_list:
                            # Skip adult content
                            if self.skip_adult and entry.get("isAdult"):
                                continue

                            anilist_id = entry.get("id")
                            if anilist_id and anilist_id in self.existing_ids:
                                continue

                            doc = parse_anilist_entry(entry)
                            if doc and doc.title:
                                append_jsonl(self.output_path, doc.model_dump())
                                if doc.anilist_id:
                                    self.existing_ids.add(doc.anilist_id)
                                page_written += 1

                        records_written += page_written
                        self.checkpoint.mark_done(page)

                        if page % 50 == 0:
                            self.checkpoint.save()
                            log_info(f"Page {page}/{total_pages} — {records_written:,} total records")

                    except Exception as e:
                        log_error(f"AniList page {page} failed: {e}")
                        self.checkpoint.mark_failed(page)

                    progress.advance(task)

            self.checkpoint.save()
            log_success(f"AniList collection complete: {records_written:,} records → {self.output_path}")
            return records_written


async def run_anilist_collection(
    max_pages: Optional[int] = None,
    skip_adult: bool = True,
) -> int:
    collector = AniListCollector(max_pages=max_pages, skip_adult=skip_adult)
    return await collector.collect()


if __name__ == "__main__":
    asyncio.run(run_anilist_collection())

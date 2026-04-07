"""
Jikan v4 Collector — fetches anime data from api.jikan.moe

Strategy:
  - Uses /anime?order_by=mal_id&page=N to paginate all anime
  - Handles 429 rate limiting automatically (Jikan allows ~3 req/sec)
  - Saves a checkpoint every 100 pages so you can resume any time
  - Appends results to data/raw/jikan_raw.jsonl incrementally

Jikan API docs: https://docs.api.jikan.moe/
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

JIKAN_BASE = "https://api.jikan.moe/v4"
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

RAW_FILE = RAW_DIR / "jikan_raw.jsonl"
CHECKPOINT_FILE = CHECKPOINT_DIR / "jikan_checkpoint.json"


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_jikan_entry(entry: dict) -> Optional[AnimeDocument]:
    """Convert a raw Jikan API anime object into an AnimeDocument."""
    try:
        def extract_names(lst: list, key: str = "name") -> list[str]:
            return [item[key] for item in (lst or []) if item.get(key)]

        # Score handling
        score = entry.get("score")
        score = float(score) if score is not None else None

        # Image URL — prefer jpg large
        images = entry.get("images", {})
        image_url = (
            images.get("jpg", {}).get("large_image_url")
            or images.get("jpg", {}).get("image_url")
        )

        # Duration in minutes (Jikan returns "24 min per ep" string)
        duration_str = entry.get("duration", "") or ""
        duration_min = None
        if "min" in duration_str:
            try:
                duration_min = int(duration_str.split("min")[0].strip().split()[-1])
            except (ValueError, IndexError):
                pass

        # Year — try aired.from first, then year field
        year = entry.get("year")
        if not year:
            aired_from = entry.get("aired", {}).get("from", "")
            if aired_from:
                try:
                    year = int(aired_from[:4])
                except (ValueError, TypeError):
                    pass

        doc = AnimeDocument(
            mal_id=entry.get("mal_id"),
            title=entry.get("title_english") or entry.get("title") or "",
            title_japanese=entry.get("title_japanese"),
            title_synonyms=[
                s.get("title", "") for s in entry.get("titles", [])
                if s.get("title") and s.get("type") not in ("Default", "Japanese")
            ],
            synopsis=entry.get("synopsis") or "",
            genres=extract_names(entry.get("genres", [])),
            themes=extract_names(entry.get("themes", [])),
            demographics=extract_names(entry.get("demographics", [])),
            media_type=entry.get("type"),
            status=entry.get("status"),
            episodes=entry.get("episodes"),
            duration_per_ep_min=duration_min,
            year=year,
            season=entry.get("season", "").capitalize() if entry.get("season") else None,
            studios=extract_names(entry.get("studios", [])),
            source=entry.get("source"),
            score=score,
            scored_by=entry.get("scored_by"),
            rank=entry.get("rank"),
            popularity=entry.get("popularity"),
            members=entry.get("members"),
            favorites=entry.get("favorites"),
            image_url=image_url,
            trailer_url=entry.get("trailer", {}).get("url") if entry.get("trailer") else None,
            mal_url=entry.get("url"),
            is_adult=entry.get("rating", "").startswith("Rx") if entry.get("rating") else False,
            data_sources=["jikan"],
        )
        doc.compute_synopsis_tokens()
        return doc

    except Exception as e:
        log_warning(f"Failed to parse MAL ID {entry.get('mal_id')}: {e}")
        return None


# ── Async fetcher ─────────────────────────────────────────────────────────────

class JikanCollector:
    """
    Async collector that pages through all anime on Jikan.

    Args:
        output_path:    Where to write JSONL records (appended incrementally)
        checkpoint_path: Resume state path
        rate_limit:     Requests per second (Jikan allows ~3, we use 2 to be safe)
        max_pages:      Set a limit for testing (None = fetch everything)
        skip_adult:     Skip Rx-rated entries
    """

    def __init__(
        self,
        output_path: Path = RAW_FILE,
        checkpoint_path: Path = CHECKPOINT_FILE,
        rate_limit: float = 2.0,
        max_pages: Optional[int] = None,
        skip_adult: bool = True,
    ) -> None:
        self.output_path = output_path
        self.checkpoint_path = checkpoint_path
        self.rate_limiter = AsyncRateLimiter(calls_per_second=rate_limit)
        self.max_pages = max_pages
        self.skip_adult = skip_adult

        # Ensure directories exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.checkpoint = CheckpointManager(self.checkpoint_path)
        self.existing_ids = load_existing_ids(self.output_path)

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _fetch_page(self, session: aiohttp.ClientSession, page: int) -> dict:
        """Fetch a single page of anime from Jikan with retry logic."""
        url = f"{JIKAN_BASE}/anime"
        params = {
            "order_by": "mal_id",
            "sort": "asc",
            "page": page,
            "limit": 25,  # Max allowed by Jikan
            "sfw": "true" if self.skip_adult else "false",
        }

        async with self.rate_limiter:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", "10"))
                    log_warning(f"Rate limited on page {page}. Waiting {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    raise aiohttp.ClientError("Rate limited — retrying")

                if resp.status != 200:
                    raise aiohttp.ClientError(f"HTTP {resp.status} on page {page}")

                return await resp.json()

    async def collect(self) -> int:
        """
        Main collection loop. Returns total records written.

        Flow:
          1. Fetch page 1 to get total pages count
          2. Iterate pages, skipping already-checkpointed ones
          3. Parse each anime entry → AnimeDocument
          4. Append to JSONL file
          5. Save checkpoint every 50 pages
        """
        connector = aiohttp.TCPConnector(limit=5)
        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

            # ── Step 1: Get total page count ──────────────────────────────────
            log_info("Fetching first page to determine total pages...")
            first_page_data = await self._fetch_page(session, 1)
            pagination = first_page_data.get("pagination", {})
            total_pages = pagination.get("last_visible_page", 1)

            if self.max_pages:
                total_pages = min(total_pages, self.max_pages)

            log_info(f"Total pages to collect: {total_pages} (~{total_pages * 25:,} anime)")

            records_written = len(self.existing_ids)
            log_info(f"Already have {records_written:,} records. Resuming...")

            # ── Step 2: Paginate ──────────────────────────────────────────────
            with make_progress() as progress:
                task = progress.add_task("Collecting from Jikan API...", total=total_pages)

                for page in range(1, total_pages + 1):
                    # Skip if already checkpointed
                    if self.checkpoint.is_done(page):
                        progress.advance(task)
                        continue

                    try:
                        data = await self._fetch_page(session, page)
                        anime_list = data.get("data", [])

                        page_written = 0
                        for entry in anime_list:
                            mal_id = entry.get("mal_id")

                            # Skip already-written IDs (handles partial page restarts)
                            if mal_id and mal_id in self.existing_ids:
                                continue

                            doc = parse_jikan_entry(entry)
                            if doc and doc.title:
                                append_jsonl(self.output_path, doc.model_dump())
                                self.existing_ids.add(doc.mal_id)
                                page_written += 1

                        records_written += page_written
                        self.checkpoint.mark_done(page)

                        # Save checkpoint every 50 pages
                        if page % 50 == 0:
                            self.checkpoint.save()
                            log_info(f"Page {page}/{total_pages} — {records_written:,} total records")

                    except Exception as e:
                        log_error(f"Page {page} failed after retries: {e}")
                        self.checkpoint.mark_failed(page)

                    progress.advance(task)

            # ── Final checkpoint save ─────────────────────────────────────────
            self.checkpoint.save()
            log_success(f"Jikan collection complete: {records_written:,} anime written to {self.output_path}")
            return records_written


# ── Convenience runner ────────────────────────────────────────────────────────

async def run_jikan_collection(
    max_pages: Optional[int] = None,
    skip_adult: bool = True,
) -> int:
    collector = JikanCollector(max_pages=max_pages, skip_adult=skip_adult)
    return await collector.collect()


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_jikan_collection())

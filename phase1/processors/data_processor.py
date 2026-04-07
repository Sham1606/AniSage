"""
Data Processor — merges Jikan + AniList records, deduplicates,
cleans, filters, and builds embedding_text for every anime.

Input:  data/raw/jikan_raw.jsonl + data/raw/anilist_raw.jsonl
Output: data/processed/anime_merged.jsonl
        data/processed/anime_clean.parquet   (for fast analysis)
        data/processed/processing_report.json
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase1.schemas.anime_schema import AnimeDocument
from phase1.utils.helpers import (
    append_jsonl,
    log_error,
    log_info,
    log_success,
    log_warning,
    make_progress,
    read_jsonl,
)

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

JIKAN_RAW = RAW_DIR / "jikan_raw.jsonl"
ANILIST_RAW = RAW_DIR / "anilist_raw.jsonl"
MERGED_OUTPUT = PROCESSED_DIR / "anime_merged.jsonl"
PARQUET_OUTPUT = PROCESSED_DIR / "anime_clean.parquet"
REPORT_OUTPUT = PROCESSED_DIR / "processing_report.json"


# ── Genre normalisation ───────────────────────────────────────────────────────
# Jikan and AniList use slightly different labels — harmonise them.

GENRE_ALIASES: dict[str, str] = {
    "Sci-Fi": "Science Fiction",
    "Sci Fi": "Science Fiction",
    "SciFi": "Science Fiction",
    "Slice of Life": "Slice of Life",
    "Shounen": "Shounen",
    "Shonen": "Shounen",
    "Shoujo": "Shoujo",
    "Shojo": "Shoujo",
    "Seinen": "Seinen",
    "Josei": "Josei",
    "Isekai": "Isekai",
    "Mecha": "Mecha",
    "Ecchi": "Ecchi",
    "Harem": "Harem",
    "Reverse Harem": "Reverse Harem",
}

def normalize_genre(g: str) -> str:
    return GENRE_ALIASES.get(g.strip(), g.strip())


# ── Merge strategy ────────────────────────────────────────────────────────────

class DataMerger:
    """
    Loads both raw JSONL files, merges on MAL ID,
    deduplicates, and produces a clean merged dataset.

    Merge priority:
      - Jikan:   title, synopsis, genres, themes, demographics,
                 score, rank, members, favorites, mal_url, image_url
      - AniList: tags (unique to AniList), mean_score, anilist_id, anilist_url
      - Both provide: media_type, status, episodes, year, season, studios, source
        → Jikan wins for these (more reliable metadata for MAL dataset)
    """

    def __init__(self) -> None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def _load_source(self, path: Path, source_name: str) -> dict[int, AnimeDocument]:
        """Load a JSONL file into a dict keyed by MAL ID."""
        if not path.exists():
            log_warning(f"{source_name} raw file not found at {path}")
            return {}

        log_info(f"Loading {source_name} raw data from {path}...")
        records = read_jsonl(path)
        result: dict[int, AnimeDocument] = {}

        for record in records:
            try:
                doc = AnimeDocument(**record)
                key = doc.mal_id or doc.anilist_id
                if key:
                    result[key] = doc
            except Exception as e:
                log_warning(f"Skipping malformed record: {e}")

        log_info(f"Loaded {len(result):,} {source_name} records")
        return result

    def merge(self) -> list[AnimeDocument]:
        """
        Merge Jikan (primary) with AniList (enrichment).
        Returns list of merged AnimeDocuments sorted by MAL ID.
        """
        jikan_docs = self._load_source(JIKAN_RAW, "Jikan")
        anilist_docs = self._load_source(ANILIST_RAW, "AniList")

        # Build AniList lookup by MAL ID (where available)
        anilist_by_mal: dict[int, AnimeDocument] = {}
        anilist_no_mal: dict[int, AnimeDocument] = {}  # keyed by anilist_id

        for doc in anilist_docs.values():
            if doc.mal_id:
                anilist_by_mal[doc.mal_id] = doc
            elif doc.anilist_id:
                anilist_no_mal[doc.anilist_id] = doc

        merged: dict[int, AnimeDocument] = {}

        # ── Start with all Jikan entries ──────────────────────────────────────
        for mal_id, jikan_doc in jikan_docs.items():
            doc = jikan_doc
            # Enrich with AniList data if MAL ID matches
            if mal_id in anilist_by_mal:
                anilist_doc = anilist_by_mal[mal_id]
                # Inject AniList-exclusive fields
                doc.tags = anilist_doc.tags  # AniList tags are the key enrichment
                doc.anilist_id = anilist_doc.anilist_id
                doc.anilist_url = anilist_doc.anilist_url
                doc.mean_score = anilist_doc.mean_score
                doc.data_sources = list(set(doc.data_sources + ["anilist"]))
            merged[mal_id] = doc

        # ── Add AniList-only entries (no MAL ID) ──────────────────────────────
        for anilist_id, doc in anilist_no_mal.items():
            merged[-(anilist_id)] = doc  # negative key to avoid collision

        log_success(f"Merged dataset: {len(merged):,} total entries")
        return sorted(merged.values(), key=lambda d: d.mal_id or 0)


# ── Processor ─────────────────────────────────────────────────────────────────

class DataProcessor:
    """
    Applies cleaning rules, filters low-quality entries,
    normalizes genres, and builds embedding_text for each anime.
    """

    def __init__(self) -> None:
        self.stats: dict = {
            "input_count": 0,
            "removed_no_synopsis": 0,
            "removed_too_short": 0,
            "removed_adult": 0,
            "removed_music_type": 0,
            "output_count": 0,
            "sources": Counter(),
            "media_types": Counter(),
            "years": Counter(),
            "top_genres": Counter(),
        }

    def process(
        self,
        docs: list[AnimeDocument],
        min_synopsis_tokens: int = 30,
        skip_adult: bool = True,
        skip_music_type: bool = True,
    ) -> list[AnimeDocument]:
        """
        Apply all cleaning and enrichment steps.

        Args:
            docs: Raw merged documents
            min_synopsis_tokens: Minimum synopsis length to keep
            skip_adult: Remove 18+ entries
            skip_music_type: Remove music videos (rarely useful for recommendations)
        """
        self.stats["input_count"] = len(docs)
        clean: list[AnimeDocument] = []

        with make_progress() as progress:
            task = progress.add_task("Processing & cleaning anime records...", total=len(docs))

            for doc in docs:

                # ── Filter: adult content ─────────────────────────────────────
                if skip_adult and doc.is_adult:
                    self.stats["removed_adult"] += 1
                    progress.advance(task)
                    continue

                # ── Filter: music videos ──────────────────────────────────────
                if skip_music_type and doc.media_type == "Music":
                    self.stats["removed_music_type"] += 1
                    progress.advance(task)
                    continue

                # ── Filter: no synopsis ───────────────────────────────────────
                if not doc.synopsis.strip():
                    self.stats["removed_no_synopsis"] += 1
                    progress.advance(task)
                    continue

                # ── Filter: synopsis too short ────────────────────────────────
                doc.compute_synopsis_tokens()
                if doc.synopsis_token_count < min_synopsis_tokens:
                    self.stats["removed_too_short"] += 1
                    progress.advance(task)
                    continue

                # ── Normalize genres ──────────────────────────────────────────
                doc.genres = list(dict.fromkeys(
                    normalize_genre(g) for g in doc.genres if g
                ))
                doc.themes = list(dict.fromkeys(g.strip() for g in doc.themes if g))
                doc.tags = list(dict.fromkeys(t.strip() for t in doc.tags if t))

                # ── Build embedding text ──────────────────────────────────────
                doc.build_embedding_text()

                # ── Collect statistics ────────────────────────────────────────
                for src in doc.data_sources:
                    self.stats["sources"][src] += 1
                if doc.media_type:
                    self.stats["media_types"][doc.media_type] += 1
                if doc.year:
                    self.stats["years"][doc.year] += 1
                for g in doc.genres:
                    self.stats["top_genres"][g] += 1

                clean.append(doc)
                progress.advance(task)

        self.stats["output_count"] = len(clean)
        return clean

    def report(self) -> dict:
        """Return processing statistics."""
        removed_total = (
            self.stats["removed_no_synopsis"]
            + self.stats["removed_too_short"]
            + self.stats["removed_adult"]
            + self.stats["removed_music_type"]
        )
        return {
            "input_count": self.stats["input_count"],
            "output_count": self.stats["output_count"],
            "removed_total": removed_total,
            "removed_breakdown": {
                "no_synopsis": self.stats["removed_no_synopsis"],
                "synopsis_too_short": self.stats["removed_too_short"],
                "adult_content": self.stats["removed_adult"],
                "music_type": self.stats["removed_music_type"],
            },
            "data_quality_pct": round(self.stats["output_count"] / max(self.stats["input_count"], 1) * 100, 1),
            "sources": dict(self.stats["sources"]),
            "media_types": dict(self.stats["media_types"].most_common()),
            "top_10_genres": dict(self.stats["top_genres"].most_common(10)),
            "year_range": {
                "min": min(self.stats["years"]) if self.stats["years"] else None,
                "max": max(self.stats["years"]) if self.stats["years"] else None,
            },
        }


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

def run_processing_pipeline(
    min_synopsis_tokens: int = 30,
    skip_adult: bool = True,
) -> int:
    """
    Full pipeline: merge → clean → save JSONL + Parquet + report.
    Returns count of output records.
    """
    log_info("=" * 60)
    log_info("PHASE 1: Data Processing Pipeline")
    log_info("=" * 60)

    # ── Step 1: Merge ─────────────────────────────────────────────────────────
    merger = DataMerger()
    merged_docs = merger.merge()

    # ── Step 2: Process ───────────────────────────────────────────────────────
    processor = DataProcessor()
    clean_docs = processor.process(
        merged_docs,
        min_synopsis_tokens=min_synopsis_tokens,
        skip_adult=skip_adult,
    )

    # ── Step 3: Save JSONL ────────────────────────────────────────────────────
    if MERGED_OUTPUT.exists():
        MERGED_OUTPUT.unlink()  # Fresh write (not append)

    log_info(f"Writing {len(clean_docs):,} records to {MERGED_OUTPUT}...")
    with make_progress() as progress:
        task = progress.add_task("Saving merged JSONL...", total=len(clean_docs))
        for doc in clean_docs:
            append_jsonl(MERGED_OUTPUT, doc.model_dump())
            progress.advance(task)

    # ── Step 4: Save Parquet ──────────────────────────────────────────────────
    log_info("Converting to Parquet for fast analysis...")
    rows = []
    for doc in clean_docs:
        row = doc.model_dump()
        # Flatten lists to comma-separated strings for Parquet compatibility
        for field in ["genres", "themes", "tags", "demographics", "studios", "data_sources", "title_synonyms"]:
            row[field] = ", ".join(row.get(field, []) or [])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(PARQUET_OUTPUT, index=False, compression="snappy")

    # ── Step 5: Save report ───────────────────────────────────────────────────
    report = processor.report()
    with open(REPORT_OUTPUT, "w") as f:
        json.dump(report, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    log_success("=" * 60)
    log_success(f"Processing complete!")
    log_success(f"  Input records:  {report['input_count']:,}")
    log_success(f"  Output records: {report['output_count']:,}")
    log_success(f"  Removed:        {report['removed_total']:,}")
    log_success(f"  Data quality:   {report['data_quality_pct']}%")
    log_success(f"  Sources:        {report['sources']}")
    log_success(f"  Output files:")
    log_success(f"    {MERGED_OUTPUT}")
    log_success(f"    {PARQUET_OUTPUT}")
    log_success(f"    {REPORT_OUTPUT}")
    log_success("=" * 60)

    return report["output_count"]


if __name__ == "__main__":
    run_processing_pipeline()

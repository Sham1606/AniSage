"""
Embedding Pipeline — Phase 2

Reads phase1/data/processed/anime_merged.jsonl
→ Batches embedding_text fields
→ Calls chosen EmbeddingModel
→ Saves embeddings.npy + metadata.jsonl to phase2/data/
→ Checkpoint-aware: resumes from last completed batch on re-run

Output files:
    phase2/data/embeddings.npy      shape (N, dim), float32, L2-normalised
    phase2/data/metadata.jsonl      one JSON record per anime (id, title, genres, etc.)
    phase2/data/embed_checkpoint.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase2.embeddings.embedding_models import EmbeddingModel, get_embedding_model

# Rich helpers reused from phase1
from phase1.utils.helpers import (
    log_error,
    log_info,
    log_success,
    log_warning,
    make_progress,
    read_jsonl,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
PHASE1_DATA = ROOT / "phase1" / "data" / "processed" / "anime_merged.jsonl"
PHASE2_DATA = ROOT / "phase2" / "data"

EMBEDDINGS_PATH = PHASE2_DATA / "embeddings.npy"
METADATA_PATH   = PHASE2_DATA / "metadata.jsonl"
CHECKPOINT_PATH = PHASE2_DATA / "embed_checkpoint.json"


# ── Metadata fields saved per record ─────────────────────────────────────────
# We store only what the RAG chain needs for display + filtering
METADATA_FIELDS = [
    "mal_id", "anilist_id", "title", "title_japanese",
    "genres", "themes", "tags", "media_type", "status",
    "episodes", "year", "season", "studios", "source",
    "score", "mean_score", "popularity", "image_url",
    "mal_url", "anilist_url", "synopsis", "data_sources",
]


# ── Checkpoint ────────────────────────────────────────────────────────────────

class EmbedCheckpoint:
    """Tracks which batch indices have been embedded."""

    def __init__(self, path: Path = CHECKPOINT_PATH) -> None:
        self.path = path
        self.completed_batches: set[int] = set()
        self.total_embedded: int = 0
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path) as f:
                data = json.load(f)
            self.completed_batches = set(data.get("completed_batches", []))
            self.total_embedded = data.get("total_embedded", 0)
            if self.completed_batches:
                log_info(f"Checkpoint: {self.total_embedded:,} records already embedded")

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump({
                "completed_batches": sorted(self.completed_batches),
                "total_embedded": self.total_embedded,
            }, f, indent=2)

    def is_done(self, batch_idx: int) -> bool:
        return batch_idx in self.completed_batches

    def mark_done(self, batch_idx: int, count: int) -> None:
        self.completed_batches.add(batch_idx)
        self.total_embedded += count

    def reset(self) -> None:
        self.completed_batches = set()
        self.total_embedded = 0
        if self.path.exists():
            self.path.unlink()


# ── Pipeline ──────────────────────────────────────────────────────────────────

class EmbedPipeline:
    """
    Full embedding pipeline.

    Args:
        model:          EmbeddingModel instance (SentenceTransformer or OpenAI)
        batch_size:     Records per embedding call (64 is good for CPU MiniLM)
        input_path:     JSONL source file (defaults to phase1 output)
        output_dir:     Where to save embeddings + metadata
        force_reembed:  If True, ignore checkpoint and re-embed everything
    """

    def __init__(
        self,
        model: Optional[EmbeddingModel] = None,
        batch_size: int = 64,
        input_path: Path = PHASE1_DATA,
        output_dir: Path = PHASE2_DATA,
        force_reembed: bool = False,
    ) -> None:
        self.model = model or get_embedding_model("sentence-transformer")
        self.batch_size = batch_size
        self.input_path = input_path
        self.output_dir = output_dir
        self.force_reembed = force_reembed

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = EmbedCheckpoint(output_dir / "embed_checkpoint.json")

        if force_reembed:
            log_warning("force_reembed=True — clearing checkpoint and existing embeddings")
            self.checkpoint.reset()
            for f in [EMBEDDINGS_PATH, METADATA_PATH]:
                if f.exists():
                    f.unlink()

    def _load_records(self) -> list[dict]:
        """Load all records from the JSONL file."""
        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {self.input_path}\n"
                "Run Phase 1 first: python main.py run-all"
            )
        log_info(f"Loading records from {self.input_path}...")
        records = read_jsonl(self.input_path)
        log_info(f"Loaded {len(records):,} records")
        return records

    def _extract_metadata(self, record: dict) -> dict:
        """Extract only the fields we need to store alongside the embedding."""
        meta = {}
        for field in METADATA_FIELDS:
            val = record.get(field)
            # Flatten lists → comma-separated strings for ChromaDB compatibility
            # (ChromaDB metadata values must be str, int, float, or bool)
            if isinstance(val, list):
                meta[field] = ", ".join(str(v) for v in val if v) if val else ""
            elif val is None:
                meta[field] = ""
            else:
                meta[field] = val
        return meta

    def run(self) -> tuple[np.ndarray, list[dict]]:
        """
        Execute the full pipeline.

        Returns:
            embeddings: np.ndarray shape (N, dim)
            metadata:   list of N metadata dicts
        """
        records = self._load_records()
        total = len(records)

        # Build batches
        batches = [
            records[i : i + self.batch_size]
            for i in range(0, total, self.batch_size)
        ]
        n_batches = len(batches)

        log_info(f"Model:      {self.model}")
        log_info(f"Batch size: {self.batch_size}")
        log_info(f"Batches:    {n_batches}")
        log_info(f"Dim:        {self.model.dim}")

        # OpenAI cost estimate
        if hasattr(self.model, "estimate_cost"):
            texts = [r.get("embedding_text", "") for r in records]
            cost = self.model.estimate_cost(texts)
            log_info(f"Estimated OpenAI cost: ~${cost:.4f} USD")

        # Load existing partial results
        all_embeddings: list[np.ndarray] = []
        all_metadata: list[dict] = []

        existing_embeddings = EMBEDDINGS_PATH if EMBEDDINGS_PATH.exists() else None
        existing_metadata   = METADATA_PATH   if METADATA_PATH.exists()   else None

        if existing_embeddings and existing_metadata and not self.force_reembed:
            log_info("Loading existing partial embeddings...")
            all_embeddings = [np.load(existing_embeddings)]
            all_metadata   = read_jsonl(existing_metadata)
            log_info(f"Resuming from {len(all_metadata):,} already embedded records")

        # ── Main embedding loop ───────────────────────────────────────────────
        start_time = time.monotonic()

        with make_progress() as progress:
            task = progress.add_task(
                f"Embedding with {self.model.name}...",
                total=n_batches,
            )

            for batch_idx, batch in enumerate(batches):
                if self.checkpoint.is_done(batch_idx):
                    progress.advance(task)
                    continue

                texts = [r.get("embedding_text", r.get("title", "")) for r in batch]

                try:
                    vectors = self.model.embed(texts)  # (batch_size, dim)
                    meta_batch = [self._extract_metadata(r) for r in batch]

                    all_embeddings.append(vectors)
                    all_metadata.extend(meta_batch)

                    self.checkpoint.mark_done(batch_idx, len(batch))

                    # Save incremental checkpoint every 10 batches
                    if (batch_idx + 1) % 10 == 0:
                        self._save(all_embeddings, all_metadata)
                        self.checkpoint.save()

                except Exception as e:
                    log_error(f"Batch {batch_idx} failed: {e}")
                    raise

                progress.advance(task)

        # ── Final save ────────────────────────────────────────────────────────
        final_embeddings = self._save(all_embeddings, all_metadata)
        self.checkpoint.save()

        elapsed = time.monotonic() - start_time
        rate = total / elapsed if elapsed > 0 else 0

        log_success("=" * 60)
        log_success(f"Embedding complete!")
        log_success(f"  Records embedded:  {len(all_metadata):,}")
        log_success(f"  Embedding shape:   {final_embeddings.shape}")
        log_success(f"  Time elapsed:      {elapsed:.1f}s ({rate:.0f} records/sec)")
        log_success(f"  Output: {EMBEDDINGS_PATH}")
        log_success(f"  Output: {METADATA_PATH}")
        log_success("=" * 60)

        return final_embeddings, all_metadata

    def _save(
        self,
        embedding_chunks: list[np.ndarray],
        metadata: list[dict],
    ) -> np.ndarray:
        """Concatenate all embedding chunks and save to disk."""
        if not embedding_chunks:
            return np.array([])

        combined = np.vstack(embedding_chunks).astype(np.float32)
        np.save(EMBEDDINGS_PATH, combined)

        # Write metadata JSONL fresh each time (it's fast)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            for record in metadata:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return combined


# ── Convenience runner ────────────────────────────────────────────────────────

def run_embed_pipeline(
    model_type: str = "sentence-transformer",
    model_name: Optional[str] = None,
    batch_size: int = 64,
    force_reembed: bool = False,
    openai_api_key: Optional[str] = None,
) -> tuple[np.ndarray, list[dict]]:
    """Top-level function called by CLI."""
    model = get_embedding_model(
        model_type=model_type,
        model_name=model_name,
        api_key=openai_api_key,
    )
    pipeline = EmbedPipeline(
        model=model,
        batch_size=batch_size,
        force_reembed=force_reembed,
    )
    return pipeline.run()
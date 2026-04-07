"""
Index Builder — Phase 2

Loads saved embeddings.npy + metadata.jsonl and populates:
  - FAISS     (primary — fast, reliable, production-grade)
  - ChromaDB  (optional — disabled by default, unreliable on Windows with large datasets)

Called by: python main.py build-index
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase1.utils.helpers import log_info, log_success, log_warning, read_jsonl
from phase2.vectordb.faiss_store import FAISSStore

ROOT             = Path(__file__).parent.parent.parent
EMBEDDINGS_PATH  = ROOT / "phase2" / "data" / "embeddings.npy"
METADATA_PATH    = ROOT / "phase2" / "data" / "metadata.jsonl"


def run_build_index(
    reset_chroma: bool = False,
    skip_chroma: bool = True,       # Default True — ChromaDB hangs on Windows
    skip_faiss: bool = False,
) -> dict:
    """
    Load embeddings + metadata from disk and populate vector stores.

    ChromaDB is skipped by default on Windows due to SQLite WAL hang issues.
    FAISS handles all semantic search needs for Phase 3 RAG.

    Args:
        reset_chroma: Drop and rebuild ChromaDB (only if skip_chroma=False)
        skip_chroma:  Skip ChromaDB (default True — use FAISS only)
        skip_faiss:   Skip FAISS

    Returns:
        dict with stats
    """
    log_info("=" * 60)
    log_info("Phase 2: Building Vector Indexes")
    log_info("=" * 60)

    # ── Validate input files ──────────────────────────────────────────────────
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {EMBEDDINGS_PATH}\n"
            "Run first: python main.py embed"
        )
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found at {METADATA_PATH}\n"
            "Run first: python main.py embed"
        )

    log_info(f"Loading embeddings from {EMBEDDINGS_PATH}...")
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    metadata   = read_jsonl(METADATA_PATH)

    log_info(f"Embeddings shape: {embeddings.shape}")
    log_info(f"Metadata records: {len(metadata):,}")

    assert len(embeddings) == len(metadata), \
        f"Count mismatch: {len(embeddings)} embeddings vs {len(metadata)} metadata"

    stats = {}

    # ── FAISS (primary store) ─────────────────────────────────────────────────
    if not skip_faiss:
        log_info("\n─── Building FAISS Index ───")
        faiss_store = FAISSStore()
        faiss_store.build(embeddings, metadata)
        faiss_store.save()
        stats["faiss"] = faiss_store.get_stats()
        log_success(f"FAISS: {faiss_store.count():,} vectors indexed")
        log_success(f"Index size: {stats['faiss']['index_size_mb']} MB")
    else:
        log_warning("Skipping FAISS (--skip-faiss)")

    # ── ChromaDB (optional, off by default) ───────────────────────────────────
    if not skip_chroma:
        log_info("\n─── Building ChromaDB Index ───")
        log_warning("ChromaDB can hang on Windows with large datasets.")
        log_warning("If it stalls, Ctrl+C and use FAISS only (default).")
        try:
            from phase2.vectordb.chromadb_store import ChromaStore
            chroma = ChromaStore(reset=reset_chroma)
            chroma.ingest(embeddings, metadata)
            stats["chromadb"] = chroma.get_stats()
            log_success(f"ChromaDB: {chroma.count():,} records indexed")
        except Exception as e:
            log_warning(f"ChromaDB failed: {e}")
            log_warning("Continuing without ChromaDB — FAISS handles all search.")
    else:
        log_info("ChromaDB skipped (use --no-skip-chroma to enable)")

    log_success("\n" + "=" * 60)
    log_success("Index build complete! Ready for Phase 3 RAG.")
    log_success("=" * 60)

    return stats
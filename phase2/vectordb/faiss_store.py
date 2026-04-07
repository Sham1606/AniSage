"""
FAISS Store — Phase 2

Production-grade vector index using Facebook AI Similarity Search.
14k records fit easily in RAM — use IndexFlatIP (exact search, cosine via
L2-normalised vectors).

Persists to:
    phase2/data/faiss_index/anime.index   — FAISS binary index
    phase2/data/faiss_index/id_map.json   — index_position → metadata

Usage:
    store = FAISSStore()
    store.build(embeddings, metadata_list)
    results = store.query(query_vector, k=5)
    store.save()

    # Next run:
    store = FAISSStore()
    store.load()
    results = store.query(query_vector, k=5)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase1.utils.helpers import log_info, log_success, log_warning

ROOT       = Path(__file__).parent.parent.parent
FAISS_DIR  = ROOT / "phase2" / "data" / "faiss_index"
INDEX_PATH = FAISS_DIR / "anime.index"
IDMAP_PATH = FAISS_DIR / "id_map.json"


class FAISSStore:
    """
    FAISS-backed vector store for fast anime similarity search.

    Why IndexFlatIP?
    - IP = Inner Product. On L2-normalised vectors, IP == cosine similarity.
    - "Flat" = exact brute-force search — no approximation error.
    - For 14k vectors, exact search is fast enough (<5ms per query).
    - If dataset grows to 100k+, switch to IndexIVFFlat for ANN speed.

    The id_map.json maps FAISS integer positions (0, 1, 2, ...) to full
    metadata dicts so we can return rich results from a raw index query.
    """

    def __init__(
        self,
        index_path: Path = INDEX_PATH,
        idmap_path: Path = IDMAP_PATH,
    ) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            raise ImportError(
                "faiss-cpu not installed. Run:\n"
                "  pip install faiss-cpu"
            )

        self.index_path = index_path
        self.idmap_path = idmap_path
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self._index = None        # faiss.Index — loaded lazily
        self._id_map: list[dict] = []  # position → metadata
        self._dim: Optional[int] = None

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(
        self,
        embeddings: np.ndarray,
        metadata_list: list[dict],
    ) -> None:
        """
        Build the FAISS index from scratch.

        Args:
            embeddings:    np.ndarray shape (N, dim), float32, L2-normalised
            metadata_list: list of N metadata dicts (title, genres, etc.)
        """
        import faiss

        assert len(embeddings) == len(metadata_list), \
            f"Mismatch: {len(embeddings)} embeddings vs {len(metadata_list)} metadata"
        assert embeddings.dtype == np.float32, \
            "Embeddings must be float32 — cast with .astype(np.float32)"

        n, dim = embeddings.shape
        self._dim = dim

        log_info(f"Building FAISS IndexFlatIP — {n:,} vectors × {dim} dims")

        # Ensure vectors are L2-normalised (IP on unit vectors = cosine sim)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalised = embeddings / np.maximum(norms, 1e-10)

        # Build index
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(normalised)

        self._id_map = metadata_list

        log_success(f"FAISS index built: {self._index.ntotal:,} vectors")

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist index and id_map to disk."""
        import faiss

        if self._index is None:
            raise RuntimeError("No index to save — call build() first")

        faiss.write_index(self._index, str(self.index_path))

        # Save id_map — keep metadata lean for fast load
        with open(self.idmap_path, "w", encoding="utf-8") as f:
            json.dump(self._id_map, f, ensure_ascii=False)

        index_mb = self.index_path.stat().st_size / (1024 * 1024)
        idmap_mb = self.idmap_path.stat().st_size / (1024 * 1024)
        log_success(f"FAISS saved: {self.index_path.name} ({index_mb:.1f} MB)")
        log_success(f"ID map saved: {self.idmap_path.name} ({idmap_mb:.1f} MB)")

    def load(self) -> bool:
        """
        Load a previously saved index from disk.
        Returns True if successful, False if no saved index exists.
        """
        import faiss

        if not self.index_path.exists() or not self.idmap_path.exists():
            return False

        self._index = faiss.read_index(str(self.index_path))

        with open(self.idmap_path, "r", encoding="utf-8") as f:
            self._id_map = json.load(f)

        self._dim = self._index.d
        log_info(
            f"FAISS index loaded: {self._index.ntotal:,} vectors × {self._dim} dims"
        )
        return True

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> list[dict]:
        """
        Find the k most similar anime to the query vector.

        Args:
            query_embedding: 1-D array of shape (dim,) or 2-D (1, dim)
            k:               Number of results

        Returns:
            list of result dicts with keys from metadata + _rank, _score
        """
        if self._index is None:
            raise RuntimeError(
                "Index not loaded. Call build() or load() first."
            )

        # Normalise query vector
        q = query_embedding.astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        q_norm = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-10)

        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(q_norm, k)

        results = []
        for rank, (score, idx) in enumerate(
            zip(scores[0].tolist(), indices[0].tolist())
        ):
            if idx < 0 or idx >= len(self._id_map):
                continue  # FAISS returns -1 for unfilled slots

            result = dict(self._id_map[idx])
            result["_rank"]  = rank + 1
            result["_score"] = round(float(score), 4)  # cosine similarity (0-1)
            results.append(result)

        return results

    # ── Stats ─────────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._index.ntotal if self._index else 0

    def is_loaded(self) -> bool:
        return self._index is not None

    def get_stats(self) -> dict:
        return {
            "total_vectors": self.count(),
            "dimensions": self._dim,
            "index_path": str(self.index_path),
            "index_size_mb": (
                round(self.index_path.stat().st_size / (1024 * 1024), 2)
                if self.index_path.exists() else None
            ),
        }
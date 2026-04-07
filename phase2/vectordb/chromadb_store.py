"""
ChromaDB Store — Phase 2

Persistent local ChromaDB collection for development.
Supports semantic search + metadata filtering (genre, year, score, type).

Collection: anime_embeddings
Stored at:  phase2/data/chroma_db/

Usage:
    store = ChromaStore()
    store.ingest(embeddings, metadata_list)
    results = store.query("dark psychological thriller", n_results=5)
    results = store.query("romance drama", where={"year": {"$gte": 2010}})
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase1.utils.helpers import log_info, log_success, log_warning, make_progress

ROOT       = Path(__file__).parent.parent.parent
CHROMA_DIR = ROOT / "phase2" / "data" / "chroma_db"
COLLECTION_NAME = "anime_embeddings"


class ChromaStore:
    """
    Wrapper around ChromaDB for anime semantic search.

    ChromaDB stores:
      - The raw embedding vector
      - A metadata dict (title, genres, year, score, etc.)
      - A document string (we use embedding_text for this)

    This lets us do:
      1. Pure vector search: "find me anime like Cowboy Bebop"
      2. Filtered search:    "find action anime from the 2000s with score > 8"
    """

    def __init__(
        self,
        persist_dir: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        reset: bool = False,
    ) -> None:
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb not installed. Run:\n"
                "  pip install chromadb"
            )

        import chromadb

        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # Persistent client — data survives restarts
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        if reset:
            log_warning(f"Resetting ChromaDB collection: {collection_name}")
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass

        # Get or create the collection.
        # IMPORTANT: embedding_function=None tells ChromaDB we are supplying
        # our own pre-computed vectors — this prevents it from trying to load
        # its default ONNX/onnxruntime embedding model (which causes a DLL
        # error on Windows if onnxruntime is not installed).
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        log_info(
            f"ChromaDB collection '{collection_name}' "
            f"— {self.collection.count():,} records at {persist_dir}"
        )

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(
        self,
        embeddings: np.ndarray,
        metadata_list: list[dict],
        batch_size: int = 50,
        skip_existing: bool = True,
    ) -> int:
        """
        Insert embeddings + metadata into the collection.

        Uses small batches (50) to avoid ChromaDB SQLite WAL hangs on Windows.
        Embeddings are passed as Python lists (not numpy) — ChromaDB requirement.

        Args:
            embeddings:    np.ndarray shape (N, dim)
            metadata_list: list of N metadata dicts
            batch_size:    ChromaDB upsert batch size (keep <= 100 on Windows)
            skip_existing: If True, skip records already in the collection

        Returns: number of records inserted
        """
        import time as _time

        assert len(embeddings) == len(metadata_list), \
            f"Mismatch: {len(embeddings)} embeddings vs {len(metadata_list)} metadata"

        existing_count = self.collection.count()

        if skip_existing and existing_count > 0:
            log_info(f"Collection already has {existing_count:,} records — skipping existing")
            start_idx = existing_count
            embeddings    = embeddings[start_idx:]
            metadata_list = metadata_list[start_idx:]
            if len(embeddings) == 0:
                log_info("All records already ingested — nothing to do")
                return 0

        total = len(embeddings)
        log_info(f"Ingesting {total:,} records into ChromaDB (batch={batch_size})...")

        inserted = 0
        offset = existing_count if skip_existing else 0

        # Pre-clean ALL metadata before the loop (avoids per-batch overhead)
        all_clean_meta = []
        for meta in metadata_list:
            clean = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                elif v is None:
                    clean[k] = ""
                else:
                    clean[k] = str(v)
            all_clean_meta.append(clean)

        with make_progress() as progress:
            task = progress.add_task("Ingesting into ChromaDB...", total=total)

            for i in range(0, total, batch_size):
                batch_emb   = embeddings[i : i + batch_size]
                batch_meta  = all_clean_meta[i : i + batch_size]
                batch_orig  = metadata_list[i : i + batch_size]

                # IDs must be unique strings
                ids = []
                for j, meta in enumerate(batch_orig):
                    mid = meta.get("mal_id") or meta.get("anilist_id")
                    ids.append(str(mid) if mid else f"idx_{offset + i + j}")

                documents = [
                    m.get("synopsis", m.get("title", ""))[:500]  # cap at 500 chars
                    for m in batch_orig
                ]

                # Convert numpy slice to plain Python list of lists
                # ChromaDB REQUIRES Python lists, not numpy arrays
                emb_list = batch_emb.tolist()

                try:
                    self.collection.upsert(
                        ids=ids,
                        embeddings=emb_list,
                        metadatas=batch_meta,
                        documents=documents,
                    )
                except Exception as e:
                    log_warning(f"Batch {i//batch_size} failed ({e}) — retrying one-by-one")
                    # Fallback: insert records one at a time to isolate bad entries
                    for k in range(len(ids)):
                        try:
                            self.collection.upsert(
                                ids=[ids[k]],
                                embeddings=[emb_list[k]],
                                metadatas=[batch_meta[k]],
                                documents=[documents[k]],
                            )
                            inserted += 1
                        except Exception as e2:
                            log_warning(f"Skipping record {ids[k]}: {e2}")
                    progress.advance(task, len(ids))
                    _time.sleep(0.1)
                    continue

                inserted += len(batch_emb)
                progress.advance(task, len(batch_emb))

                # Small sleep every 10 batches to let SQLite WAL flush on Windows
                if (i // batch_size) % 10 == 0:
                    _time.sleep(0.1)

        log_success(f"ChromaDB ingestion complete: {inserted:,} records inserted")
        log_success(f"Total collection size: {self.collection.count():,}")
        return inserted

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        n_results: int = 5,
        where: Optional[dict] = None,
        include_embeddings: bool = False,
    ) -> list[dict]:
        """
        Semantic search against the collection.

        Either query_text OR query_embedding must be provided.
        If query_text is given, it must be embedded first — pass the
        embedding model separately and use query_embedding for efficiency.

        Args:
            query_text:        Raw text (not recommended — no model here)
            query_embedding:   Pre-computed query vector (preferred)
            n_results:         Number of results to return
            where:             Metadata filter dict (ChromaDB filter syntax)
                               Examples:
                                 {"year": {"$gte": 2000}}
                                 {"media_type": "TV"}
                                 {"score": {"$gte": 8.0}}
            include_embeddings: Include raw vectors in results

        Returns:
            list of result dicts with keys:
              id, title, score (similarity), distance, genres, year,
              synopsis, mal_id, anilist_id, image_url, ...
        """
        if query_embedding is None and query_text is None:
            raise ValueError("Either query_text or query_embedding must be provided")

        include = ["metadatas", "distances", "documents"]
        if include_embeddings:
            include.append("embeddings")

        kwargs: dict[str, Any] = {
            "n_results": min(n_results, self.collection.count()),
            "include": include,
        }

        if query_embedding is not None:
            kwargs["query_embeddings"] = [query_embedding.tolist()]
        else:
            # query_texts would trigger ChromaDB's internal embedding function
            # (ONNX) which fails on Windows without onnxruntime. Raise clearly.
            raise ValueError(
                "query_text is not supported — pass query_embedding instead.\n"
                "Embed your query first: embedding = model.embed_one(text)"
            )

        if where:
            kwargs["where"] = where

        try:
            results = self.collection.query(**kwargs)
        except Exception as e:
            log_warning(f"ChromaDB query failed: {e}")
            return []

        # Flatten ChromaDB's nested response format
        output = []
        ids       = results.get("ids", [[]])[0]
        metas     = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        documents = results.get("documents", [[]])[0]

        for i, (rid, meta, dist, doc) in enumerate(
            zip(ids, metas, distances, documents)
        ):
            result = dict(meta)
            result["_id"]         = rid
            result["_rank"]       = i + 1
            result["_distance"]   = round(float(dist), 4)
            result["_similarity"] = round(1 - float(dist), 4)  # cosine: dist=1-sim
            output.append(result)

        return output

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return collection statistics."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_records": count,
            "persist_dir": str(self.persist_dir),
        }

    def count(self) -> int:
        return self.collection.count()

    def reset_collection(self) -> None:
        """Drop and recreate the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log_warning(f"Collection '{self.collection_name}' has been reset")
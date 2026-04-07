"""
Anime Retriever — Phase 3

Wraps Phase 2's FAISSStore + EmbeddingModel into a single clean interface
for the RAG chain to call.

Usage:
    retriever = AnimeRetriever()
    candidates = retriever.retrieve("dark psychological thriller", k=10)
    # Returns list of anime dicts ready for the LLM prompt
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase1.utils.helpers import log_info, log_warning
from phase2.embeddings.embedding_models import EmbeddingModel, get_embedding_model
from phase2.vectordb.faiss_store import FAISSStore

ROOT = Path(__file__).parent.parent.parent


class AnimeRetriever:
    """
    Single-call interface: text query → ranked anime candidates.

    Handles:
      - Lazy-loading the FAISS index (loaded once, reused across calls)
      - Embedding queries with the same model used during Phase 2
      - Optional score threshold filtering
      - Optional metadata post-filtering (type, year, score)
    """

    def __init__(
        self,
        model_type: str = "sentence-transformer",
        model_name: str = "all-MiniLM-L6-v2",
        faiss_index_dir: Optional[Path] = None,
    ) -> None:
        self.model: EmbeddingModel = get_embedding_model(model_type, model_name)

        index_dir = faiss_index_dir or ROOT / "phase2" / "data" / "faiss_index"
        self.store = FAISSStore(
            index_path=index_dir / "anime.index",
            idmap_path=index_dir / "id_map.json",
        )

        # Load index once at startup
        loaded = self.store.load()
        if not loaded:
            raise RuntimeError(
                f"FAISS index not found at {index_dir}\n"
                "Run first: python main.py build-index"
            )
        log_info(f"Retriever ready: {self.store.count():,} anime indexed")

    def retrieve(
        self,
        query: str,
        k: int = 10,
        min_score: Optional[float] = None,
        filter_type: Optional[str] = None,
        filter_min_year: Optional[int] = None,
        filter_max_year: Optional[int] = None,
        filter_min_mal_score: Optional[float] = None,
        exclude_titles: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Retrieve the top-k most relevant anime for a query.

        Args:
            query:               Natural language search query
            k:                   Number of candidates to retrieve (fetch more, then filter)
            min_score:           Minimum cosine similarity threshold (0-1)
            filter_type:         Only return this media type ("TV", "Movie", "OVA", etc.)
            filter_min_year:     Earliest release year
            filter_max_year:     Latest release year
            filter_min_mal_score: Minimum MAL score (0-10)
            exclude_titles:      List of title strings to exclude from results

        Returns:
            List of candidate dicts with metadata + _rank, _score fields
        """
        # Fetch more than k to allow for post-filtering
        fetch_k = min(k * 3, self.store.count())

        # Embed the query
        query_vec = self.model.embed_one(query)

        # FAISS search
        raw_results = self.store.query(query_vec, k=fetch_k)

        # ── Post-filtering ────────────────────────────────────────────────────
        filtered = []
        exclude_lower = {t.lower() for t in (exclude_titles or [])}

        for result in raw_results:
            # Similarity threshold
            if min_score and result.get("_score", 0) < min_score:
                continue

            # Title exclusion
            title = result.get("title", "")
            if title.lower() in exclude_lower:
                continue

            # Media type filter
            if filter_type:
                result_type = result.get("media_type", "")
                if result_type.lower() != filter_type.lower():
                    continue

            # Year filter
            result_year = result.get("year")
            if result_year:
                try:
                    y = int(result_year)
                    if filter_min_year and y < filter_min_year:
                        continue
                    if filter_max_year and y > filter_max_year:
                        continue
                except (ValueError, TypeError):
                    pass

            # MAL score filter
            if filter_min_mal_score:
                mal_score = result.get("score") or result.get("mean_score")
                if mal_score:
                    try:
                        if float(mal_score) < filter_min_mal_score:
                            continue
                    except (ValueError, TypeError):
                        pass

            filtered.append(result)

            if len(filtered) >= k:
                break

        # Re-rank by score and reassign _rank
        filtered.sort(key=lambda x: x.get("_score", 0), reverse=True)
        for i, r in enumerate(filtered):
            r["_rank"] = i + 1

        return filtered

    def retrieve_multi_query(
        self,
        queries: list[str],
        k_per_query: int = 5,
        deduplicate: bool = True,
    ) -> list[dict]:
        """
        Retrieve candidates for multiple queries and merge results.
        Useful for complex requests: "action anime like Cowboy Bebop but more serious"
        → queries: ["action space anime", "serious character drama science fiction"]

        Args:
            queries:       List of query strings
            k_per_query:   Candidates per query
            deduplicate:   Remove duplicate titles (keep highest score)

        Returns:
            Merged, deduplicated, re-ranked candidate list
        """
        all_results: list[dict] = []
        for query in queries:
            results = self.retrieve(query, k=k_per_query)
            all_results.extend(results)

        if not deduplicate:
            return all_results

        # Deduplicate by title — keep highest scoring occurrence
        seen: dict[str, dict] = {}
        for result in all_results:
            title = result.get("title", "")
            if title not in seen or result.get("_score", 0) > seen[title].get("_score", 0):
                seen[title] = result

        merged = sorted(seen.values(), key=lambda x: x.get("_score", 0), reverse=True)
        for i, r in enumerate(merged):
            r["_rank"] = i + 1

        return merged
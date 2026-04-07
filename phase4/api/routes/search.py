"""
Search & Anime Router — Phase 4

Endpoints:
  POST /search            → Direct FAISS semantic search (no LLM)
  GET  /anime/{mal_id}    → Get anime details by MAL ID
  GET  /anime/random      → Get a random anime from the index
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase4.api.middleware.rate_limiter import rate_limit_global
from phase4.api.models.schemas import AnimeResult, SearchRequest, SearchResponse

router = APIRouter(tags=["Search & Anime"])

# ── Shared retriever (lazy-loaded singleton) ──────────────────────────────────

_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        from phase3.retrieval.anime_retriever import AnimeRetriever
        _retriever = AnimeRetriever()
    return _retriever


# ── POST /search ──────────────────────────────────────────────────────────────

@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic anime search",
    description=(
        "Search the anime database using natural language. "
        "Returns ranked results based on semantic similarity — no LLM involved. "
        "Much faster than /chat, ideal for autocomplete or browse features."
    ),
    dependencies=[Depends(rate_limit_global)],
)
async def search(body: SearchRequest) -> SearchResponse:
    """
    Direct FAISS semantic search — returns raw results without LLM generation.

    Use this for:
    - Quick search bars / autocomplete
    - Browse by genre/year
    - Testing retrieval quality independently of the LLM

    For conversational recommendations with explanations, use POST /chat instead.
    """
    retriever = get_retriever()

    try:
        raw_results = retriever.retrieve(
            query=body.query,
            k=body.k,
            filter_type=body.filter_type,
            filter_min_year=body.filter_min_year,
            filter_max_year=body.filter_max_year,
            filter_min_mal_score=body.filter_min_score,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)[:200]}",
        )

    results = [AnimeResult.from_faiss_result(r) for r in raw_results]

    return SearchResponse(
        query=body.query,
        results=results,
        total=len(results),
    )


# ── GET /anime/random ─────────────────────────────────────────────────────────

@router.get(
    "/anime/random",
    response_model=AnimeResult,
    summary="Get a random anime",
    description="Returns a random anime from the database. Useful for discovery / surprise me features.",
    dependencies=[Depends(rate_limit_global)],
)
async def get_random_anime(
    min_score: Optional[float] = Query(None, ge=0, le=10, description="Minimum MAL score filter"),
    media_type: Optional[str]  = Query(None, description="Filter by type: TV, Movie, OVA, etc."),
) -> AnimeResult:
    """Return a random anime, optionally filtered by score or type."""
    retriever = get_retriever()
    id_map = retriever.store._id_map

    if not id_map:
        raise HTTPException(status_code=503, detail="Anime index not loaded")

    # Apply filters
    pool = id_map
    if min_score is not None:
        filtered = []
        for entry in pool:
            s = entry.get("score") or entry.get("mean_score")
            try:
                if s and float(s) >= min_score:
                    filtered.append(entry)
            except (ValueError, TypeError):
                pass
        pool = filtered

    if media_type is not None:
        pool = [e for e in pool if e.get("media_type", "").lower() == media_type.lower()]

    if not pool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No anime found matching the given filters",
        )

    entry = random.choice(pool)
    result = dict(entry)
    result["_rank"]  = 1
    result["_score"] = 1.0
    return AnimeResult.from_faiss_result(result)


# ── GET /anime/{mal_id} ───────────────────────────────────────────────────────

@router.get(
    "/anime/{mal_id}",
    response_model=AnimeResult,
    summary="Get anime details by MAL ID",
    description="Look up a specific anime by its MyAnimeList ID.",
    dependencies=[Depends(rate_limit_global)],
)
async def get_anime(mal_id: int) -> AnimeResult:
    """
    Retrieve details for a specific anime by MAL ID.

    Returns the full metadata: title, genres, synopsis, score, images, links.
    MAL IDs can be found in search results or from myanimelist.net URLs.
    """
    retriever = get_retriever()

    # Search the id_map for this MAL ID
    id_map = retriever.store._id_map
    if not id_map:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Anime index not loaded",
        )

    # Linear scan — fast enough for 14k entries
    for entry in id_map:
        entry_mal_id = entry.get("mal_id")
        try:
            if entry_mal_id and int(entry_mal_id) == mal_id:
                # Build a fake result dict for the schema
                result = dict(entry)
                result["_rank"]  = 1
                result["_score"] = 1.0
                return AnimeResult.from_faiss_result(result)
        except (ValueError, TypeError):
            continue

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Anime with MAL ID {mal_id} not found in the index",
    )
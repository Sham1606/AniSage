"""
API Models — Phase 4

Pydantic v2 models for all request/response schemas.
Used by FastAPI for automatic validation and OpenAPI docs generation.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """POST /chat request body."""
    message:    str            = Field(..., min_length=1, max_length=2000,
                                       description="User's message or anime request")
    session_id: Optional[str]  = Field(None, description="Session ID — omit to start a new session")

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be blank")
        return v.strip()

    model_config = {"json_schema_extra": {
        "examples": [{
            "message": "I want dark psychological anime with complex characters",
            "session_id": None,
        }]
    }}


class ChatResponse(BaseModel):
    """POST /chat response body (non-streaming)."""
    session_id: str   = Field(..., description="Use this in subsequent requests to continue the conversation")
    message:    str   = Field(..., description="AniSage's recommendation response")
    turn:       int   = Field(..., description="Turn number in this session")
    is_new_session: bool = Field(..., description="True if this was the first message in a new session")


class StreamChunk(BaseModel):
    """Individual chunk in a streaming response."""
    delta:      str  = Field(..., description="Text chunk")
    done:       bool = Field(False, description="True on the final chunk")
    session_id: str  = Field(..., description="Session ID for this conversation")


# ── Search ────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """POST /search request body — direct FAISS search without LLM."""
    query:           str            = Field(..., min_length=1, max_length=500)
    k:               int            = Field(5, ge=1, le=20, description="Number of results")
    filter_type:     Optional[str]  = Field(None, description="Media type filter: TV, Movie, OVA, etc.")
    filter_min_year: Optional[int]  = Field(None, ge=1900, le=2030)
    filter_max_year: Optional[int]  = Field(None, ge=1900, le=2030)
    filter_min_score:Optional[float]= Field(None, ge=0.0, le=10.0, description="Minimum MAL score")

    model_config = {"json_schema_extra": {
        "examples": [{
            "query": "samurai revenge feudal japan",
            "k": 5,
            "filter_type": "TV",
            "filter_min_score": 7.0,
        }]
    }}


class AnimeResult(BaseModel):
    """A single anime search result."""
    rank:         int            = Field(..., description="Result rank (1 = most similar)")
    score:        float          = Field(..., description="Cosine similarity score (0-1)")
    mal_id:       Optional[int]  = None
    title:        str            = ""
    year:         Optional[int]  = None
    media_type:   str            = ""
    mal_score:    Optional[float]= Field(None, description="MyAnimeList community score (0-10)")
    genres:       str            = ""
    synopsis:     str            = ""
    image_url:    str            = ""
    mal_url:      str            = ""

    @classmethod
    def from_faiss_result(cls, r: dict) -> "AnimeResult":
        """Build from a raw FAISS result dict."""
        mal_score = r.get("score") or r.get("mean_score")
        try:
            mal_score = float(mal_score) if mal_score else None
        except (ValueError, TypeError):
            mal_score = None

        year = r.get("year")
        try:
            year = int(year) if year else None
        except (ValueError, TypeError):
            year = None

        mal_id = r.get("mal_id")
        try:
            mal_id = int(mal_id) if mal_id else None
        except (ValueError, TypeError):
            mal_id = None

        return cls(
            rank=r.get("_rank", 0),
            score=round(float(r.get("_score", 0)), 4),
            mal_id=mal_id,
            title=r.get("title", ""),
            year=year,
            media_type=r.get("media_type", ""),
            mal_score=mal_score,
            genres=r.get("genres", ""),
            synopsis=(r.get("synopsis", "") or "")[:300],
            image_url=r.get("image_url", "") or "",
            mal_url=r.get("mal_url", "") or "",
        )


class SearchResponse(BaseModel):
    """POST /search response."""
    query:   str              = Field(..., description="The original query")
    results: list[AnimeResult] = Field(..., description="Ranked anime results")
    total:   int              = Field(..., description="Number of results returned")


# ── Session ───────────────────────────────────────────────────────────────────

class SessionInfo(BaseModel):
    """GET /session/{session_id} response."""
    session_id:   str            = ""
    turn_count:   int            = 0
    exists:       bool           = False
    profile:      Optional[str]  = None   # preference profile summary


class ResetResponse(BaseModel):
    """POST /session/{session_id}/reset response."""
    session_id: str  = ""
    message:    str  = "Session reset successfully"


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """GET /health response."""
    status:           str  = "ok"
    faiss_loaded:     bool = False
    faiss_vectors:    int  = 0
    active_sessions:  int  = 0
    llm_backend:      str  = ""
    llm_model:        str  = ""


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error response."""
    error:   str = ""
    detail:  str = ""
    code:    int = 400
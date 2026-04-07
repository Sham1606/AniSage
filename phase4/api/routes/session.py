"""
Session Router — Phase 4

Endpoints:
  GET    /session/{session_id}         → Get session info
  POST   /session/{session_id}/reset   → Reset conversation history
  DELETE /session/{session_id}         → Delete session
  GET    /sessions/stats               → Manager stats (admin)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from phase4.api.core.session_manager import SessionManager, get_session_manager
from phase4.api.models.schemas import ResetResponse, SessionInfo

router = APIRouter(prefix="/session", tags=["Sessions"])


@router.get(
    "/{session_id}",
    response_model=SessionInfo,
    summary="Get session info",
)
async def get_session(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager),
) -> SessionInfo:
    """Return info about an existing session — turn count, preference profile."""
    chain = manager.get(session_id)
    if chain is None:
        return SessionInfo(session_id=session_id, exists=False)

    return SessionInfo(
        session_id=session_id,
        turn_count=chain.memory.turn_count,
        exists=True,
        profile=chain.memory.profile.to_summary() or None,
    )


@router.post(
    "/{session_id}/reset",
    response_model=ResetResponse,
    summary="Reset conversation history",
    description="Clears turn history and preference profile. Session ID remains valid.",
)
async def reset_session(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager),
) -> ResetResponse:
    """Reset a session to a fresh state without losing the session ID."""
    found = manager.reset(session_id)
    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )
    return ResetResponse(
        session_id=session_id,
        message="Session reset — conversation history and profile cleared",
    )


@router.delete(
    "/{session_id}",
    summary="Delete a session",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_session(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager),
) -> None:
    """Delete a session and free its memory."""
    manager.delete(session_id)


@router.get(
    "s/stats",
    summary="Session manager statistics",
    description="Returns active session count and capacity info.",
)
async def session_stats(
    manager: SessionManager = Depends(get_session_manager),
) -> dict:
    return manager.stats()
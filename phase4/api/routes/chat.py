"""
Chat Router — Phase 4

Endpoints:
  POST /chat          → Non-streaming chat (returns full response)
  POST /chat/stream   → Server-Sent Events streaming chat
  GET  /chat/models   → List available LLM models
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from phase4.api.core.config import settings
from phase4.api.core.session_manager import SessionManager, get_session_manager
from phase4.api.middleware.rate_limiter import rate_limit_chat, rate_limit_global
from phase4.api.models.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
)

router = APIRouter(prefix="/chat", tags=["Chat"])


# ── POST /chat — non-streaming ────────────────────────────────────────────────

@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a message to AniSage",
    description="Send an anime request or follow-up. Returns the full response at once.",
    dependencies=[Depends(rate_limit_global), Depends(rate_limit_chat)],
)
async def chat(
    body: ChatRequest,
    manager: SessionManager = Depends(get_session_manager),
) -> ChatResponse:
    """
    Non-streaming chat endpoint.

    - If session_id is omitted, a new session is created and returned
    - Pass session_id in subsequent requests to continue the conversation
    - Each session maintains its own conversation history and preference profile
    """
    # Get or create session
    sid = body.session_id or manager.new_session_id()
    chain, is_new = manager.get_or_create(sid)

    try:
        # Run the RAG chain in a thread pool (it's sync)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: chain.chat(body.message, stream=False),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM error: {str(e)[:200]}",
        )

    return ChatResponse(
        session_id=sid,
        message=response,
        turn=chain.memory.turn_count,
        is_new_session=is_new,
    )


# ── POST /chat/stream — Server-Sent Events ────────────────────────────────────

@router.post(
    "/stream",
    summary="Streaming chat via Server-Sent Events",
    description=(
        "Returns a text/event-stream response. "
        "Each SSE event is a JSON object with 'delta', 'done', and 'session_id' fields. "
        "The final event has done=true."
    ),
    dependencies=[Depends(rate_limit_global), Depends(rate_limit_chat)],
)
async def chat_stream(
    body: ChatRequest,
    manager: SessionManager = Depends(get_session_manager),
) -> StreamingResponse:
    """
    Streaming chat endpoint using Server-Sent Events.

    Client usage (JavaScript):
        const es = new EventSource('/chat/stream');
        // or with fetch for POST:
        const resp = await fetch('/chat/stream', {method:'POST', body: JSON.stringify({message:'...'}), headers:{...}});
        const reader = resp.body.getReader();
        // read chunks and parse SSE lines

    Each SSE event:
        data: {"delta": "text chunk", "done": false, "session_id": "uuid"}

    Final event:
        data: {"delta": "", "done": true, "session_id": "uuid"}
    """
    sid = body.session_id or manager.new_session_id()
    chain, is_new = manager.get_or_create(sid)

    async def event_generator() -> AsyncIterator[str]:
        """Yield SSE-formatted chunks from the RAG chain."""
        try:
            loop = asyncio.get_event_loop()
            queue: asyncio.Queue = asyncio.Queue()

            # Run sync streaming generator in thread pool, push chunks to async queue
            def run_stream():
                try:
                    for chunk in chain.chat_stream(body.message):
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, Exception(str(e)))
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

            # Run in thread pool
            loop.run_in_executor(None, run_stream)

            # Yield chunks as SSE events
            while True:
                item = await queue.get()

                if item is None:
                    # Stream finished — send done event
                    yield f"data: {json.dumps({'delta': '', 'done': True, 'session_id': sid})}\n\n"
                    break
                elif isinstance(item, Exception):
                    error_data = json.dumps({
                        "error": str(item),
                        "done": True,
                        "session_id": sid,
                    })
                    yield f"data: {error_data}\n\n"
                    break
                else:
                    chunk_data = json.dumps({
                        "delta":      item,
                        "done":       False,
                        "session_id": sid,
                    })
                    yield f"data: {chunk_data}\n\n"

        except asyncio.CancelledError:
            # Client disconnected — clean up gracefully
            return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",    # disable nginx buffering
            "Access-Control-Allow-Origin": "*",
            "X-Session-Id":                sid,
        },
    )


# ── GET /chat/models ──────────────────────────────────────────────────────────

@router.get(
    "/models",
    summary="List available LLM models",
)
async def list_models() -> dict:
    """Return available models for the configured backend."""
    from phase3.chains.rag_chain import BACKENDS
    backend = settings.LLM_BACKEND
    config  = BACKENDS.get(backend, {})
    return {
        "backend":       backend,
        "current_model": settings.LLM_MODEL,
        "available":     config.get("models", []),
    }
"""
Phase 4 — FastAPI Application (phase4/api/app.py)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

# ── Must set these BEFORE any torch/sentence-transformers import ──────────────
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # CPU only
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from phase4.api.core.config import settings
from phase4.api.core.session_manager import get_session_manager
from phase4.api.models.schemas import HealthResponse
from phase4.api.routes import chat, search, session


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    errors = settings.validate()
    if errors:
        print(f"[WARNING] Config issues: {errors}")
    try:
        from phase4.api.routes.search import get_retriever
        retriever = get_retriever()
        print(f"[AniSage] FAISS loaded: {retriever.store.count():,} anime")
        print(f"[AniSage] Backend: {settings.LLM_BACKEND} / {settings.LLM_MODEL}")
        print(f"[AniSage] Docs: http://localhost:{settings.API_PORT}/docs")
    except Exception as e:
        print(f"[WARNING] Failed to pre-load FAISS: {e}")
    yield
    # Shutdown
    print("[AniSage] Shutting down...")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        lifespan=lifespan,
        title="AniSage API",
        description=(
            "AI-powered anime recommendation API.\n\n"
            "Uses FAISS semantic search + Groq LLM to provide personalised "
            "anime recommendations through a conversational interface.\n\n"
            "**Quick start:**\n"
            "1. `POST /chat` with `{\"message\": \"I want dark psychological anime\"}`\n"
            "2. Use the returned `session_id` in follow-up requests\n"
            "3. For streaming responses, use `POST /chat/stream`"
        ),
        version="0.4.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        response.headers["X-Process-Time"] = f"{time.monotonic() - start:.3f}s"
        return response

    app.include_router(chat.router)
    app.include_router(search.router)
    app.include_router(session.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/docs")

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health() -> HealthResponse:
        manager = get_session_manager()
        faiss_loaded, faiss_vectors = False, 0
        try:
            from phase4.api.routes.search import get_retriever
            retriever = get_retriever()
            faiss_loaded  = retriever.store.is_loaded()
            faiss_vectors = retriever.store.count()
        except Exception:
            pass
        return HealthResponse(
            status="ok" if faiss_loaded else "degraded",
            faiss_loaded=faiss_loaded,
            faiss_vectors=faiss_vectors,
            active_sessions=manager.stats()["active_sessions"],
            llm_backend=settings.LLM_BACKEND,
            llm_model=settings.LLM_MODEL,
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "phase4.api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
    )
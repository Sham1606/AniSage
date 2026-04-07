"""
Phase 4 Configuration

All settings loaded from environment variables / .env file.
Provides a single config object imported everywhere in the API.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Load .env automatically
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent.parent.parent.parent / ".env"
    if _env.exists():
        load_dotenv(_env, encoding="utf-8-sig")
except ImportError:
    pass


class Settings:
    # ── LLM Backend ───────────────────────────────────────────────────────────
    LLM_BACKEND:     str   = os.environ.get("LLM_BACKEND",   "groq")
    GROQ_API_KEY:    str   = os.environ.get("GROQ_API_KEY",  "")
    OPENAI_API_KEY:  str   = os.environ.get("OPENAI_API_KEY","")
    LLM_MODEL:       str   = os.environ.get("LLM_MODEL",     "llama-3.3-70b-versatile")
    LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.7"))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    RETRIEVAL_K:     int   = int(os.environ.get("RETRIEVAL_K", "10"))

    # ── API Server ────────────────────────────────────────────────────────────
    API_HOST:        str   = os.environ.get("API_HOST", "0.0.0.0")
    API_PORT:        int   = int(os.environ.get("API_PORT", "8000"))
    API_RELOAD:      bool  = os.environ.get("API_RELOAD", "true").lower() == "true"

    # ── Rate limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_RPM:  int   = int(os.environ.get("RATE_LIMIT_RPM", "30"))   # requests per minute per IP
    RATE_LIMIT_CHAT: int   = int(os.environ.get("RATE_LIMIT_CHAT", "20"))  # chat turns per minute per session

    # ── Session ───────────────────────────────────────────────────────────────
    SESSION_TTL_SECONDS: int = int(os.environ.get("SESSION_TTL_SECONDS", str(60 * 60 * 2)))  # 2 hours
    MAX_SESSIONS:        int = int(os.environ.get("MAX_SESSIONS", "500"))

    # ── CORS ──────────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:8501,http://127.0.0.1:8501"
    ).split(",")

    # ── Paths ─────────────────────────────────────────────────────────────────
    ROOT: Path = Path(__file__).parent.parent.parent.parent

    @classmethod
    def get_api_key(cls) -> str:
        """Return the active API key based on configured backend."""
        if cls.LLM_BACKEND == "groq":
            return cls.GROQ_API_KEY
        return cls.OPENAI_API_KEY

    @classmethod
    def validate(cls) -> list[str]:
        """Return list of validation errors (empty = all good)."""
        errors = []
        key = cls.get_api_key()
        if not key:
            env_var = "GROQ_API_KEY" if cls.LLM_BACKEND == "groq" else "OPENAI_API_KEY"
            errors.append(f"{env_var} is not set")
        faiss_index = cls.ROOT / "phase2" / "data" / "faiss_index" / "anime.index"
        if not faiss_index.exists():
            errors.append(f"FAISS index not found at {faiss_index} — run: python main.py build-index")
        return errors


settings = Settings()
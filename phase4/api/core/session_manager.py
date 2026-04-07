"""
Session Manager — Phase 4

Manages RAG chain instances per session.
Each session_id maps to one AnimeRAGChain instance with its own
conversation memory and preference profile.

Sessions expire after TTL (default 2 hours) and are cleaned up automatically.
No Redis required — pure in-memory for Phase 4 (Redis optional for Phase 6 scaling).

Usage:
    manager = get_session_manager()       # singleton
    chain = manager.get_or_create(sid)    # creates new or returns existing
    manager.delete(sid)                   # explicit cleanup
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Optional

from phase4.api.core.config import settings


class SessionEntry:
    """Wraps a RAG chain with last-accessed timestamp."""
    __slots__ = ("chain", "created_at", "last_accessed", "turn_count")

    def __init__(self, chain) -> None:
        self.chain         = chain
        self.created_at    = time.time()
        self.last_accessed = time.time()
        self.turn_count    = 0

    def touch(self) -> None:
        self.last_accessed = time.time()

    def is_expired(self, ttl: int) -> bool:
        return (time.time() - self.last_accessed) > ttl


class SessionManager:
    """
    Thread-safe in-memory session store.

    - Sessions created lazily on first /chat call
    - Background thread cleans up expired sessions every 5 minutes
    - Hard cap at MAX_SESSIONS — oldest sessions evicted when full
    """

    def __init__(
        self,
        ttl:          int = settings.SESSION_TTL_SECONDS,
        max_sessions: int = settings.MAX_SESSIONS,
    ) -> None:
        self._sessions: dict[str, SessionEntry] = {}
        self._lock      = threading.RLock()
        self.ttl        = ttl
        self.max_sessions = max_sessions

        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="session-cleanup"
        )
        self._cleanup_thread.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_or_create(self, session_id: str) -> tuple[any, bool]:
        """
        Get existing session or create a new one.

        Returns:
            (chain, is_new) — is_new=True if this session was just created
        """
        with self._lock:
            if session_id in self._sessions:
                entry = self._sessions[session_id]
                if not entry.is_expired(self.ttl):
                    entry.touch()
                    return entry.chain, False
                else:
                    # Expired — remove and recreate
                    del self._sessions[session_id]

            # Evict oldest if at capacity
            if len(self._sessions) >= self.max_sessions:
                self._evict_oldest()

            chain = self._create_chain()
            self._sessions[session_id] = SessionEntry(chain)
            return chain, True

    def get(self, session_id: str) -> Optional[any]:
        """Get existing session chain, or None if not found / expired."""
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry and not entry.is_expired(self.ttl):
                entry.touch()
                return entry.chain
            return None

    def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def reset(self, session_id: str) -> bool:
        """Reset a session to fresh state (new memory, same session_id)."""
        with self._lock:
            if session_id in self._sessions:
                old_entry = self._sessions[session_id]
                chain = self._create_chain()
                entry = SessionEntry(chain)
                self._sessions[session_id] = entry
                return True
            return False

    @staticmethod
    def new_session_id() -> str:
        return str(uuid.uuid4())

    def stats(self) -> dict:
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "max_sessions":    self.max_sessions,
                "ttl_seconds":     self.ttl,
            }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _create_chain(self):
        """Import here to avoid circular imports at module load time."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from phase3.chains.rag_chain import AnimeRAGChain

        return AnimeRAGChain(
            api_key=settings.get_api_key(),
            backend=settings.LLM_BACKEND,
            model=settings.LLM_MODEL,
            retrieval_k=settings.RETRIEVAL_K,
            temperature=settings.LLM_TEMPERATURE,
            stream=True,
        )

    def _evict_oldest(self) -> None:
        """Remove the session with the oldest last_accessed time."""
        if not self._sessions:
            return
        oldest_id = min(self._sessions, key=lambda sid: self._sessions[sid].last_accessed)
        del self._sessions[oldest_id]

    def _cleanup_loop(self) -> None:
        """Background thread: remove expired sessions every 5 minutes."""
        while True:
            time.sleep(300)  # 5 minutes
            self._cleanup_expired()

    def _cleanup_expired(self) -> None:
        with self._lock:
            expired = [
                sid for sid, entry in self._sessions.items()
                if entry.is_expired(self.ttl)
            ]
            for sid in expired:
                del self._sessions[sid]


# ── Singleton ─────────────────────────────────────────────────────────────────

_manager: Optional[SessionManager] = None
_manager_lock = threading.Lock()


def get_session_manager() -> SessionManager:
    """Return the global SessionManager singleton (lazy init)."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = SessionManager()
    return _manager
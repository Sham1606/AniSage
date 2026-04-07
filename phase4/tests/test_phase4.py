"""
Phase 4 API Tests — Run with: pytest phase4/tests/test_phase4.py -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Project root on sys.path — must happen before any phase* import
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _load_app():
    """
    Load phase4/app.py regardless of where it sits in the package tree.
    Tries package import first, falls back to direct file load.
    """
    # Try normal import (works if conftest.py added ROOT to sys.path)
    try:
        from phase4.api.app import create_app
        return create_app
    except ModuleNotFoundError:
        pass
    try:
        from phase4.app import create_app
        return create_app
    except ModuleNotFoundError:
        pass

    # Fallback: load by file path directly
    app_candidates = [
        ROOT / "phase4" / "api" / "app.py",
        ROOT / "phase4" / "app.py",
    ]
    for candidate in app_candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("phase4_app", candidate)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.create_app

    raise ImportError(f"Could not find app.py in {ROOT}/phase4/")


@pytest.fixture(scope="module")
def client():
    """TestClient with mocked LLM chain and real FAISS index."""
    from fastapi.testclient import TestClient

    with patch("phase4.api.core.session_manager.SessionManager._create_chain") as mock_create:
        mock_chain = MagicMock()
        mock_chain.chat.return_value = "Here are my top anime recommendations!"
        mock_chain.memory.turn_count = 1
        mock_chain.memory.profile.to_summary.return_value = ""
        mock_chain.memory.is_first_turn.return_value = True
        mock_chain.memory.session_id = "test-session"

        def mock_stream(msg):
            yield "Here are "
            yield "my top "
            yield "recommendations!"

        mock_chain.chat_stream.side_effect = mock_stream
        mock_create.return_value = mock_chain

        create_app = _load_app()
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:

    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        for key in ("status", "faiss_loaded", "active_sessions", "llm_backend"):
            assert key in data

    def test_root_redirects_to_docs(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 302, 307, 308)
        assert "/docs" in resp.headers.get("location", "")


# ── Chat ──────────────────────────────────────────────────────────────────────

class TestChat:

    def test_chat_returns_200(self, client):
        assert client.post("/chat", json={"message": "I want dark anime"}).status_code == 200

    def test_chat_response_schema(self, client):
        data = client.post("/chat", json={"message": "horror anime"}).json()
        for key in ("session_id", "message", "turn", "is_new_session"):
            assert key in data

    def test_chat_creates_session_id(self, client):
        data = client.post("/chat", json={"message": "hello"}).json()
        assert data["session_id"] and len(data["session_id"]) > 0

    def test_chat_with_existing_session(self, client):
        resp1 = client.post("/chat", json={"message": "dark anime"})
        sid   = resp1.json()["session_id"]
        resp2 = client.post("/chat", json={"message": "more", "session_id": sid})
        assert resp2.status_code == 200
        assert resp2.json()["session_id"] == sid

    def test_chat_empty_message_rejected(self, client):
        assert client.post("/chat", json={"message": ""}).status_code == 422

    def test_chat_message_too_long_rejected(self, client):
        assert client.post("/chat", json={"message": "x" * 3000}).status_code == 422

    def test_chat_models_endpoint(self, client):
        data = client.get("/chat/models").json()
        assert "backend" in data and "available" in data


# ── Search ────────────────────────────────────────────────────────────────────

class TestSearch:

    def test_search_returns_200(self, client):
        assert client.post("/search", json={"query": "samurai anime"}).status_code == 200

    def test_search_response_schema(self, client):
        data = client.post("/search", json={"query": "dark fantasy"}).json()
        for key in ("query", "results", "total"):
            assert key in data

    def test_search_returns_results(self, client):
        data = client.post("/search", json={"query": "action adventure"}).json()
        assert data["total"] > 0

    def test_search_result_schema(self, client):
        results = client.post("/search", json={"query": "romance"}).json()["results"]
        if results:
            for key in ("rank", "score", "title"):
                assert key in results[0]

    def test_search_k_parameter(self, client):
        data = client.post("/search", json={"query": "mecha", "k": 3}).json()
        assert data["total"] <= 3

    def test_search_empty_query_rejected(self, client):
        assert client.post("/search", json={"query": ""}).status_code == 422

    def test_anime_random(self, client):
        resp = client.get("/anime/random")
        assert resp.status_code == 200
        assert "title" in resp.json()

    def test_anime_random_with_filter(self, client):
        assert client.get("/anime/random?min_score=7.0").status_code in (200, 404)

    def test_anime_by_id_known(self, client):
        assert client.get("/anime/1").status_code in (200, 404)

    def test_anime_by_id_nonexistent(self, client):
        assert client.get("/anime/999999999").status_code == 404


# ── Session ───────────────────────────────────────────────────────────────────

class TestSession:

    def test_session_info_nonexistent(self, client):
        resp = client.get("/session/does-not-exist")
        assert resp.status_code == 200
        assert resp.json()["exists"] is False

    def test_session_created_after_chat(self, client):
        sid  = client.post("/chat", json={"message": "horror"}).json()["session_id"]
        data = client.get(f"/session/{sid}").json()
        assert data["exists"] is True and data["session_id"] == sid

    def test_session_reset(self, client):
        sid  = client.post("/chat", json={"message": "action"}).json()["session_id"]
        resp = client.post(f"/session/{sid}/reset")
        assert resp.status_code == 200

    def test_session_reset_nonexistent(self, client):
        assert client.post("/session/nonexistent/reset").status_code == 404

    def test_session_delete(self, client):
        sid = client.post("/chat", json={"message": "slice of life"}).json()["session_id"]
        assert client.delete(f"/session/{sid}").status_code == 204

    def test_sessions_stats(self, client):
        data = client.get("/sessions/stats").json()
        assert "active_sessions" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Phase 3 Tests

Run with: pytest phase3/tests/test_phase3.py -v

Tests:
  - ConversationMemory: history management, profile extraction, trimming
  - Prompt templates: candidate formatting, message building
  - AnimeRetriever: end-to-end retrieval with real FAISS index
  - AnimeRAGChain: full pipeline with mocked OpenAI (no API cost)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── ConversationMemory tests ──────────────────────────────────────────────────

class TestConversationMemory:

    @pytest.fixture
    def memory(self):
        from phase3.memory.conversation_memory import ConversationMemory
        return ConversationMemory(system_prompt="You are a helpful anime assistant.")

    def test_initial_state(self, memory):
        assert memory.turn_count == 0
        assert memory.is_first_turn()
        assert memory.profile.is_empty()

    def test_add_user_message(self, memory):
        memory.add_user("I want dark anime")
        history = memory.get_history()
        assert len(history) == 2  # system + user
        assert history[1]["role"] == "user"
        assert history[1]["content"] == "I want dark anime"

    def test_add_assistant_increments_turn(self, memory):
        memory.add_user("Hello")
        memory.add_assistant("Hi!")
        assert memory.turn_count == 1
        assert not memory.is_first_turn()

    def test_history_starts_with_system(self, memory):
        memory.add_user("test")
        history = memory.get_history()
        assert history[0]["role"] == "system"

    def test_get_last_assistant_message(self, memory):
        memory.add_user("q1")
        memory.add_assistant("a1")
        memory.add_user("q2")
        memory.add_assistant("a2")
        assert memory.get_last_assistant_message() == "a2"

    def test_get_last_assistant_none_when_empty(self, memory):
        assert memory.get_last_assistant_message() is None

    def test_history_trimming(self, memory):
        """History should not grow unbounded."""
        for i in range(30):
            memory.add_user(f"question {i}")
            memory.add_assistant(f"answer {i}")
        history = memory.get_history()
        # Should be trimmed: system + at most MAX_HISTORY_TURNS*2 messages
        assert len(history) <= memory.MAX_HISTORY_TURNS * 2 + 1

    def test_profile_injection_in_system_message(self, memory):
        """Profile summary should appear in system message once profile is populated."""
        memory.profile.liked_genres = ["Action", "Horror"]
        history = memory.get_history()
        assert "Action" in history[0]["content"]
        assert "Horror" in history[0]["content"]

    def test_serialization(self, memory, tmp_path):
        memory.add_user("test query")
        memory.add_assistant("test response")
        save_path = tmp_path / "session.json"
        memory.save(save_path)
        assert save_path.exists()
        data = json.loads(save_path.read_text())
        assert data["turn_count"] == 1
        assert len(data["history"]) == 2


class TestPreferenceProfile:

    @pytest.fixture
    def memory(self):
        from phase3.memory.conversation_memory import ConversationMemory
        return ConversationMemory(system_prompt="test")

    def test_genre_extraction_action(self, memory):
        memory.update_profile_from_query("I love action anime")
        assert "Action" in memory.profile.liked_genres

    def test_genre_extraction_romance(self, memory):
        memory.update_profile_from_query("romance and slice of life please")
        assert "Romance" in memory.profile.liked_genres
        assert "Slice of Life" in memory.profile.liked_genres

    def test_format_extraction_movie(self, memory):
        memory.update_profile_from_query("recommend a good anime movie")
        assert "Movie" in memory.profile.preferred_types

    def test_no_duplicate_genres(self, memory):
        memory.update_profile_from_query("action anime")
        memory.update_profile_from_query("more action anime")
        assert memory.profile.liked_genres.count("Action") == 1

    def test_profile_to_summary_empty(self, memory):
        assert memory.profile.to_summary() == ""

    def test_profile_to_summary_with_data(self, memory):
        memory.profile.liked_genres = ["Horror", "Mystery"]
        summary = memory.profile.to_summary()
        assert "Horror" in summary
        assert "Mystery" in summary


# ── Prompt template tests ─────────────────────────────────────────────────────

class TestPromptTemplates:

    @pytest.fixture
    def sample_candidates(self):
        return [
            {
                "_rank": 1, "_score": 0.85,
                "title": "Death Note", "year": 2006, "media_type": "TV",
                "score": 8.62, "genres": "Mystery, Supernatural, Thriller",
                "themes": "Detective, Psychological",
                "synopsis": "Light Yagami discovers a notebook that kills anyone whose name is written in it.",
            },
            {
                "_rank": 2, "_score": 0.72,
                "title": "Monster", "year": 2004, "media_type": "TV",
                "score": 8.68, "genres": "Drama, Mystery, Thriller",
                "themes": "Adult Cast, Detective, Medical",
                "synopsis": "A brilliant surgeon saves a criminal's life but becomes entangled in serial murders.",
            },
        ]

    def test_format_candidates_returns_string(self, sample_candidates):
        from phase3.prompts.prompt_templates import format_candidates
        result = format_candidates(sample_candidates)
        assert isinstance(result, str)
        assert "Death Note" in result
        assert "Monster" in result

    def test_format_candidates_includes_score(self, sample_candidates):
        from phase3.prompts.prompt_templates import format_candidates
        result = format_candidates(sample_candidates)
        assert "0.850" in result

    def test_format_candidates_empty_list(self):
        from phase3.prompts.prompt_templates import format_candidates
        result = format_candidates([])
        assert "No relevant" in result

    def test_format_candidates_truncates_synopsis(self, sample_candidates):
        from phase3.prompts.prompt_templates import format_candidates
        long_synopsis = "A" * 500
        sample_candidates[0]["synopsis"] = long_synopsis
        result = format_candidates(sample_candidates, max_synopsis=100)
        # The synopsis in output should be truncated
        assert long_synopsis[:500] not in result

    def test_build_user_message_contains_query(self, sample_candidates):
        from phase3.prompts.prompt_templates import build_user_message
        result = build_user_message("dark thriller anime", sample_candidates)
        assert "dark thriller anime" in result

    def test_build_user_message_followup_flag(self, sample_candidates):
        from phase3.prompts.prompt_templates import build_user_message
        result_first   = build_user_message("query", sample_candidates, is_followup=False)
        result_followup = build_user_message("query", sample_candidates, is_followup=True)
        assert result_first != result_followup

    def test_get_system_prompt_not_empty(self):
        from phase3.prompts.prompt_templates import get_system_prompt
        prompt = get_system_prompt()
        assert len(prompt) > 100
        assert "AniSage" in prompt or "anime" in prompt.lower()


# ── AnimeRetriever tests ──────────────────────────────────────────────────────

class TestAnimeRetriever:
    """These tests require the FAISS index to be built (phase2/data/faiss_index/)."""

    @pytest.fixture(scope="class")
    def retriever(self):
        """Load retriever once for all tests in this class."""
        try:
            from phase3.retrieval.anime_retriever import AnimeRetriever
            return AnimeRetriever()
        except RuntimeError:
            pytest.skip("FAISS index not found — run: python main.py build-index")

    def test_retriever_loads(self, retriever):
        assert retriever.store.count() > 0

    def test_retrieve_returns_results(self, retriever):
        results = retriever.retrieve("dark fantasy anime", k=5)
        assert len(results) > 0

    def test_retrieve_result_count(self, retriever):
        results = retriever.retrieve("romance high school", k=5)
        assert len(results) <= 5

    def test_retrieve_results_have_required_fields(self, retriever):
        results = retriever.retrieve("action adventure", k=3)
        for r in results:
            assert "title" in r
            assert "_rank" in r
            assert "_score" in r

    def test_retrieve_sorted_by_score(self, retriever):
        results = retriever.retrieve("samurai sword", k=5)
        scores = [r["_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_known_anime(self, retriever):
        """Well-known samurai anime should surface for a samurai query."""
        results = retriever.retrieve("samurai revenge feudal japan", k=10)
        titles = [r.get("title", "").lower() for r in results]
        # At least one result should contain "samurai" in the title
        assert any("samurai" in t for t in titles), \
            f"Expected samurai anime in results, got: {titles}"

    def test_multi_query_deduplication(self, retriever):
        results = retriever.retrieve_multi_query(
            ["action anime", "adventure anime"],
            k_per_query=5,
        )
        titles = [r.get("title") for r in results]
        # No duplicate titles
        assert len(titles) == len(set(titles))

    def test_exclude_titles(self, retriever):
        all_results = retriever.retrieve("action anime", k=5)
        if not all_results:
            pytest.skip("No results")
        exclude = [all_results[0]["title"]]
        filtered = retriever.retrieve("action anime", k=5, exclude_titles=exclude)
        result_titles = [r["title"] for r in filtered]
        assert exclude[0] not in result_titles


# ── AnimeRAGChain integration test (mocked OpenAI) ────────────────────────────

class TestAnimeRAGChain:
    """
    Tests the full pipeline with a mocked OpenAI client.
    No actual API calls — no cost.
    """

    @pytest.fixture
    def chain(self):
        """Create chain with mocked OpenAI."""
        try:
            from phase3.chains.rag_chain import AnimeRAGChain
        except Exception:
            pytest.skip("Phase 3 chain not available")

        with patch("phase3.chains.rag_chain.AnimeRetriever") as MockRetriever:
            # Mock FAISS retriever
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = [
                {
                    "_rank": 1, "_score": 0.85,
                    "title": "Mock Anime", "year": 2020,
                    "media_type": "TV", "score": 8.5,
                    "genres": "Action, Fantasy",
                    "synopsis": "A mock anime for testing purposes.",
                }
            ]
            mock_retriever.store.count.return_value = 14003
            MockRetriever.return_value = mock_retriever

            chain = AnimeRAGChain(api_key="sk-test-fake-key")

            # Mock the OpenAI client
            mock_response = MagicMock()
            mock_response.choices[0].message.content = (
                "Based on your request, I recommend **Mock Anime** — "
                "a fantastic action fantasy from 2020 with a score of 8.5."
            )
            chain.client.chat.completions.create = MagicMock(return_value=mock_response)

            return chain

    def test_chain_initializes(self, chain):
        assert chain.model == "gpt-4o-mini"
        assert chain.memory.is_first_turn()

    def test_chat_returns_string(self, chain):
        response = chain.chat("dark fantasy anime", stream=False)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_chat_increments_turn(self, chain):
        chain.chat("first question", stream=False)
        assert chain.memory.turn_count == 1

    def test_multiple_turns_build_history(self, chain):
        chain.chat("first question", stream=False)
        chain.chat("follow up question", stream=False)
        history = chain.memory.get_history()
        # system + 2 user + 2 assistant = 5
        assert len(history) == 5

    def test_reset_clears_memory(self, chain):
        chain.chat("a question", stream=False)
        chain.reset()
        assert chain.memory.turn_count == 0
        assert chain.memory.is_first_turn()

    def test_get_stats(self, chain):
        stats = chain.get_stats()
        assert "model" in stats
        assert "turn_count" in stats
        assert "index_size" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
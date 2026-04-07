"""
RAG Chain — Phase 3

Pipeline:
  User query
    → AnimeRetriever (FAISS semantic search)
    → prompt_templates.build_user_message (context injection)
    → ConversationMemory (full chat history)
    → LLM via Groq (default) or OpenAI
    → Response streamed back to user

Groq is the default backend — free, fast (~500 tok/s), supports:
  - llama-3.3-70b-versatile  (best quality, default)
  - llama-3.1-8b-instant     (fastest)
  - mixtral-8x7b-32768       (large context window)
  - gemma2-9b-it             (Google's model)

Groq's API is OpenAI-compatible, so the same SDK works for both.
Just point base_url at Groq's endpoint.

Usage:
    # Groq (default)
    chain = AnimeRAGChain(api_key="gsk_...")
    # OpenAI fallback
    chain = AnimeRAGChain(api_key="sk-...", backend="openai")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase1.utils.helpers import log_error, log_info, log_warning
from phase3.memory.conversation_memory import ConversationMemory
from phase3.prompts.prompt_templates import build_user_message, get_system_prompt
from phase3.retrieval.anime_retriever import AnimeRetriever

ROOT = Path(__file__).parent.parent.parent

# ── Backend configs ───────────────────────────────────────────────────────────

BACKENDS = {
    "groq": {
        "base_url":    "https://api.groq.com/openai/v1",
        "env_var":     "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "models": [
            "llama-3.3-70b-versatile",   # best quality
            "llama-3.1-8b-instant",      # fastest
            "mixtral-8x7b-32768",        # large context
            "gemma2-9b-it",              # Google
        ],
    },
    "openai": {
        "base_url":    None,             # uses default OpenAI URL
        "env_var":     "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    },
}


class AnimeRAGChain:
    """
    Full RAG pipeline for anime recommendations.

    Each instance manages ONE conversation session.
    Create a new instance for each new chat session.

    Args:
        api_key:     API key for the chosen backend
        backend:     "groq" (default, free) or "openai"
        model:       Model name — defaults to best model for the backend
        retrieval_k: Number of FAISS candidates per query (default: 10)
        temperature: LLM creativity — 0.7 is good for recommendations
        stream:      Enable token streaming (default: True)
    """

    def __init__(
        self,
        api_key:     Optional[str] = None,
        backend:     str = "groq",
        model:       Optional[str] = None,
        retrieval_k: int = 10,
        temperature: float = 0.7,
        stream:      bool = True,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

        if backend not in BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'. Choose: {list(BACKENDS)}")

        config = BACKENDS[backend]

        # Resolve API key
        key = api_key or os.environ.get(config["env_var"])
        if not key:
            raise ValueError(
                f"{config['env_var']} not set.\n"
                f"Either pass api_key= or set the {config['env_var']} env var.\n"
                f"{'Get a free Groq key at: https://console.groq.com/keys' if backend == 'groq' else 'Get an OpenAI key at: https://platform.openai.com/api-keys'}"
            )

        from openai import OpenAI

        # Groq uses OpenAI-compatible API — just override base_url
        client_kwargs = {"api_key": key}
        if config["base_url"]:
            client_kwargs["base_url"] = config["base_url"]

        self.client      = OpenAI(**client_kwargs)
        self.backend     = backend
        self.model       = model or config["default_model"]
        self.retrieval_k = retrieval_k
        self.temperature = temperature
        self.stream      = stream

        # Initialise retriever (loads FAISS index once)
        self.retriever = AnimeRetriever()

        # Initialise conversation memory
        self.memory = ConversationMemory(system_prompt=get_system_prompt())

        log_info(f"RAG chain ready — backend={backend}, model={self.model}, k={retrieval_k}")

    # ── Intent detection ─────────────────────────────────────────────────────

    # Queries shorter than this (words) skip FAISS retrieval
    _MIN_RETRIEVAL_WORDS = 3

    # Short phrases that are clearly conversational — no retrieval needed
    _SKIP_RETRIEVAL_PHRASES = {
        "hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "sure",
        "yes", "no", "cool", "great", "nice", "sounds good", "got it",
        "lol", "haha", "bye", "goodbye", "see you", "quit", "exit",
    }

    def _needs_retrieval(self, user_input: str) -> bool:
        """
        Decide if this query should trigger FAISS retrieval.
        Skips retrieval for greetings, acknowledgements, and very short phrases
        that clearly aren't anime requests.
        """
        text = user_input.strip().lower().rstrip("!?.")

        # Exact match against known conversational phrases
        if text in self._SKIP_RETRIEVAL_PHRASES:
            return False

        # Very short inputs (1-2 words) that aren't "more" or "next"
        words = text.split()
        if len(words) <= 2 and text not in {"more", "next", "another", "more please", "go on"}:
            return False

        return True

    def _is_preference_query(self, user_input: str) -> bool:
        """
        Returns True only if the user is expressing their OWN preferences.
        Prevents extracting genres from the LLM's suggestions or follow-up questions.

        Only runs profile extraction when the user is clearly stating what THEY want.
        """
        text = user_input.lower()

        # Strong intent signals — user is expressing their own preferences
        intent_signals = [
            "i want", "i like", "i love", "i prefer", "i enjoy",
            "recommend", "suggest", "show me", "find me",
            "looking for", "something with", "give me",
            "i'm into", "i am into", "fan of",
        ]
        return any(signal in text for signal in intent_signals)

    # ── Main chat interface ───────────────────────────────────────────────────

    def chat(self, user_input: str, stream: Optional[bool] = None) -> str:
        """
        Process a user message, retrieve relevant anime, call LLM, return response.
        """
        should_stream = stream if stream is not None else self.stream

        # Only update preference profile when user is expressing their own preferences
        if self._is_preference_query(user_input):
            self.memory.update_profile_from_query(user_input)

        # Only retrieve from FAISS when the query is an actual anime request
        if self._needs_retrieval(user_input):
            candidates = self._retrieve(user_input)
        else:
            candidates = []
            log_info("Skipping FAISS retrieval for conversational input")

        is_followup = not self.memory.is_first_turn()
        user_message = build_user_message(user_input, candidates, is_followup=is_followup)
        self.memory.add_user(user_message)

        if should_stream:
            response = self._call_llm_streaming(self.memory.get_history())
        else:
            response = self._call_llm(self.memory.get_history())

        self.memory.add_assistant(response)
        return response

    def chat_stream(self, user_input: str) -> Iterator[str]:
        """
        Streaming version — yields response chunks as they arrive.
        Use for CLI interactive mode and Streamlit UI.
        """
        if self._is_preference_query(user_input):
            self.memory.update_profile_from_query(user_input)

        if self._needs_retrieval(user_input):
            candidates = self._retrieve(user_input)
        else:
            candidates = []
            log_info("Skipping FAISS retrieval for conversational input")

        is_followup = not self.memory.is_first_turn()
        user_message = build_user_message(user_input, candidates, is_followup=is_followup)
        self.memory.add_user(user_message)

        full_response = ""
        for chunk in self._stream_llm(self.memory.get_history()):
            full_response += chunk
            yield chunk

        self.memory.add_assistant(full_response)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _retrieve(self, query: str) -> list[dict]:
        profile = self.memory.profile
        filter_type          = profile.preferred_types[0] if len(profile.preferred_types) == 1 else None
        filter_min_mal_score = profile.min_score
        filter_min_year      = profile.year_range[0] if profile.year_range else None
        filter_max_year      = profile.year_range[1] if profile.year_range else None

        try:
            candidates = self.retriever.retrieve(
                query=query,
                k=self.retrieval_k,
                filter_type=filter_type,
                filter_min_year=filter_min_year,
                filter_max_year=filter_max_year,
                filter_min_mal_score=filter_min_mal_score,
                exclude_titles=profile.avoided_titles or None,
            )
        except Exception as e:
            log_warning(f"Retrieval failed: {e} — returning empty candidates")
            candidates = []

        log_info(f"Retrieved {len(candidates)} candidates for: '{query[:50]}'")
        return candidates

    # ── LLM calls ─────────────────────────────────────────────────────────────

    def _call_llm(self, messages: list[dict]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1500,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            log_error(f"LLM API error ({self.backend}): {e}")
            raise

    def _call_llm_streaming(self, messages: list[dict]) -> str:
        full = ""
        for chunk in self._stream_llm(messages):
            full += chunk
        return full

    def _stream_llm(self, messages: list[dict]) -> Iterator[str]:
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1500,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            log_error(f"LLM streaming error ({self.backend}): {e}")
            raise

    # ── Session utilities ─────────────────────────────────────────────────────

    def reset(self) -> None:
        self.memory = ConversationMemory(system_prompt=get_system_prompt())
        log_info("Conversation reset — new session started")

    def save_session(self, path: Optional[Path] = None) -> Path:
        if path is None:
            sessions_dir = ROOT / "phase3" / "data" / "sessions"
            path = sessions_dir / f"session_{self.memory.session_id}.json"
        self.memory.save(path)
        log_info(f"Session saved to {path}")
        return path

    def get_stats(self) -> dict:
        return {
            "backend":    self.backend,
            "model":      self.model,
            "turn_count": self.memory.turn_count,
            "session_id": self.memory.session_id,
            "profile":    self.memory.profile.to_summary() or "No profile yet",
            "index_size": self.retriever.store.count(),
        }
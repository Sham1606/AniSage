"""
Conversation Memory — Phase 3

Manages multi-turn chat history and a running user preference profile.
No external dependencies — pure Python, stored in memory during a session.

The memory has two layers:
  1. message_history  : Full OpenAI-format message list for the LLM
  2. preference_profile: Extracted user preferences (genres, themes, seen/avoided)

Usage:
    memory = ConversationMemory()
    memory.add_user("I want dark psychological anime")
    memory.add_assistant("Here are my top picks: ...")
    memory.get_history()          # → list of {role, content} dicts
    memory.get_profile_summary()  # → "Likes: dark, psychological. Dislikes: mecha"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── Preference Profile ────────────────────────────────────────────────────────

@dataclass
class PreferenceProfile:
    """
    Running summary of what the user has told us across the conversation.
    Updated incrementally as the conversation progresses.
    """
    liked_genres:     list[str] = field(default_factory=list)
    disliked_genres:  list[str] = field(default_factory=list)
    liked_themes:     list[str] = field(default_factory=list)
    disliked_themes:  list[str] = field(default_factory=list)
    preferred_types:  list[str] = field(default_factory=list)   # TV, Movie, OVA
    year_range:       Optional[tuple[int, int]] = None
    min_score:        Optional[float] = None
    seen_titles:      list[str] = field(default_factory=list)   # titles user mentioned watching
    avoided_titles:   list[str] = field(default_factory=list)   # titles user wants to avoid
    free_text_notes:  list[str] = field(default_factory=list)   # catch-all for nuanced prefs

    def to_summary(self) -> str:
        """
        Generate a concise preference summary to inject into the system context.
        Only includes non-empty fields.
        """
        parts = []
        if self.liked_genres:
            parts.append(f"Preferred genres: {', '.join(self.liked_genres)}")
        if self.disliked_genres:
            parts.append(f"Disliked genres: {', '.join(self.disliked_genres)}")
        if self.liked_themes:
            parts.append(f"Liked themes: {', '.join(self.liked_themes)}")
        if self.disliked_themes:
            parts.append(f"Disliked themes: {', '.join(self.disliked_themes)}")
        if self.preferred_types:
            parts.append(f"Preferred format: {', '.join(self.preferred_types)}")
        if self.year_range:
            parts.append(f"Year preference: {self.year_range[0]}–{self.year_range[1]}")
        if self.min_score:
            parts.append(f"Minimum MAL score: {self.min_score}")
        if self.seen_titles:
            parts.append(f"Already watched: {', '.join(self.seen_titles[:10])}")
        if self.avoided_titles:
            parts.append(f"Please avoid: {', '.join(self.avoided_titles)}")
        if self.free_text_notes:
            parts.append(f"Other notes: {'; '.join(self.free_text_notes[-3:])}")
        return "\n".join(parts) if parts else ""

    def is_empty(self) -> bool:
        return not self.to_summary()


# ── Conversation Memory ───────────────────────────────────────────────────────

class ConversationMemory:
    """
    Manages the full conversation state for one chat session.

    message_history format (OpenAI standard):
        [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]

    The system message is rebuilt on each turn to include the latest
    preference profile summary, so the LLM always has fresh context.
    """

    # Max messages to keep in history (older ones are trimmed)
    # Keeps token costs predictable — 20 turns ≈ ~6k tokens for chat history
    MAX_HISTORY_TURNS = 20

    def __init__(
        self,
        system_prompt: str,
        session_id: Optional[str] = None,
    ) -> None:
        self.system_prompt  = system_prompt
        self.session_id     = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.profile        = PreferenceProfile()
        self._turns: list[dict] = []   # alternating user/assistant messages only
        self.turn_count     = 0

    # ── Message management ────────────────────────────────────────────────────

    def add_user(self, content: str) -> None:
        """Add a user message to history."""
        self._turns.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str) -> None:
        """Add an assistant message to history."""
        self._turns.append({"role": "assistant", "content": content})
        self.turn_count += 1
        self._trim()

    def _trim(self) -> None:
        """Keep only the last MAX_HISTORY_TURNS * 2 messages (user+assistant pairs)."""
        max_messages = self.MAX_HISTORY_TURNS * 2
        if len(self._turns) > max_messages:
            self._turns = self._turns[-max_messages:]

    def get_history(self) -> list[dict]:
        """
        Return full message list ready to pass to OpenAI API.
        System message is first, rebuilt each time to include current profile.
        """
        system_content = self.system_prompt

        # Inject preference profile if we have one
        profile_summary = self.profile.to_summary()
        if profile_summary:
            system_content = (
                f"{self.system_prompt}\n\n"
                f"## User Preference Profile (built from this conversation)\n"
                f"{profile_summary}"
            )

        return [
            {"role": "system", "content": system_content},
            *self._turns,
        ]

    def get_last_assistant_message(self) -> Optional[str]:
        """Return the most recent assistant response."""
        for msg in reversed(self._turns):
            if msg["role"] == "assistant":
                return msg["content"]
        return None

    def is_first_turn(self) -> bool:
        return self.turn_count == 0

    # ── Profile updates ───────────────────────────────────────────────────────

    def update_profile_from_query(self, query: str) -> None:
        """
        Lightweight keyword-based preference extraction from user queries.
        Updates the profile incrementally — no LLM call needed here.

        For production, this could be replaced with an LLM extraction step,
        but keyword matching covers most common cases well.
        """
        q = query.lower()

        # Genre signals
        genre_map = {
            "action":      "Action",
            "romance":     "Romance",
            "comedy":      "Comedy",
            "horror":      "Horror",
            "thriller":    "Thriller",
            "mystery":     "Mystery",
            "sci-fi":      "Science Fiction",
            "science fiction": "Science Fiction",
            "fantasy":     "Fantasy",
            "slice of life": "Slice of Life",
            "drama":       "Drama",
            "sports":      "Sports",
            "mecha":       "Mecha",
            "isekai":      "Isekai",
            "shounen":     "Shounen",
            "shoujo":      "Shoujo",
            "seinen":      "Seinen",
        }
        for keyword, genre in genre_map.items():
            if keyword in q:
                if genre not in self.profile.liked_genres:
                    self.profile.liked_genres.append(genre)

        # Negative signals
        avoid_signals = ["not too", "no mecha", "avoid", "don't want", "not a fan of", "without"]
        for signal in avoid_signals:
            if signal in q:
                # Capture what comes after the signal
                idx = q.find(signal) + len(signal)
                rest = q[idx:idx+30].strip()
                if rest:
                    self.profile.free_text_notes.append(f"User said '{signal} {rest}'")

        # Format signals
        if "movie" in q:
            if "Movie" not in self.profile.preferred_types:
                self.profile.preferred_types.append("Movie")
        if "series" in q or "tv show" in q or "long" in q:
            if "TV" not in self.profile.preferred_types:
                self.profile.preferred_types.append("TV")

        # Score preference
        import re
        score_match = re.search(r"score[d]?\s+(?:above|over|at least)?\s*(\d+(?:\.\d+)?)", q)
        if score_match:
            self.profile.min_score = float(score_match.group(1))

        # Year preferences
        decade_match = re.search(r"(\d{4})s?", q)
        if decade_match:
            decade = int(decade_match.group(1))
            if 1960 <= decade <= 2030:
                self.profile.year_range = (decade, decade + 9)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize session to dict (for saving/logging)."""
        return {
            "session_id":  self.session_id,
            "turn_count":  self.turn_count,
            "history":     self._turns,
            "profile":     {
                "liked_genres":    self.profile.liked_genres,
                "disliked_genres": self.profile.disliked_genres,
                "liked_themes":    self.profile.liked_themes,
                "disliked_themes": self.profile.disliked_themes,
                "preferred_types": self.profile.preferred_types,
                "year_range":      list(self.profile.year_range) if self.profile.year_range else None,
                "min_score":       self.profile.min_score,
                "seen_titles":     self.profile.seen_titles,
                "avoided_titles":  self.profile.avoided_titles,
                "free_text_notes": self.profile.free_text_notes,
            },
        }

    def save(self, path: Path) -> None:
        """Save session to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
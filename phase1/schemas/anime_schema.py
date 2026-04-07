"""
Anime data schema — the single source of truth for every anime document
in the pipeline. Both Jikan and AniList collectors produce AnimeDocument objects.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import re


class AnimeDocument(BaseModel):
    """
    Canonical representation of a single anime entry.
    Produced by collectors, consumed by the embedding pipeline.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    mal_id: Optional[int] = Field(None, description="MyAnimeList ID")
    anilist_id: Optional[int] = Field(None, description="AniList ID")
    title: str = Field(..., description="Primary English title")
    title_japanese: Optional[str] = Field(None, description="Japanese title")
    title_synonyms: list[str] = Field(default_factory=list)

    # ── Content ───────────────────────────────────────────────────────────────
    synopsis: str = Field(default="", description="Full plot synopsis")
    synopsis_token_count: int = Field(default=0, description="Approximate token count of synopsis")
    genres: list[str] = Field(default_factory=list, description="Standardised genre labels")
    themes: list[str] = Field(default_factory=list, description="Thematic tags (e.g. School, Military)")
    demographics: list[str] = Field(default_factory=list, description="e.g. Shounen, Josei")
    tags: list[str] = Field(default_factory=list, description="Extended tags from AniList")

    # ── Metadata ──────────────────────────────────────────────────────────────
    media_type: Optional[str] = Field(None, description="TV, Movie, OVA, ONA, Special, Music")
    status: Optional[str] = Field(None, description="Finished, Airing, Not yet aired")
    episodes: Optional[int] = Field(None, ge=0)
    duration_per_ep_min: Optional[int] = Field(None, ge=0, description="Minutes per episode")
    year: Optional[int] = Field(None, ge=1900, le=2030)
    season: Optional[str] = Field(None, description="Winter/Spring/Summer/Fall")
    studios: list[str] = Field(default_factory=list)
    source: Optional[str] = Field(None, description="Manga, Light Novel, Original, etc.")

    # ── Scores ────────────────────────────────────────────────────────────────
    score: Optional[float] = Field(None, ge=0.0, le=10.0, description="MAL score 0-10")
    scored_by: Optional[int] = Field(None, ge=0)
    rank: Optional[int] = Field(None, ge=0)
    popularity: Optional[int] = Field(None, ge=0)
    members: Optional[int] = Field(None, ge=0)
    favorites: Optional[int] = Field(None, ge=0)
    mean_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="AniList mean (0-100)")

    # ── Assets ────────────────────────────────────────────────────────────────
    image_url: Optional[str] = Field(None, description="Poster image URL")
    trailer_url: Optional[str] = Field(None)
    mal_url: Optional[str] = Field(None)
    anilist_url: Optional[str] = Field(None)

    # ── Pipeline metadata ─────────────────────────────────────────────────────
    data_sources: list[str] = Field(default_factory=list, description="Which APIs contributed data")
    is_adult: bool = Field(default=False)
    embedding_text: str = Field(default="", description="Pre-built text for embedding (set by processor)")

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("synopsis", mode="before")
    @classmethod
    def clean_synopsis(cls, v: str) -> str:
        if not v:
            return ""
        # Strip HTML tags
        v = re.sub(r"<[^>]+>", "", v)
        # Collapse excessive whitespace
        v = re.sub(r"\s+", " ", v).strip()
        # Remove MAL-specific boilerplate
        v = re.sub(r"\[Written by MAL Rewrite\]", "", v, flags=re.IGNORECASE).strip()
        v = re.sub(r"\(Source:.*?\)", "", v, flags=re.IGNORECASE).strip()
        return v

    @field_validator("title", mode="before")
    @classmethod
    def clean_title(cls, v: str) -> str:
        if not v:
            raise ValueError("Title cannot be empty")
        return v.strip()

    @field_validator("genres", "themes", "demographics", "tags", mode="before")
    @classmethod
    def normalize_list(cls, v) -> list[str]:
        if not v:
            return []
        return [str(x).strip() for x in v if x]

    @field_validator("media_type", mode="before")
    @classmethod
    def normalize_type(cls, v) -> Optional[str]:
        if not v:
            return None
        mapping = {
            "tv": "TV", "movie": "Movie", "ova": "OVA", "ona": "ONA",
            "special": "Special", "music": "Music",
            "TV": "TV", "Movie": "Movie", "OVA": "OVA", "ONA": "ONA",
        }
        return mapping.get(str(v).lower(), str(v))

    def compute_synopsis_tokens(self) -> None:
        """Rough approximation: 1 token ≈ 4 characters."""
        self.synopsis_token_count = len(self.synopsis) // 4

    def is_embeddable(self) -> bool:
        """Return True if this entry has enough data to produce a meaningful embedding."""
        return (
            len(self.synopsis) >= 50  # minimum ~12 tokens
            and len(self.title) > 0
        )

    def build_embedding_text(self) -> str:
        """
        Construct the rich text string that will be converted to a vector.
        Combines all semantic signals into one document string.
        """
        parts = [f"Title: {self.title}"]

        if self.title_japanese:
            parts.append(f"Japanese Title: {self.title_japanese}")

        if self.media_type:
            parts.append(f"Type: {self.media_type}")

        if self.year:
            season_str = f"{self.season} {self.year}" if self.season else str(self.year)
            parts.append(f"Year: {season_str}")

        if self.genres:
            parts.append(f"Genres: {', '.join(self.genres)}")

        if self.themes:
            parts.append(f"Themes: {', '.join(self.themes)}")

        if self.demographics:
            parts.append(f"Demographics: {', '.join(self.demographics)}")

        if self.tags:
            # Top 15 tags from AniList — most relevant for semantic search
            parts.append(f"Tags: {', '.join(self.tags[:15])}")

        if self.studios:
            parts.append(f"Studios: {', '.join(self.studios)}")

        if self.source:
            parts.append(f"Source: {self.source}")

        if self.episodes:
            parts.append(f"Episodes: {self.episodes}")

        if self.score:
            parts.append(f"Score: {self.score}/10")

        if self.synopsis:
            parts.append(f"\nSynopsis: {self.synopsis}")

        self.embedding_text = "\n".join(parts)
        return self.embedding_text

    def merge_with(self, other: "AnimeDocument") -> "AnimeDocument":
        """
        Merge data from another source into this document.
        'other' fills in missing fields — self takes priority for existing data.
        """
        for field_name in self.model_fields:
            self_val = getattr(self, field_name)
            other_val = getattr(other, field_name)

            # Merge lists (union, deduplicated)
            if isinstance(self_val, list) and isinstance(other_val, list):
                merged = list(dict.fromkeys(self_val + other_val))
                object.__setattr__(self, field_name, merged)
            # Fill empty scalar fields from other
            elif not self_val and other_val:
                object.__setattr__(self, field_name, other_val)

        # Track all contributing sources
        self.data_sources = list(set(self.data_sources + other.data_sources))
        return self

    class Config:
        populate_by_name = True

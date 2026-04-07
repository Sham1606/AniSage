"""
Phase 1 Tests — Schema validation, parser correctness, and processor logic.

Run with: pytest phase1/tests/ -v
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase1.schemas.anime_schema import AnimeDocument
from phase1.collectors.jikan_collector import parse_jikan_entry
from phase1.collectors.anilist_collector import parse_anilist_entry


# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_JIKAN_ENTRY = {
    "mal_id": 1,
    "title": "Cowboy Bebop",
    "title_english": "Cowboy Bebop",
    "title_japanese": "カウボーイビバップ",
    "titles": [{"type": "Synonym", "title": "CB"}],
    "synopsis": "In the year 2071, humanity has colonized several planets and moons "
                "of the solar system leaving the now uninhabitable surface of Earth behind. "
                "The Inter Solar System Police attempts to keep peace in the galaxy, aided in part "
                "by outlaw bounty hunters. [Written by MAL Rewrite]",
    "type": "TV",
    "status": "Finished Airing",
    "episodes": 26,
    "duration": "24 min per ep",
    "year": 1998,
    "season": "spring",
    "score": 8.75,
    "scored_by": 900000,
    "rank": 28,
    "popularity": 39,
    "members": 1900000,
    "favorites": 80000,
    "genres": [{"name": "Action"}, {"name": "Adventure"}],
    "themes": [{"name": "Space"}],
    "demographics": [{"name": "Seinen"}],
    "studios": [{"name": "Sunrise"}],
    "source": "Original",
    "rating": "R - 17+ (violence & profanity)",
    "images": {"jpg": {"large_image_url": "https://cdn.myanimelist.net/images/1.jpg"}},
    "url": "https://myanimelist.net/anime/1",
    "trailer": {"url": "https://youtube.com/watch?v=abc"},
    "aired": {"from": "1998-04-03T00:00:00+00:00"},
}

MOCK_ANILIST_ENTRY = {
    "id": 1,
    "idMal": 1,
    "title": {"english": "Cowboy Bebop", "romaji": "Cowboy Bebop", "native": "カウボーイビバップ"},
    "description": "Follow Spike Spiegel, a bounty hunter traversing the solar system.",
    "genres": ["Action", "Adventure", "Drama", "Sci-Fi"],
    "tags": [
        {"name": "Space", "rank": 95, "isMediaSpoiler": False},
        {"name": "Bounty Hunter", "rank": 90, "isMediaSpoiler": False},
        {"name": "Episodic", "rank": 80, "isMediaSpoiler": False},
        {"name": "Twist Ending", "rank": 40, "isMediaSpoiler": True},  # spoiler — should be excluded
    ],
    "format": "TV",
    "status": "FINISHED",
    "episodes": 26,
    "duration": 24,
    "season": "SPRING",
    "seasonYear": 1998,
    "startDate": {"year": 1998, "month": 4, "day": 3},
    "studios": {"nodes": [{"name": "Sunrise"}]},
    "source": "ORIGINAL",
    "averageScore": 87,
    "meanScore": 87,
    "popularity": 200000,
    "favourites": 50000,
    "coverImage": {"large": "https://s4.anilist.co/file/1.jpg"},
    "siteUrl": "https://anilist.co/anime/1",
    "isAdult": False,
}


# ── Schema tests ──────────────────────────────────────────────────────────────

class TestAnimeSchema:

    def test_basic_construction(self):
        doc = AnimeDocument(title="Test Anime", synopsis="A story about something.")
        assert doc.title == "Test Anime"
        assert doc.synopsis == "A story about something."

    def test_synopsis_cleans_mal_boilerplate(self):
        doc = AnimeDocument(
            title="Test",
            synopsis="A great story. [Written by MAL Rewrite] (Source: ANN)"
        )
        assert "[Written by MAL Rewrite]" not in doc.synopsis
        assert "(Source: ANN)" not in doc.synopsis

    def test_synopsis_strips_html(self):
        doc = AnimeDocument(title="Test", synopsis="<b>Bold text</b> and <i>italics</i>.")
        assert "<b>" not in doc.synopsis
        assert "Bold text" in doc.synopsis

    def test_empty_title_raises(self):
        with pytest.raises(Exception):
            AnimeDocument(title="", synopsis="Some synopsis here with enough content.")

    def test_score_bounds(self):
        with pytest.raises(Exception):
            AnimeDocument(title="Test", score=11.0)  # above 10.0

    def test_genre_normalization(self):
        doc = AnimeDocument(title="Test", genres=["Sci-Fi", "  Action ", ""])
        assert "Science Fiction" in doc.genres
        assert "Action" in doc.genres
        assert "" not in doc.genres

    def test_build_embedding_text(self):
        doc = AnimeDocument(
            title="Cowboy Bebop",
            synopsis="A bounty hunter story set in space.",
            genres=["Action", "Adventure"],
            themes=["Space"],
            year=1998,
            score=8.75,
        )
        text = doc.build_embedding_text()
        assert "Cowboy Bebop" in text
        assert "Action" in text
        assert "bounty hunter" in text
        assert "1998" in text

    def test_is_embeddable_with_short_synopsis(self):
        doc = AnimeDocument(title="Test", synopsis="Short.")
        assert not doc.is_embeddable()

    def test_is_embeddable_with_good_synopsis(self):
        doc = AnimeDocument(
            title="Test",
            synopsis="A longer synopsis that clearly has more than fifty characters in total for testing.",
        )
        assert doc.is_embeddable()

    def test_token_count_approximation(self):
        doc = AnimeDocument(title="Test", synopsis="A" * 400)  # ~100 tokens
        doc.compute_synopsis_tokens()
        assert doc.synopsis_token_count == 100

    def test_merge_fills_missing_fields(self):
        doc1 = AnimeDocument(title="Anime A", synopsis="First synopsis here with content.", mal_id=1)
        doc2 = AnimeDocument(title="Anime A", anilist_id=100, mal_id=1, tags=["Action", "Drama"])
        doc1.merge_with(doc2)
        assert doc1.anilist_id == 100
        assert "Action" in doc1.tags

    def test_merge_self_takes_priority(self):
        doc1 = AnimeDocument(title="Original Title", synopsis="Good synopsis text here.", score=8.0, mal_id=1)
        doc2 = AnimeDocument(title="Other Title", synopsis="Different synopsis.", score=7.0, mal_id=1)
        doc1.merge_with(doc2)
        assert doc1.title == "Original Title"
        assert doc1.score == 8.0


# ── Jikan parser tests ────────────────────────────────────────────────────────

class TestJikanParser:

    def test_parses_valid_entry(self):
        doc = parse_jikan_entry(MOCK_JIKAN_ENTRY)
        assert doc is not None
        assert doc.mal_id == 1
        assert doc.title == "Cowboy Bebop"
        assert doc.episodes == 26
        assert doc.score == 8.75
        assert "Action" in doc.genres
        assert doc.duration_per_ep_min == 24
        assert doc.year == 1998
        assert doc.season == "Spring"
        assert "jikan" in doc.data_sources
        assert doc.is_adult is False

    def test_strips_mal_boilerplate_from_synopsis(self):
        doc = parse_jikan_entry(MOCK_JIKAN_ENTRY)
        assert "[Written by MAL Rewrite]" not in doc.synopsis

    def test_handles_missing_english_title(self):
        entry = MOCK_JIKAN_ENTRY.copy()
        entry["title_english"] = None
        entry["title"] = "カウボーイビバップ"
        doc = parse_jikan_entry(entry)
        assert doc is not None
        assert doc.title == "カウボーイビバップ"

    def test_returns_none_for_empty_entry(self):
        doc = parse_jikan_entry({})
        assert doc is None

    def test_adult_detection(self):
        entry = MOCK_JIKAN_ENTRY.copy()
        entry["rating"] = "Rx - Hentai"
        doc = parse_jikan_entry(entry)
        assert doc.is_adult is True


# ── AniList parser tests ──────────────────────────────────────────────────────

class TestAniListParser:

    def test_parses_valid_entry(self):
        doc = parse_anilist_entry(MOCK_ANILIST_ENTRY)
        assert doc is not None
        assert doc.anilist_id == 1
        assert doc.mal_id == 1
        assert doc.title == "Cowboy Bebop"
        assert doc.episodes == 26
        assert doc.mean_score == 87.0
        assert "anilist" in doc.data_sources

    def test_filters_spoiler_tags(self):
        doc = parse_anilist_entry(MOCK_ANILIST_ENTRY)
        # "Twist Ending" is marked as spoiler — must be excluded
        assert "Twist Ending" not in doc.tags

    def test_includes_non_spoiler_tags(self):
        doc = parse_anilist_entry(MOCK_ANILIST_ENTRY)
        assert "Space" in doc.tags
        assert "Bounty Hunter" in doc.tags

    def test_status_mapping(self):
        doc = parse_anilist_entry(MOCK_ANILIST_ENTRY)
        assert doc.status == "Finished Airing"

    def test_season_mapping(self):
        doc = parse_anilist_entry(MOCK_ANILIST_ENTRY)
        assert doc.season == "Spring"

    def test_format_mapping(self):
        doc = parse_anilist_entry(MOCK_ANILIST_ENTRY)
        assert doc.media_type == "TV"

    def test_falls_back_to_romaji_when_no_english(self):
        entry = MOCK_ANILIST_ENTRY.copy()
        entry["title"] = {"english": None, "romaji": "Cowboy Bebop", "native": "カウボーイビバップ"}
        doc = parse_anilist_entry(entry)
        assert doc.title == "Cowboy Bebop"

    def test_returns_none_when_no_title(self):
        entry = MOCK_ANILIST_ENTRY.copy()
        entry["title"] = {"english": None, "romaji": None, "native": None}
        doc = parse_anilist_entry(entry)
        assert doc is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Prompt Templates — Phase 3

Formats retrieved FAISS candidates into structured context for the LLM.
"""

from __future__ import annotations
from pathlib import Path

def get_system_prompt() -> str:
    """Load system prompt from disk. Resolves path relative to this file."""
    prompt_path = Path(__file__).resolve().parent / "system_prompt.txt"
    if not prompt_path.exists():
        # Fallback: search up from cwd
        for candidate in [
            Path("phase3/prompts/system_prompt.txt"),
            Path(__file__).parent / "system_prompt.txt",
        ]:
            if candidate.exists():
                prompt_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"system_prompt.txt not found. Expected at: {prompt_path}"
            )
    return prompt_path.read_text(encoding="utf-8").strip()


def format_candidates(candidates: list[dict], max_synopsis: int = 300) -> str:
    """Format FAISS results into an LLM-readable context block."""
    if not candidates:
        return "No relevant anime found in the database for this query."

    lines = ["## Retrieved Anime Candidates\n"]

    for c in candidates:
        rank      = c.get("_rank", "?")
        score     = c.get("_score", 0)
        title     = c.get("title", "Unknown")
        year      = c.get("year", "")
        mtype     = c.get("media_type", "")
        mal_score = c.get("score", "") or c.get("mean_score", "")
        genres    = c.get("genres", "")
        themes    = c.get("themes", "")
        tags      = c.get("tags", "")
        synopsis  = c.get("synopsis", "")

        if synopsis and len(synopsis) > max_synopsis:
            synopsis = synopsis[:max_synopsis].rsplit(" ", 1)[0] + "..."

        lines.append(f"### [{rank}] {title}")
        lines.append(f"- **Similarity**: {score:.3f}")

        meta_parts = []
        if mtype:     meta_parts.append(f"Type: {mtype}")
        if year:      meta_parts.append(f"Year: {year}")
        if mal_score: meta_parts.append(f"MAL Score: {mal_score}/10")
        if meta_parts:
            lines.append(f"- **Info**: {' | '.join(meta_parts)}")
        if genres:
            lines.append(f"- **Genres**: {genres}")
        if themes:
            lines.append(f"- **Themes**: {themes}")
        if tags:
            lines.append(f"- **Tags**: {str(tags)[:120]}")
        if synopsis:
            lines.append(f"- **Synopsis**: {synopsis}")
        lines.append("")

    return "\n".join(lines)


def build_user_message(
    user_query: str,
    candidates: list[dict],
    is_followup: bool = False,
) -> str:
    """Build the full user message: retrieved context + query."""

    # No candidates = conversational turn (greeting, thanks, clarification)
    # Don't inject any retrieval context — just pass the message naturally
    if not candidates:
        return user_query

    candidate_block = format_candidates(candidates)

    if is_followup:
        instruction = (
            "The user is refining their search based on prior conversation. "
            "Use these new candidates alongside their stated preferences.\n\n"
        )
    else:
        instruction = (
            "Using the retrieved anime candidates below, "
            "provide personalised recommendations for the user's request.\n\n"
        )

    return (
        f"{instruction}"
        f"{candidate_block}\n"
        f"---\n\n"
        f"**User request**: {user_query}"
    )
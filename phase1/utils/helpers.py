"""
Shared utilities: structured logging, rate limiting, checkpoint management.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import orjson
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()


# ── Logging ───────────────────────────────────────────────────────────────────

def log_info(msg: str) -> None:
    console.print(f"[bold cyan]ℹ[/bold cyan]  {msg}")

def log_success(msg: str) -> None:
    console.print(f"[bold green]✔[/bold green]  {msg}")

def log_warning(msg: str) -> None:
    console.print(f"[bold yellow]⚠[/bold yellow]  {msg}")

def log_error(msg: str) -> None:
    console.print(f"[bold red]✘[/bold red]  {msg}")


# ── Progress bar ──────────────────────────────────────────────────────────────

def make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


# ── Async rate limiter ────────────────────────────────────────────────────────

class AsyncRateLimiter:
    """
    Token-bucket rate limiter for async HTTP calls.

    Usage:
        limiter = AsyncRateLimiter(calls_per_second=2.0)
        async with limiter:
            response = await session.get(url)
    """

    def __init__(self, calls_per_second: float = 2.0) -> None:
        self.min_interval = 1.0 / calls_per_second
        self._last_call: float = 0.0
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "AsyncRateLimiter":
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


# ── Checkpoint management ─────────────────────────────────────────────────────

class CheckpointManager:
    """
    Saves and loads progress so collection can resume after interruption.

    Checkpoint file format (JSON):
        {
            "completed_ids": [1, 2, 3, ...],
            "failed_ids": [5, 10, ...],
            "last_updated": "2024-01-01T00:00:00"
        }
    """

    def __init__(self, checkpoint_path: Path) -> None:
        self.path = checkpoint_path
        self.completed_ids: set[int] = set()
        self.failed_ids: set[int] = set()
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path, "rb") as f:
                data = orjson.loads(f.read())
            self.completed_ids = set(data.get("completed_ids", []))
            self.failed_ids = set(data.get("failed_ids", []))
            log_info(f"Checkpoint loaded: {len(self.completed_ids)} completed, {len(self.failed_ids)} failed")

    def save(self) -> None:
        data = {
            "completed_ids": sorted(self.completed_ids),
            "failed_ids": sorted(self.failed_ids),
            "total_completed": len(self.completed_ids),
        }
        with open(self.path, "wb") as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    def mark_done(self, anime_id: int) -> None:
        self.completed_ids.add(anime_id)
        self.failed_ids.discard(anime_id)

    def mark_failed(self, anime_id: int) -> None:
        if anime_id not in self.completed_ids:
            self.failed_ids.add(anime_id)

    def is_done(self, anime_id: int) -> bool:
        return anime_id in self.completed_ids

    def pending_count(self, total: int) -> int:
        return total - len(self.completed_ids)


# ── JSONL I/O ─────────────────────────────────────────────────────────────────

def append_jsonl(path: Path, record: dict) -> None:
    """Append a single JSON record to a JSONL file."""
    with open(path, "ab") as f:
        f.write(orjson.dumps(record) + b"\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read all records from a JSONL file."""
    records = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(orjson.loads(line))
    return records


def load_existing_ids(path: Path) -> set[int]:
    """Get the set of mal_ids already written to a JSONL file."""
    if not path.exists():
        return set()
    ids: set[int] = set()
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                record = orjson.loads(line)
                if mid := record.get("mal_id"):
                    ids.add(mid)
    return ids

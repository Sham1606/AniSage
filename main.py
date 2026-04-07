"""
Phase 1 CLI — Anime RAG Data Collection & Processing Pipeline

Usage:
    # Collect from both sources then process:
    python main.py run-all

    # Individual steps:
    python main.py collect-jikan
    python main.py collect-anilist
    python main.py process

    # Quick test run (100 entries each):
    python main.py run-all --test

    # Check dataset stats:
    python main.py stats
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

# ── Windows DLL Load Fix (WinError 1114) ──────────────────────────────────────
import os
import sys
if sys.platform == "win32":
    # 1. Prevent OpenMP runtime conflicts
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    try:
        # 2. Add torch/lib to DLL search path and PATH
        root = Path(__file__).resolve().parent
        for site_pkgs in [
            root / "phase4" / "ragani" / "Lib" / "site-packages",
            root / ".venv" / "Lib" / "site-packages",
            Path(sys.executable).parent.parent / "Lib" / "site-packages",
        ]:
            t_lib = site_pkgs / "torch" / "lib"
            if t_lib.exists():
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(t_lib))
                os.environ["PATH"] = str(t_lib) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass
# ─────────────────────────────────────────────────────────────────────────────

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="anime-rag-phase1",
    help="🎌 Anime RAG — Phase 1: Data Collection & Processing",
    add_completion=False,
)
console = Console()

DATA_DIR = Path(__file__).resolve().parent / "data"


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def collect_jikan(
    test: bool = typer.Option(False, "--test", is_flag=True, help="Collect only 10 pages (~250 anime) for testing"),
    skip_adult: bool = typer.Option(True, "--skip-adult", is_flag=True),
):
    """Collect anime from MyAnimeList via the Jikan v4 API."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from phase1.collectors.jikan_collector import run_jikan_collection

    console.print("\n[bold purple]🎌 Starting Jikan Collection...[/bold purple]\n")
    max_pages = 10 if test else None
    total = asyncio.run(run_jikan_collection(max_pages=max_pages, skip_adult=skip_adult))
    console.print(f"\n[bold green]✔ Jikan: {total:,} records collected[/bold green]")


@app.command()
def collect_anilist(
    test: bool = typer.Option(False, "--test", is_flag=True, help="Collect only 10 pages (~500 anime) for testing"),
    skip_adult: bool = typer.Option(True, "--skip-adult", is_flag=True),
):
    """Collect anime from AniList via GraphQL API."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from phase1.collectors.anilist_collector import run_anilist_collection

    console.print("\n[bold blue]⚡ Starting AniList Collection...[/bold blue]\n")
    max_pages = 10 if test else None
    total = asyncio.run(run_anilist_collection(max_pages=max_pages, skip_adult=skip_adult))
    console.print(f"\n[bold green]✔ AniList: {total:,} records collected[/bold green]")


@app.command()
def process(
    min_tokens: int = typer.Option(30, "--min-tokens", help="Minimum synopsis token count to keep an entry"),
    skip_adult: bool = typer.Option(True, "--skip-adult", is_flag=True),
):
    """Merge, clean, and build embedding text for all collected anime."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from phase1.processors.data_processor import run_processing_pipeline

    console.print("\n[bold yellow]⚙  Running Data Processing Pipeline...[/bold yellow]\n")
    total = run_processing_pipeline(min_synopsis_tokens=min_tokens, skip_adult=skip_adult)
    console.print(f"\n[bold green]✔ Processing complete: {total:,} clean anime records[/bold green]")


@app.command("run-all")
def run_all(
    test: bool = typer.Option(False, "--test", is_flag=True, help="Quick test with ~500 anime total"),
    skip_adult: bool = typer.Option(True, "--skip-adult", is_flag=True),
):
    """
    Run the full Phase 1 pipeline:
    collect Jikan → collect AniList → merge & process.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from phase1.collectors.jikan_collector import run_jikan_collection
    from phase1.collectors.anilist_collector import run_anilist_collection
    from phase1.processors.data_processor import run_processing_pipeline

    console.rule("[bold purple]🎌 Anime RAG — Phase 1 Full Pipeline[/bold purple]")
    max_pages = 10 if test else None

    # Step 1: Jikan
    console.print("\n[bold]Step 1/3 — Collecting from Jikan (MyAnimeList)...[/bold]")
    j_total = asyncio.run(run_jikan_collection(max_pages=max_pages, skip_adult=skip_adult))

    # Step 2: AniList
    console.print("\n[bold]Step 2/3 — Collecting from AniList...[/bold]")
    a_total = asyncio.run(run_anilist_collection(max_pages=max_pages, skip_adult=skip_adult))

    # Step 3: Process
    console.print("\n[bold]Step 3/3 — Merging & Processing...[/bold]")
    final_total = run_processing_pipeline(skip_adult=skip_adult)

    # Summary
    console.rule("[bold green]Phase 1 Complete[/bold green]")
    table = Table(title="Phase 1 Summary", show_header=True, header_style="bold cyan")
    table.add_column("Step", style="bold")
    table.add_column("Records", justify="right")
    table.add_row("Jikan collected", f"{j_total:,}")
    table.add_row("AniList collected", f"{a_total:,}")
    table.add_row("[bold green]Final clean dataset", f"[bold green]{final_total:,}")
    console.print(table)
    console.print(f"\n📁 Output: {DATA_DIR / 'processed' / 'anime_merged.jsonl'}")
    console.print("✅ Ready for Phase 2: Vector Embeddings!\n")


@app.command()
def stats():
    """Show statistics about the current processed dataset."""
    report_path = DATA_DIR / "processed" / "processing_report.json"
    merged_path = DATA_DIR / "processed" / "anime_merged.jsonl"

    if not report_path.exists():
        console.print("[red]No processing report found. Run 'process' first.[/red]")
        raise typer.Exit(1)

    with open(report_path) as f:
        report = json.load(f)

    console.rule("[bold]Dataset Statistics[/bold]")

    # Summary table
    t = Table(show_header=True, header_style="bold magenta")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    t.add_row("Total input records", f"{report['input_count']:,}")
    t.add_row("Clean output records", f"[bold green]{report['output_count']:,}[/bold green]")
    t.add_row("Removed (total)", f"[yellow]{report['removed_total']:,}[/yellow]")
    t.add_row("  ↳ No synopsis", str(report["removed_breakdown"]["no_synopsis"]))
    t.add_row("  ↳ Synopsis too short", str(report["removed_breakdown"]["synopsis_too_short"]))
    t.add_row("  ↳ Adult content", str(report["removed_breakdown"]["adult_content"]))
    t.add_row("Data quality %", f"{report['data_quality_pct']}%")
    t.add_row("Year range", f"{report['year_range']['min']} – {report['year_range']['max']}")
    console.print(t)

    # Sources
    console.print("\n[bold]Data Sources:[/bold]")
    for src, count in report["sources"].items():
        console.print(f"  {src}: {count:,} records")

    # Media types
    console.print("\n[bold]Media Types:[/bold]")
    for mt, count in list(report["media_types"].items())[:6]:
        console.print(f"  {mt}: {count:,}")

    # Top genres
    console.print("\n[bold]Top 10 Genres:[/bold]")
    for genre, count in list(report["top_10_genres"].items())[:10]:
        console.print(f"  {genre}: {count:,}")

    # File size
    if merged_path.exists():
        size_mb = merged_path.stat().st_size / (1024 * 1024)
        console.print(f"\n📁 JSONL size: {size_mb:.1f} MB")


@app.command()
def validate(
    sample: int = typer.Option(5, "--sample", help="Number of records to display"),
):
    """Validate the processed dataset by printing sample records."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from phase1.utils.helpers import read_jsonl
    from phase1.schemas.anime_schema import AnimeDocument

    merged_path = DATA_DIR / "processed" / "anime_merged.jsonl"
    if not merged_path.exists():
        console.print("[red]No processed data found. Run 'process' first.[/red]")
        raise typer.Exit(1)

    records = read_jsonl(merged_path)
    console.print(f"\n[bold]Total records in file: {len(records):,}[/bold]\n")

    # Show sample
    for i, record in enumerate(records[:sample]):
        doc = AnimeDocument(**record)
        console.rule(f"[cyan]Record {i+1}: {doc.title}[/cyan]")
        console.print(f"  MAL ID:      {doc.mal_id}")
        console.print(f"  AniList ID:  {doc.anilist_id}")
        console.print(f"  Type:        {doc.media_type} | {doc.year}")
        console.print(f"  Score:       {doc.score}/10 | AniList: {doc.mean_score}/100")
        console.print(f"  Genres:      {', '.join(doc.genres[:5])}")
        console.print(f"  Themes:      {', '.join(doc.themes[:5])}")
        console.print(f"  Tags:        {', '.join(doc.tags[:5])}")
        console.print(f"  Synopsis:    {doc.synopsis[:150]}...")
        console.print(f"  Sources:     {doc.data_sources}")
        console.print(f"  Embeddable:  {doc.is_embeddable()}")
        console.print(f"  Embed text preview:\n    {doc.embedding_text[:200]}...\n")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 COMMANDS — Vector Embeddings
# ══════════════════════════════════════════════════════════════════════════════

@app.command()
def embed(
    model_type: str = typer.Option(
        "sentence-transformer",
        "--model-type",
        help="'sentence-transformer' (free, local) or 'openai' (requires API key)",
    ),
    model_name: str = typer.Option(
        None, "--model-name",
        help="Override model name (e.g. all-mpnet-base-v2 or text-embedding-3-small)",
    ),
    batch_size: int = typer.Option(64, "--batch-size", help="Records per embedding batch"),
    force: bool = typer.Option(False, "--force", is_flag=True, help="Re-embed even if checkpoint exists"),
):
    """Embed all anime records from Phase 1 into dense vectors."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from phase2.embeddings.embed_pipeline import run_embed_pipeline

    console.print("\n[bold cyan]🧠 Phase 2 — Embedding Pipeline[/bold cyan]\n")
    console.print(f"  Model type : [yellow]{model_type}[/yellow]")
    console.print(f"  Batch size : [yellow]{batch_size}[/yellow]")
    console.print(f"  Force      : [yellow]{force}[/yellow]\n")

    embeddings, metadata = run_embed_pipeline(
        model_type=model_type,
        model_name=model_name,
        batch_size=batch_size,
        force_reembed=force,
    )
    console.print(f"\n[bold green]✔ Embeddings complete: {len(metadata):,} records embedded[/bold green]")
    console.print(f"[bold green]✔ Shape: {embeddings.shape}[/bold green]")
    console.print("\n[dim]Next step: python main.py build-index[/dim]")


@app.command("build-index")
def build_index(
    reset_chroma:    bool = typer.Option(False, "--reset-chroma",    is_flag=True, help="Drop and rebuild ChromaDB collection"),
    skip_chroma:     bool = typer.Option(True,  "--skip-chroma",  is_flag=True, help="Skip ChromaDB (default: True — avoids Windows hang)"),
    skip_faiss:      bool = typer.Option(False, "--skip-faiss",      is_flag=True, help="Skip FAISS step"),
):
    """
    Load saved embeddings and build vector indexes.
    By default builds FAISS only (fast, reliable on all platforms).
    Use --no-skip-chroma to also build ChromaDB (may hang on Windows).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from phase2.vectordb.index_builder import run_build_index

    console.print("\n[bold cyan]🗄️  Phase 2 — Building Vector Indexes[/bold cyan]\n")
    if skip_chroma:
        console.print("[dim]ChromaDB skipped (FAISS only mode)[/dim]\n")

    stats = run_build_index(
        reset_chroma=reset_chroma,
        skip_chroma=skip_chroma,
        skip_faiss=skip_faiss,
    )

    if "chromadb" in stats:
        console.print(f"\n[green]ChromaDB:[/green] {stats['chromadb']['total_records']:,} records")
    if "faiss" in stats:
        console.print(f"\n[bold green]✔ FAISS:[/bold green] {stats['faiss']['total_vectors']:,} vectors · {stats['faiss']['dimensions']} dims · {stats['faiss']['index_size_mb']} MB")

    console.print("\n[dim]Next step: python main.py search \"dark fantasy with demons\"[/dim]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language query, e.g. 'dark fantasy with demons'"),
    top_k: int = typer.Option(5, "--top-k", help="Number of results to return"),
    store: str = typer.Option("faiss", "--store", help="'faiss' or 'chroma'"),
    model_type: str = typer.Option("sentence-transformer", "--model-type"),
):
    """
    Semantic anime search — type a query, get the top-K most similar anime.
    This is the Phase 2 retrieval demo (RAG chain comes in Phase 3).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from phase2.embeddings.embedding_models import get_embedding_model

    console.print(f"\n[bold cyan]🔍 Searching:[/bold cyan] [yellow]\"{query}\"[/yellow]\n")

    # Embed the query
    model = get_embedding_model(model_type)
    q_vec = model.embed_one(query)

    if store == "faiss":
        from phase2.vectordb.faiss_store import FAISSStore
        faiss_store = FAISSStore()
        if not faiss_store.load():
            console.print("[red]FAISS index not found. Run: python main.py build-index[/red]")
            raise typer.Exit(1)
        results = faiss_store.query(q_vec, k=top_k)

    else:  # chroma
        from phase2.vectordb.chromadb_store import ChromaStore
        chroma = ChromaStore()
        if chroma.count() == 0:
            console.print("[red]ChromaDB is empty. Run: python main.py build-index[/red]")
            raise typer.Exit(1)
        results = chroma.query(query_embedding=q_vec, n_results=top_k)

    # ── Display results ───────────────────────────────────────────────────────
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit(0)

    console.rule(f"[bold]Top {len(results)} Results[/bold]")

    for r in results:
        score_key = "_score" if store == "faiss" else "_similarity"
        score = r.get(score_key, 0)
        score_color = "green" if score > 0.7 else "yellow" if score > 0.5 else "red"

        console.print(
            f"\n[bold white]#{r['_rank']}[/bold white]  "
            f"[bold cyan]{r.get('title', 'Unknown')}[/bold cyan]  "
            f"[{score_color}]({score:.3f})[/{score_color}]"
        )

        year = r.get("year", "")
        mtype = r.get("media_type", "")
        score_10 = r.get("score", "")
        if year or mtype:
            console.print(f"     [dim]{mtype}  {year}  {'★ ' + str(score_10) if score_10 else ''}[/dim]")

        genres = r.get("genres", "")
        if genres:
            console.print(f"     [magenta]{genres[:80]}[/magenta]")

        synopsis = r.get("synopsis", "")
        if synopsis:
            console.print(f"     {synopsis[:120]}...")

    console.print()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 COMMANDS — RAG Chain
# ══════════════════════════════════════════════════════════════════════════════

@app.command()
def chat(
    api_key:      str  = typer.Option(None,              "--api-key",      help="API key (Groq or OpenAI)"),
    backend:      str  = typer.Option("groq",            "--backend",      help="LLM backend: 'groq' (default, free) or 'openai'"),
    model:        str  = typer.Option(None,              "--model",        help="Model name. Groq default: llama-3.3-70b-versatile"),
    k:            int  = typer.Option(10,                "--k",            help="Number of anime candidates to retrieve per query"),
    no_stream:    bool = typer.Option(False,             "--no-stream",    is_flag=True, help="Disable streaming output"),
    save_session: bool = typer.Option(False,             "--save-session", is_flag=True, help="Save session to JSON on exit"),
):
    """
    Start an interactive anime recommendation chat.
    Powered by FAISS semantic search + Groq LLM (free, fast).

    Default backend is Groq — get a free key at https://console.groq.com/keys
    Type 'quit' to exit, 'reset' to start over, 'stats' for session info.
    """
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent))
    from phase3.chains.rag_chain import AnimeRAGChain, BACKENDS

    # Load from .env if no key passed
    if not api_key:
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            try:
                import dotenv
                dotenv.load_dotenv(env_path, encoding="utf-8")
            except Exception:
                # Manual parse fallback (handles BOM/encoding issues)
                for line in env_path.read_text(encoding="utf-8-sig").splitlines():
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k_name, _, v = line.partition("=")
                        os.environ[k_name.strip()] = v.strip().strip('"')
        env_var = BACKENDS[backend]["env_var"]
        api_key = os.environ.get(env_var)

    if not api_key:
        env_var = BACKENDS.get(backend, {}).get("env_var", "API_KEY")
        console.print(f"[red]Error: {env_var} not set.[/red]")
        if backend == "groq":
            console.print("Get a free Groq key at: [cyan]https://console.groq.com/keys[/cyan]")
        console.print(f"Then: echo {env_var}=your-key-here > .env")
        raise typer.Exit(1)

    console.print()
    console.rule("[bold cyan]🎌 AniSage — Anime Recommendation Assistant[/bold cyan]")
    chosen_model = model or BACKENDS[backend]["default_model"]
    console.print(f"[dim]Backend: {backend} | Model: {chosen_model} | Top-{k} candidates per query[/dim]")
    console.print("[dim]Commands: 'quit' exit  'reset' new session  'stats' show profile[/dim]")
    console.print()

    try:
        chain = AnimeRAGChain(
            api_key=api_key, backend=backend, model=model,
            retrieval_k=k, stream=not no_stream,
        )
    except Exception as e:
        console.print(f"[red]Failed to initialise RAG chain: {e}[/red]")
        raise typer.Exit(1)

    console.print("[bold green]✔ Ready! Ask me anything about anime.[/bold green]")
    console.print("[dim]Example: 'I want dark psychological anime with complex characters'[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye! 👋[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye! 👋[/dim]")
            break

        if user_input.lower() == "reset":
            chain.reset()
            console.print("[yellow]Session reset — starting fresh![/yellow]\n")
            continue

        if user_input.lower() == "stats":
            stats = chain.get_stats()
            console.print(f"[dim]Model: {stats['model']} | Turns: {stats['turn_count']} | Index: {stats['index_size']:,} anime[/dim]")
            if stats['profile'] != 'No profile yet':
                console.print(f"[dim]Profile:\n{stats['profile']}[/dim]")
            console.print()
            continue

        console.print()
        console.print("[bold magenta]AniSage:[/bold magenta] ", end="")

        try:
            if not no_stream:
                for chunk in chain.chat_stream(user_input):
                    console.print(chunk, end="")
                console.print("\n")
            else:
                response = chain.chat(user_input, stream=False)
                console.print(response)
                console.print()
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")

    if save_session:
        saved_path = chain.save_session()
        console.print(f"[dim]Session saved to {saved_path}[/dim]")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 COMMANDS — FastAPI Server
# ══════════════════════════════════════════════════════════════════════════════

@app.command()
def serve(
    host:   str  = typer.Option("0.0.0.0", "--host",   help="Bind host"),
    port:   int  = typer.Option(8000,      "--port",   help="Bind port"),
    reload: bool = typer.Option(False,     "--reload", is_flag=True, help="Auto-reload on code changes (dev mode)"),
    workers:int  = typer.Option(1,         "--workers",help="Number of worker processes"),
):
    """
    Start the AniSage FastAPI server.
    Visit http://localhost:8000/docs for interactive API docs.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed. Run: pip install uvicorn[/red]")
        raise typer.Exit(1)

    console.print()
    console.rule("[bold cyan]🚀 AniSage API Server[/bold cyan]")
    console.print(f"[dim]Host: {host}:{port} | Reload: {reload} | Workers: {workers}[/dim]")
    console.print(f"[dim]Docs: http://localhost:{port}/docs[/dim]")
    console.print(f"[dim]Health: http://localhost:{port}/health[/dim]\n")

    uvicorn.run(
        "phase4.api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="info",
    )


if __name__ == "__main__":
    app()
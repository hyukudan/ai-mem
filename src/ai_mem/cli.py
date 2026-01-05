import json
import os
import time
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config, update_config
from .memory import MemoryManager

app = typer.Typer(help="AI Memory: Persistent memory for any LLM.")
console = Console()


def get_memory_manager() -> MemoryManager:
    try:
        return MemoryManager()
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1)


@app.command()
def config(
    llm_provider: Optional[str] = typer.Option(None, help="LLM provider (gemini, openai-compatible, vllm)"),
    llm_model: Optional[str] = typer.Option(None, help="LLM model name"),
    llm_api_key: Optional[str] = typer.Option(None, help="LLM API key"),
    llm_base_url: Optional[str] = typer.Option(None, help="LLM base URL (OpenAI-compatible)"),
    embeddings_provider: Optional[str] = typer.Option(None, help="Embeddings provider (fastembed, gemini, openai-compatible, auto)"),
    embeddings_model: Optional[str] = typer.Option(None, help="Embeddings model name"),
    embeddings_api_key: Optional[str] = typer.Option(None, help="Embeddings API key"),
    embeddings_base_url: Optional[str] = typer.Option(None, help="Embeddings base URL (OpenAI-compatible)"),
    data_dir: Optional[str] = typer.Option(None, help="Data directory for SQLite + vector store"),
    sqlite_path: Optional[str] = typer.Option(None, help="Override SQLite database path"),
    vector_dir: Optional[str] = typer.Option(None, help="Override vector store directory"),
    show: bool = typer.Option(False, help="Show current configuration"),
):
    """Configure providers, embeddings, and storage paths."""
    if show or not any(
        [
            llm_provider,
            llm_model,
            llm_api_key,
            llm_base_url,
            embeddings_provider,
            embeddings_model,
            embeddings_api_key,
            embeddings_base_url,
            data_dir,
            sqlite_path,
            vector_dir,
        ]
    ):
        config_data = load_config().model_dump()
        console.print(json.dumps(config_data, indent=2))
        return

    patch = {"llm": {}, "embeddings": {}, "storage": {}}
    if llm_provider:
        patch["llm"]["provider"] = llm_provider
    if llm_model:
        patch["llm"]["model"] = llm_model
    if llm_api_key:
        patch["llm"]["api_key"] = llm_api_key
    if llm_base_url:
        patch["llm"]["base_url"] = llm_base_url
    if embeddings_provider:
        patch["embeddings"]["provider"] = embeddings_provider
    if embeddings_model:
        patch["embeddings"]["model"] = embeddings_model
    if embeddings_api_key:
        patch["embeddings"]["api_key"] = embeddings_api_key
    if embeddings_base_url:
        patch["embeddings"]["base_url"] = embeddings_base_url
    if data_dir:
        patch["storage"]["data_dir"] = data_dir
    if sqlite_path:
        patch["storage"]["sqlite_path"] = sqlite_path
    if vector_dir:
        patch["storage"]["vector_dir"] = vector_dir

    update_config(patch)
    console.print("[green]Configuration updated.[/green]")


@app.command()
def add(
    content: str = typer.Argument(..., help="Text to store as memory"),
    obs_type: str = typer.Option("note", help="Observation type"),
    project: Optional[str] = typer.Option(None, help="Project path override"),
    tag: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tags to associate"),
    no_summary: bool = typer.Option(False, help="Disable summarization"),
):
    """Store a new memory observation."""
    manager = get_memory_manager()
    obs = manager.add_observation(
        content=content,
        obs_type=obs_type,
        project=project,
        tags=tag or [],
        summarize=not no_summary,
    )
    console.print(f"[green]Memory stored![/green] (ID: {obs.id})")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results"),
    project: Optional[str] = typer.Option(None, help="Project path filter"),
    obs_type: Optional[str] = typer.Option(None, help="Observation type filter"),
    all_projects: bool = typer.Option(False, help="Search across all projects"),
    date_start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD or epoch)"),
    date_end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD or epoch)"),
    output: str = typer.Option("text", "--format", "-f", help="Output format: text, json"),
):
    """Search memory index (compact results)."""
    manager = get_memory_manager()
    if not project and not all_projects:
        project = os.getcwd()
    results = manager.search(
        query,
        limit=limit,
        project=project,
        obs_type=obs_type,
        date_start=date_start,
        date_end=date_end,
    )
    if output == "json":
        print(json.dumps([item.model_dump() for item in results], indent=2))
        return

    table = Table(title=f"Search results for: {query}")
    table.add_column("ID", style="dim")
    table.add_column("Summary", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Project", style="green")
    for item in results:
        table.add_row(item.id, item.summary, item.type or "-", item.project)
    console.print(table)


@app.command()
def timeline(
    anchor_id: Optional[str] = typer.Option(None, help="Anchor observation ID"),
    query: Optional[str] = typer.Option(None, help="Query to find anchor"),
    depth_before: int = typer.Option(3, help="Observations before anchor"),
    depth_after: int = typer.Option(3, help="Observations after anchor"),
    project: Optional[str] = typer.Option(None, help="Project path filter"),
    all_projects: bool = typer.Option(False, help="Search across all projects"),
    output: str = typer.Option("text", "--format", "-f", help="Output format: text, json"),
):
    """Get chronological context around an observation."""
    manager = get_memory_manager()
    if not project and not all_projects:
        project = os.getcwd()
    results = manager.timeline(
        anchor_id=anchor_id,
        query=query,
        depth_before=depth_before,
        depth_after=depth_after,
        project=project,
    )
    if output == "json":
        print(json.dumps([item.model_dump() for item in results], indent=2))
        return

    table = Table(title="Timeline")
    table.add_column("ID", style="dim")
    table.add_column("Summary", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Project", style="green")
    for item in results:
        table.add_row(item.id, item.summary, item.type or "-", item.project)
    console.print(table)


@app.command()
def stats(
    project: Optional[str] = typer.Option(None, help="Project path filter"),
    all_projects: bool = typer.Option(False, help="Include all projects"),
    obs_type: Optional[str] = typer.Option(None, help="Observation type filter"),
    date_start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD or epoch)"),
    date_end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD or epoch)"),
    tag_limit: int = typer.Option(10, help="Max tags to include"),
    day_limit: int = typer.Option(14, help="Max days to include"),
    type_tag_limit: int = typer.Option(3, help="Max tags per type to include"),
    output: str = typer.Option("text", "--format", "-f", help="Output format: text, json"),
):
    """Show observation counts grouped by type and project."""
    manager = get_memory_manager()
    if not project and not all_projects:
        project = os.getcwd()
    data = manager.get_stats(
        project=project,
        obs_type=obs_type,
        date_start=date_start,
        date_end=date_end,
        tag_limit=tag_limit,
        day_limit=day_limit,
        type_tag_limit=type_tag_limit,
    )
    if output == "json":
        print(json.dumps(data, indent=2))
        return

    console.print(f"[bold]Total observations:[/bold] {data.get('total', 0)}")
    console.print(
        f"[bold]Recent (last {data.get('day_limit', day_limit)} days):[/bold] "
        f"{data.get('recent_total', 0)}"
    )
    trend_delta = data.get("trend_delta", 0)
    trend_pct = data.get("trend_pct")
    if trend_pct is None:
        trend_label = f"{trend_delta:+d}" if isinstance(trend_delta, int) else f"{trend_delta:+}"
    else:
        trend_label = f"{trend_delta:+} ({trend_pct:+.1f}%)"
    console.print(f"[bold]Change vs previous:[/bold] {trend_label}")
    by_type = data.get("by_type", [])
    if by_type:
        table = Table(title="By type")
        table.add_column("Type", style="magenta")
        table.add_column("Count", style="green")
        for item in by_type:
            table.add_row(item.get("type", "-"), str(item.get("count", 0)))
        console.print(table)

    if all_projects or not project:
        by_project = data.get("by_project", [])
        if by_project:
            table = Table(title="By project")
            table.add_column("Project", style="cyan")
            table.add_column("Count", style="green")
            for item in by_project:
                table.add_row(item.get("project", "-"), str(item.get("count", 0)))
            console.print(table)

    top_tags = data.get("top_tags", [])
    if top_tags:
        table = Table(title="Top tags")
        table.add_column("Tag", style="cyan")
        table.add_column("Count", style="green")
        for item in top_tags:
            table.add_row(item.get("tag", "-"), str(item.get("count", 0)))
        console.print(table)

    by_day = data.get("by_day", [])
    if by_day:
        table = Table(title="Recent days")
        table.add_column("Day", style="cyan")
        table.add_column("Count", style="green")
        for item in by_day:
            table.add_row(item.get("day", "-"), str(item.get("count", 0)))
        console.print(table)

    top_tags_by_type = data.get("top_tags_by_type", [])
    if top_tags_by_type and not obs_type:
        table = Table(title="Top tags by type")
        table.add_column("Type", style="magenta")
        table.add_column("Tag", style="cyan")
        table.add_column("Count", style="green")
        for group in top_tags_by_type:
            for item in group.get("tags", []):
                table.add_row(group.get("type", "-"), item.get("tag", "-"), str(item.get("count", 0)))
        console.print(table)


@app.command()
def get(
    ids: List[str] = typer.Argument(..., help="Observation IDs"),
    output: str = typer.Option("json", "--format", "-f", help="Output format: text, json"),
):
    """Fetch full observation details by ID."""
    manager = get_memory_manager()
    results = manager.get_observations(ids)
    if output == "json":
        print(json.dumps(results, indent=2))
        return

    for obs in results:
        console.print(f"[bold]{obs['id']}[/bold] {obs['summary']}")
        console.print(obs["content"])
        console.print("")


@app.command()
def server(
    port: int = typer.Option(8000, help="Port to run the API server"),
):
    """Start the API + web viewer server."""
    from .server import start_server

    console.print(f"[green]Starting ai-mem server at http://localhost:{port}[/green]")
    start_server(port=port)


@app.command()
def mcp():
    """Start MCP stdio server with search tools."""
    from .mcp_server import run_stdio

    run_stdio()


@app.command()
def ingest(
    path: str = typer.Argument(".", help="Path to the project directory to ingest"),
    dry_run: bool = typer.Option(False, help="Scan files without storing them"),
):
    """Ingest a project's codebase into memory."""
    from .ingest import ingest_project

    manager = get_memory_manager()
    abs_path = os.path.abspath(path)
    console.print(f"[bold blue]Ingesting project: {abs_path}[/bold blue]")
    if dry_run:
        console.print("[dim]Dry run mode enabled. No memories will be stored.[/dim]")
    with console.status("[bold green]Scanning and indexing files..."):
        count = ingest_project(abs_path, manager, dry_run=dry_run)
    if not dry_run:
        console.print(f"[green]Successfully ingested {count} files into memory![/green]")


@app.command()
def watch(
    file: Optional[str] = typer.Option(None, help="File to tail and ingest"),
    command: Optional[str] = typer.Option(None, help="Command to run and ingest output"),
    obs_type: str = typer.Option("tool_output", help="Observation type to store"),
    project: Optional[str] = typer.Option(None, help="Project path override"),
    batch_lines: int = typer.Option(20, help="Lines per observation batch"),
    interval: float = typer.Option(1.0, help="Polling interval for file watch"),
    tag: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tags to associate"),
):
    """Watch a file or command output and store it as memory."""
    if not file and not command:
        console.print("[red]Provide --file or --command.[/red]")
        raise typer.Exit(1)
    if file and command:
        console.print("[red]Use only one of --file or --command.[/red]")
        raise typer.Exit(1)

    manager = get_memory_manager()
    project_name = project or os.getcwd()
    manager.start_session(project=project_name, goal="Watch mode")

    def flush(buffer: List[str]) -> None:
        if not buffer:
            return
        payload = "".join(buffer).strip()
        if not payload:
            buffer.clear()
            return
        manager.add_observation(
            content=payload,
            obs_type=obs_type,
            project=project_name,
            tags=tag or [],
            summarize=False,
        )
        buffer.clear()

    try:
        buffer: List[str] = []
        if file:
            with open(file, "r", encoding="utf-8", errors="ignore") as handle:
                handle.seek(0, os.SEEK_END)
                console.print(f"[green]Watching file:[/green] {file}")
                while True:
                    line = handle.readline()
                    if not line:
                        time.sleep(interval)
                        continue
                    buffer.append(line)
                    if len(buffer) >= batch_lines:
                        flush(buffer)
        else:
            import subprocess

            console.print(f"[green]Running and watching:[/green] {command}")
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                buffer.append(line)
                if len(buffer) >= batch_lines:
                    flush(buffer)
            flush(buffer)
    except KeyboardInterrupt:
        flush(buffer)
    finally:
        manager.close_session()


@app.command()
def export(
    path: str = typer.Argument(..., help="Path to write exported JSON"),
    project: Optional[str] = typer.Option(None, help="Project path filter"),
    limit: Optional[int] = typer.Option(None, help="Limit number of observations"),
):
    """Export observations to a JSON file."""
    manager = get_memory_manager()
    data = manager.export_observations(project=project, limit=limit)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    console.print(f"[green]Exported {len(data)} observations to {path}[/green]")


@app.command("import")
def import_memories(
    path: str = typer.Argument(..., help="Path to JSON file to import"),
    project: Optional[str] = typer.Option(None, help="Override project for imported items"),
    dedupe: bool = typer.Option(True, help="Skip duplicates by content hash"),
):
    """Import observations from a JSON export."""
    manager = get_memory_manager()
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        console.print("[red]Import file must contain a list of observations.[/red]")
        raise typer.Exit(1)

    imported = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        obs_type = item.get("type") or "note"
        if not content:
            continue
        manager.add_observation(
            content=content,
            obs_type=obs_type,
            project=project or item.get("project"),
            tags=item.get("tags") or [],
            metadata=item.get("metadata") or {},
            title=item.get("title"),
            summarize=False,
            dedupe=dedupe,
            summary=item.get("summary"),
            created_at=item.get("created_at"),
            importance_score=item.get("importance_score", 0.5),
        )
        imported += 1
    console.print(f"[green]Imported {imported} observations[/green]")


@app.command()
def delete(
    obs_id: str = typer.Argument(..., help="Observation ID to delete"),
    force: bool = typer.Option(False, help="Skip confirmation prompt"),
):
    """Delete a single observation."""
    if not force:
        confirm = typer.confirm(f"Delete observation {obs_id}?")
        if not confirm:
            raise typer.Exit(1)
    manager = get_memory_manager()
    if manager.delete_observation(obs_id):
        console.print(f"[green]Deleted observation {obs_id}[/green]")
    else:
        console.print(f"[yellow]Observation not found: {obs_id}[/yellow]")


@app.command()
def delete_project(
    project: str = typer.Argument(..., help="Project path to delete"),
    force: bool = typer.Option(False, help="Skip confirmation prompt"),
):
    """Delete all observations for a project."""
    if not force:
        confirm = typer.confirm(f"Delete all observations for {project}?")
        if not confirm:
            raise typer.Exit(1)
    manager = get_memory_manager()
    deleted = manager.delete_project(project)
    console.print(f"[green]Deleted {deleted} observations for {project}[/green]")


if __name__ == "__main__":
    app()

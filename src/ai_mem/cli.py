import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config, update_config, validate_config
from .exceptions import ConfigurationError
from .context import build_context, estimate_tokens
from .memory import MemoryManager

app = typer.Typer(help="AI Memory: Persistent memory for any LLM.")
snapshot_app = typer.Typer(help="Snapshot export/import for incremental sync")
console = Console()

app.add_typer(snapshot_app, name="snapshot")


def _infer_format(path: str, fmt: Optional[str], default: str) -> str:
    if fmt:
        return fmt.lower()
    suffix = Path(path).suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    return default

def _write_snapshot_file(path: str, fmt: str, meta: Dict[str, Any], items: List[Dict[str, Any]]) -> None:
    if fmt == "json":
        payload = {"snapshot": meta, "observations": items}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return
    if fmt in {"jsonl", "ndjson"}:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"snapshot": meta}, ensure_ascii=True))
            handle.write("\n")
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=True))
                handle.write("\n")
        return
    if fmt == "csv":
        if not items:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerows([])
            return
        headers = sorted(items[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for item in items:
                writer.writerow(item)
        return
    console.print(f"[red]Unsupported format: {fmt}[/red]")
    raise typer.Exit(1)

def _read_export_file(path: str, fmt: str) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    metadata: Optional[Dict[str, Any]] = None
    items: List[Dict[str, Any]] = []
    if fmt == "json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            metadata = payload.get("snapshot")
            if isinstance(payload.get("observations"), list):
                items = payload["observations"]
            elif isinstance(payload.get("data"), list):
                items = payload["data"]
            else:
                console.print("[red]JSON export must contain an observations list.[/red]")
                raise typer.Exit(1)
        elif isinstance(payload, list):
            items = payload
        else:
            console.print("[red]Invalid export format.[/red]")
            raise typer.Exit(1)
        return items, metadata
    if fmt in {"jsonl", "ndjson"}:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                parsed: Dict[str, Any]
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    console.print("[red]Invalid JSON line detected in export file.[/red]")
                    raise typer.Exit(1)
                if metadata is None and isinstance(parsed, dict):
                    header = parsed.get("snapshot")
                    if isinstance(header, dict):
                        metadata = header
                        continue
                items.append(parsed)
        return items, metadata
    if fmt == "csv":
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                items.append(row)
        return items, metadata
    console.print(f"[red]Unsupported format: {fmt}[/red]")
    raise typer.Exit(1)

def _snapshot_key(item: Dict[str, Any]) -> str:
    content_hash = item.get("content_hash")
    if content_hash:
        return str(content_hash)
    parts = (
        str(item.get("project") or ""),
        str(item.get("session_id") or ""),
        str(item.get("content") or ""),
        str(item.get("created_at") or ""),
    )
    return "|".join(parts)


def _import_items(
    manager: MemoryManager,
    items: List[Dict[str, Any]],
    project_override: Optional[str],
    dedupe: bool,
) -> int:
    imported = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        obs_type = item.get("type") or item.get("obs_type") or "note"
        if not content:
            continue
        item_project = item.get("project") or None
        session_id = item.get("session_id") or None
        if project_override and item_project and project_override != item_project:
            session_id = None
        tags = _parse_tags_value(item.get("tags"))
        metadata = _parse_metadata_value(item.get("metadata"))
        assets = _parse_assets_value(item.get("assets"))
        created_at = _parse_float(item.get("created_at"))
        importance_score = _parse_float(item.get("importance_score"))
        result = manager.add_observation(
            content=content,
            obs_type=obs_type,
            project=project_override or item_project,
            session_id=session_id,
            tags=tags,
            metadata=metadata,
            title=item.get("title") or None,
            summarize=False,
            dedupe=dedupe,
            summary=item.get("summary") or None,
            created_at=created_at,
            importance_score=importance_score if importance_score is not None else 0.5,
            assets=assets,
        )
        if result:
            imported += 1
    return imported


def _parse_tags_value(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_metadata_value(value: object) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return {}
    return {}


def _parse_assets_value(value: object) -> List[Dict[str, Any]]:
    if not value:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    return []


def _parse_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _env_bool(name: str) -> Optional[bool]:
    value = os.environ.get(name)
    if value is None:
        return None
    text = value.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return True


def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_list(name: str) -> List[str]:
    value = os.environ.get(name)
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_metadata_pair(value: str) -> Dict[str, Any]:
    text = value.strip()
    if not text:
        return {}
    if "=" in text:
        key, raw = text.split("=", 1)
    else:
        return {text: ""}
    try:
        parsed = json.loads(raw)
        return {key.strip(): parsed}
    except json.JSONDecodeError:
        return {key.strip(): raw}


def _load_json_file(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data
    except Exception as exc:
        console.print(f"[yellow]Failed to load JSON from {path}: {exc}[/yellow]")
    return {}


def _read_asset_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except Exception as exc:
        console.print(f"[yellow]Failed to read asset {path}: {exc}[/yellow]")
    return None


def _build_asset_entries(paths: Optional[List[str]], asset_type: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not paths:
        return entries
    for item in paths:
        if not item:
            continue
        entry = {
            "type": asset_type,
            "name": Path(item).name,
            "path": item,
        }
        content = _read_asset_file(item)
        if content is not None:
            entry["content"] = content
        entries.append(entry)
    return entries


def _read_content(value: Optional[str], content_file: Optional[str]) -> str:
    if value is not None:
        return value
    if content_file:
        if content_file == "-":
            return sys.stdin.read()
        try:
            with open(content_file, "r", encoding="utf-8") as handle:
                return handle.read()
        except OSError:
            return ""
    env_value = os.environ.get("AI_MEM_CONTENT")
    return env_value or ""


def get_memory_manager() -> MemoryManager:
    try:
        return MemoryManager()
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1)


@app.command()
def config(
    llm_provider: Optional[str] = typer.Option(
        None, help="LLM provider (gemini, anthropic, bedrock, azure-openai, openai-compatible, vllm)"
    ),
    llm_model: Optional[str] = typer.Option(None, help="LLM model name"),
    llm_api_key: Optional[str] = typer.Option(None, help="LLM API key"),
    llm_base_url: Optional[str] = typer.Option(None, help="LLM base URL (OpenAI-compatible)"),
    llm_api_version: Optional[str] = typer.Option(None, help="LLM API version (Azure OpenAI)"),
    embeddings_provider: Optional[str] = typer.Option(
        None, help="Embeddings provider (fastembed, gemini, bedrock, azure-openai, openai-compatible, auto)"
    ),
    embeddings_model: Optional[str] = typer.Option(None, help="Embeddings model name"),
    embeddings_api_key: Optional[str] = typer.Option(None, help="Embeddings API key"),
    embeddings_base_url: Optional[str] = typer.Option(None, help="Embeddings base URL (OpenAI-compatible)"),
    embeddings_api_version: Optional[str] = typer.Option(None, help="Embeddings API version (Azure OpenAI)"),
    data_dir: Optional[str] = typer.Option(None, help="Data directory for SQLite + vector store"),
    sqlite_path: Optional[str] = typer.Option(None, help="Override SQLite database path"),
    vector_dir: Optional[str] = typer.Option(None, help="Override vector store directory"),
    vector_provider: Optional[str] = typer.Option(
        None, help="Vector store provider (chroma or pgvector)"
    ),
    vector_chroma_collection: Optional[str] = typer.Option(
        None, help="Chroma collection name used by the chroma provider"
    ),
    pgvector_dsn: Optional[str] = typer.Option(None, help="Postgres DSN for pgvector"),
    pgvector_table: Optional[str] = typer.Option(None, help="pgvector table name"),
    pgvector_dimension: Optional[int] = typer.Option(None, help="pgvector embedding dimension"),
    pgvector_index_type: Optional[str] = typer.Option(None, help="pgvector index type (ivfflat/hnsw)"),
    pgvector_lists: Optional[int] = typer.Option(None, help="pgvector ivfflat lists count"),
    context_total: Optional[int] = typer.Option(None, help="Context index item count"),
    context_full: Optional[int] = typer.Option(None, help="Context full item count"),
    context_types: Optional[str] = typer.Option(None, help="Comma-separated context observation types"),
    context_tags: Optional[str] = typer.Option(None, help="Comma-separated context tag filters"),
    context_full_field: Optional[str] = typer.Option(None, help="Context full field (content or summary)"),
    context_show_tokens: Optional[bool] = typer.Option(
        None, "--context-show-tokens/--context-hide-tokens", help="Toggle context token estimates"
    ),
    context_wrap: Optional[bool] = typer.Option(
        None, "--context-wrap/--context-no-wrap", help="Toggle <ai-mem-context> wrapping"
    ),
    show: bool = typer.Option(False, help="Show current configuration"),
):
    """Configure providers, embeddings, and storage paths."""
    if show or not any(
        [
            llm_provider,
            llm_model,
            llm_api_key,
            llm_base_url,
            llm_api_version,
            embeddings_provider,
            embeddings_model,
            embeddings_api_key,
            embeddings_base_url,
            embeddings_api_version,
            data_dir,
            sqlite_path,
            vector_dir,
            vector_provider,
            vector_chroma_collection,
            pgvector_dsn,
            pgvector_table,
            pgvector_dimension,
            pgvector_index_type,
            pgvector_lists,
            context_total,
            context_full,
            context_types,
            context_tags,
            context_full_field,
            context_show_tokens,
            context_wrap,
        ]
    ):
        config_data = load_config().model_dump()
        console.print(json.dumps(config_data, indent=2))
        return

    patch = {"llm": {}, "embeddings": {}, "storage": {}, "context": {}, "vector": {}}
    if llm_provider:
        patch["llm"]["provider"] = llm_provider
    if llm_model:
        patch["llm"]["model"] = llm_model
    if llm_api_key:
        patch["llm"]["api_key"] = llm_api_key
    if llm_base_url:
        patch["llm"]["base_url"] = llm_base_url
    if llm_api_version:
        patch["llm"]["api_version"] = llm_api_version
    if embeddings_provider:
        patch["embeddings"]["provider"] = embeddings_provider
    if embeddings_model:
        patch["embeddings"]["model"] = embeddings_model
    if embeddings_api_key:
        patch["embeddings"]["api_key"] = embeddings_api_key
    if embeddings_base_url:
        patch["embeddings"]["base_url"] = embeddings_base_url
    if embeddings_api_version:
        patch["embeddings"]["api_version"] = embeddings_api_version
    if data_dir:
        patch["storage"]["data_dir"] = data_dir
    if sqlite_path:
        patch["storage"]["sqlite_path"] = sqlite_path
    if vector_dir:
        patch["storage"]["vector_dir"] = vector_dir
    if vector_provider:
        patch["vector"]["provider"] = vector_provider
    if vector_chroma_collection:
        patch["vector"]["chroma_collection"] = vector_chroma_collection
    if pgvector_dsn:
        patch["vector"]["pgvector_dsn"] = pgvector_dsn
    if pgvector_table:
        patch["vector"]["pgvector_table"] = pgvector_table
    if pgvector_dimension is not None:
        patch["vector"]["pgvector_dimension"] = pgvector_dimension
    if pgvector_index_type:
        patch["vector"]["pgvector_index_type"] = pgvector_index_type
    if pgvector_lists is not None:
        patch["vector"]["pgvector_lists"] = pgvector_lists
    if context_total is not None:
        patch["context"]["total_observation_count"] = context_total
    if context_full is not None:
        patch["context"]["full_observation_count"] = context_full
    if context_types is not None:
        patch["context"]["observation_types"] = [item.strip() for item in context_types.split(",") if item.strip()]
    if context_tags is not None:
        patch["context"]["tag_filters"] = [item.strip() for item in context_tags.split(",") if item.strip()]
    if context_full_field:
        patch["context"]["full_observation_field"] = context_full_field
    if context_show_tokens is not None:
        patch["context"]["show_token_estimates"] = context_show_tokens
    if context_wrap is not None:
        patch["context"]["wrap_context_tag"] = context_wrap

    update_config(patch)
    console.print("[green]Configuration updated.[/green]")


@app.command(name="config-validate")
def config_validate(
    strict: bool = typer.Option(False, "--strict", help="Treat warnings as errors"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only show errors, suppress warnings"),
):
    """Validate the current configuration.

    Checks for missing required values, invalid settings, and provider-specific requirements.
    """
    try:
        config = load_config()
        is_valid, warnings = validate_config(config, strict=strict)

        if not quiet and warnings:
            console.print("[yellow]Warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  • {warning}")

        console.print("[green]Configuration is valid.[/green]")

    except ConfigurationError as e:
        console.print(f"[red]Configuration validation failed:[/red]")
        if hasattr(e, "details") and e.details:
            errors = e.details.get("errors", [])
            for error in errors:
                console.print(f"  [red]✗[/red] {error}")
            warnings = e.details.get("warnings", [])
            if not quiet:
                for warning in warnings:
                    console.print(f"  [yellow]![/yellow] {warning}")
        else:
            console.print(f"  {e}")
        raise typer.Exit(1)


@app.command()
def add(
    content: str = typer.Argument(..., help="Text to store as memory"),
    obs_type: str = typer.Option("note", help="Observation type"),
    project: Optional[str] = typer.Option(None, help="Project path override"),
    session_id: Optional[str] = typer.Option(None, help="Session ID override"),
    tag: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tags to associate"),
    no_summary: bool = typer.Option(False, help="Disable summarization"),
    metadata: Optional[List[str]] = typer.Option(None, "--metadata", "-m", help="Metadata key=value pairs"),
    metadata_file: Optional[str] = typer.Option(None, help="JSON file with metadata"),
    attach_file: Optional[List[str]] = typer.Option(None, "--file", "-f", help="Attach file asset paths"),
    attach_diff: Optional[List[str]] = typer.Option(None, "--diff", help="Attach diff/patch asset paths"),
    event_id: Optional[str] = typer.Option(None, "--event-id", help="Event ID for idempotency (skip if already processed)"),
    host: Optional[str] = typer.Option(None, "--host", help="Host identifier (claude-code, gemini, cursor, etc.)"),
):
    """Store a new memory observation."""
    async def _run():
        manager = get_memory_manager()
        await manager.initialize()
        try:
            if session_id and not project:
                current_project = None
            else:
                current_project = project

            metadata_payload: Dict[str, Any] = {}
            for item in metadata or []:
                metadata_payload.update(_parse_metadata_pair(item))
            if metadata_file:
                metadata_payload.update(_load_json_file(metadata_file))
            assets: List[Dict[str, Any]] = []
            assets.extend(_build_asset_entries(attach_file, "file"))
            assets.extend(_build_asset_entries(attach_diff, "diff"))
            obs = await manager.add_observation(
                content=content,
                obs_type=obs_type,
                project=current_project,
                session_id=session_id,
                tags=tag or [],
                summarize=not no_summary,
                metadata=metadata_payload or None,
                assets=assets or None,
                event_id=event_id,
                host=host,
            )
            if not obs:
                console.print("[yellow]Skipped: content marked as private.[/yellow]")
                return
            console.print(f"[green]Memory stored![/green] (ID: {obs.id})")
        finally:
            await manager.close()

    asyncio.run(_run())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results"),
    project: Optional[str] = typer.Option(None, help="Project path filter"),
    obs_type: Optional[str] = typer.Option(None, help="Observation type filter"),
    session_id: Optional[str] = typer.Option(None, help="Session ID filter"),
    all_projects: bool = typer.Option(False, help="Search across all projects"),
    date_start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD or epoch)"),
    date_end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD or epoch)"),
    since: Optional[str] = typer.Option(None, help="Relative start window (e.g. 24h, 7d)"),
    tag: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tag filters (any match)"),
    output: str = typer.Option("text", "--format", "-f", help="Output format: text, json"),
    show_tokens: bool = typer.Option(False, "--show-tokens/--hide-tokens", help="Show token estimates"),
):
    """Search memory index (compact results)."""
    async def _run():
        manager = get_memory_manager()
        await manager.initialize()
        try:
            if session_id:
                current_project = None
            elif not project and not all_projects:
                current_project = os.getcwd()
            else:
                current_project = project

            if all_projects:
                current_project = None
            results = await manager.search(
                query,
                limit=limit,
                project=current_project,
                obs_type=obs_type,
                session_id=session_id,
                date_start=date_start,
                date_end=date_end,
                since=since,
                tag_filters=tag,
            )
            if output == "json":
                payload = [item.model_dump() for item in results]
                if show_tokens:
                    for row, item in zip(payload, results):
                        row["token_estimate"] = estimate_tokens(item.summary or "")
                print(json.dumps(payload, indent=2))
                return

            table = Table(title=f"Search results for: {query}")
            table.add_column("ID", style="dim")
            table.add_column("Summary", style="cyan")
            if show_tokens:
                table.add_column("Tokens", style="dim", justify="right")
            table.add_column("Type", style="magenta")
            table.add_column("Project", style="green")
            for item in results:
                row = [item.id, item.summary]
                if show_tokens:
                    row.append(str(estimate_tokens(item.summary or "")))
                row.extend([item.type or "-", item.project])
                table.add_row(*row)
            console.print(table)
        finally:
            await manager.close()

    asyncio.run(_run())


@app.command(name="mem-search")
def mem_search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results"),
    project: Optional[str] = typer.Option(None, help="Project path filter"),
    obs_type: Optional[str] = typer.Option(None, help="Observation type filter"),
    session_id: Optional[str] = typer.Option(None, help="Session ID filter"),
    all_projects: bool = typer.Option(False, help="Search across all projects"),
    date_start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD or epoch)"),
    date_end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD or epoch)"),
    since: Optional[str] = typer.Option(None, help="Relative start window (e.g. 24h, 7d)"),
    tag: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tag filters (any match)"),
    output: str = typer.Option("text", "--format", "-f", help="Output format: text, json"),
    show_tokens: bool = typer.Option(False, "--show-tokens/--hide-tokens", help="Show token estimates"),
):
    """Alias for search (friendly name for MCP parity)."""
    return search(
        query=query,
        limit=limit,
        project=project,
        obs_type=obs_type,
        session_id=session_id,
        all_projects=all_projects,
        date_start=date_start,
        date_end=date_end,
        since=since,
        tag=tag,
        output=output,
        show_tokens=show_tokens,
    )


@app.command()
def timeline(
    anchor_id: Optional[str] = typer.Option(None, help="Anchor observation ID"),
    query: Optional[str] = typer.Option(None, help="Query to find anchor"),
    depth_before: int = typer.Option(3, help="Observations before anchor"),
    depth_after: int = typer.Option(3, help="Observations after anchor"),
    project: Optional[str] = typer.Option(None, help="Project path filter"),
    all_projects: bool = typer.Option(False, help="Search across all projects"),
    obs_type: Optional[str] = typer.Option(None, help="Observation type filter"),
    session_id: Optional[str] = typer.Option(None, help="Session ID filter"),
    date_start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD or epoch)"),
    date_end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD or epoch)"),
    since: Optional[str] = typer.Option(None, help="Relative start window (e.g. 24h, 7d)"),
    tag: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tag filters (any match)"),
    output: str = typer.Option("text", "--format", "-f", help="Output format: text, json"),
    show_tokens: bool = typer.Option(False, "--show-tokens/--hide-tokens", help="Show token estimates"),
):
    """Get chronological context around an observation."""
    async def _run():
        manager = get_memory_manager()
        await manager.initialize()
        try:
            if session_id:
                current_project = None
            elif not project and not all_projects:
                current_project = os.getcwd()
            else:
                current_project = project

            if all_projects:
                current_project = None
            results = await manager.timeline(
                anchor_id=anchor_id,
                query=query,
                depth_before=depth_before,
                depth_after=depth_after,
                project=current_project,
                obs_type=obs_type,
                session_id=session_id,
                date_start=date_start,
                date_end=date_end,
                since=since,
                tag_filters=tag,
            )
            if output == "json":
                payload = [item.model_dump() for item in results]
                if show_tokens:
                    for row, item in zip(payload, results):
                        row["token_estimate"] = estimate_tokens(item.summary or "")
                print(json.dumps(payload, indent=2))
                return

            table = Table(title="Timeline")
            table.add_column("ID", style="dim")
            table.add_column("Summary", style="cyan")
            if show_tokens:
                table.add_column("Tokens", style="dim", justify="right")
            table.add_column("Type", style="magenta")
            table.add_column("Project", style="green")
            for item in results:
                row = [item.id, item.summary]
                if show_tokens:
                    row.append(str(estimate_tokens(item.summary or "")))
                row.extend([item.type or "-", item.project])
                table.add_row(*row)
            console.print(table)
        finally:
            await manager.close()

    asyncio.run(_run())


@app.command()
def context(
    query: Optional[str] = typer.Option(None, help="Optional search query"),
    project: Optional[str] = typer.Option(None, help="Project path filter"),
    all_projects: bool = typer.Option(False, help="Include all projects"),
    obs_type: Optional[str] = typer.Option(None, help="Observation type filter"),
    obs_types: Optional[str] = typer.Option(None, help="Comma-separated observation types"),
    tag: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tag filters (any match)"),
    session_id: Optional[str] = typer.Option(None, help="Session ID filter"),
    total: Optional[int] = typer.Option(None, help="Max index items"),
    full: Optional[int] = typer.Option(None, help="Max full items"),
    full_field: Optional[str] = typer.Option(None, help="Full context field (content or summary)"),
    show_tokens: Optional[bool] = typer.Option(
        None, "--show-tokens/--hide-tokens", help="Toggle token estimates"
    ),
    no_wrap: bool = typer.Option(False, help="Disable <ai-mem-context> wrapper"),
    output: str = typer.Option("text", "--format", "-f", help="Output format: text, json"),
):
    """Generate formatted context for injection into other LLMs."""
    async def _run():
        manager = get_memory_manager()
        await manager.initialize()
        try:
            if session_id:
                current_project = None
            elif not project and not all_projects:
                current_project = os.getcwd()
            else:
                current_project = project

            if all_projects:
                current_project = None
            context_text, meta = await build_context(
                manager,
                project=current_project,
                session_id=session_id,
                query=query,
                obs_type=obs_type,
                obs_types=[item.strip() for item in obs_types.split(",") if item.strip()] if obs_types else None,
                tag_filters=tag,
                total_count=total,
                full_count=full,
                full_field=full_field,
                show_tokens=show_tokens,
                wrap=not no_wrap,
            )
            if output == "json":
                print(json.dumps({"context": context_text, "metadata": meta}, indent=2))
                return
            console.print(context_text)
        finally:
            await manager.close()

    asyncio.run(_run())


@app.command()
def endless(
    query: Optional[str] = typer.Option(None, help="Optional search query"),
    project: Optional[str] = typer.Option(None, help="Project path filter"),
    session_id: Optional[str] = typer.Option(None, help="Session ID filter"),
    obs_type: Optional[str] = typer.Option(None, help="Observation type filter"),
    tag: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tag filters (any match)"),
    total: Optional[int] = typer.Option(None, help="Base index count"),
    full: Optional[int] = typer.Option(None, help="Base full count"),
    full_field: Optional[str] = typer.Option(None, help="Full context field (content or summary)"),
    show_tokens: Optional[bool] = typer.Option(None, "--show-tokens/--hide-tokens", help="Toggle token estimates"),
    no_wrap: bool = typer.Option(False, help="Disable <ai-mem-context> wrapper"),
    interval: int = typer.Option(60, help="Seconds between context refreshes"),
    token_limit: Optional[int] = typer.Option(1200, help="Target total token count (index+full)"),
    json_output: bool = typer.Option(False, "--json", help="Print JSON per iteration"),
):
    """Run Endless Mode: auto-refresh context with rolling window control."""
    async def _run():
        manager = get_memory_manager()
        await manager.initialize()
        
        nonlocal project
        if session_id:
            project = None
        elif not project:
            project = os.getcwd()
            
        console.print("[cyan]Starting Endless Mode (Ctrl+C to stop)[/cyan]")
        current_total = total if total is not None else manager.config.context.total_observation_count
        full_count = full if full is not None else manager.config.context.full_observation_count
        try:
            while True:
                context_text, meta = await build_context(
                    manager,
                    project=project,
                    session_id=session_id,
                    query=query,
                    obs_type=obs_type,
                    tag_filters=tag,
                    total_count=current_total,
                    full_count=full_count,
                    full_field=full_field,
                    show_tokens=show_tokens,
                    wrap=not no_wrap,
                )
                tokens = meta.get("tokens", {})
                total_tokens = tokens.get("total") or 0
                scoreboard = meta.get("scoreboard") or {}
                cache = manager.search_cache_summary()
                if json_output:
                    payload = {
                        "context": context_text,
                        "metadata": meta,
                        "cache": cache,
                    }
                    console.print(json.dumps(payload, indent=2))
                else:
                    console.print(context_text)
                    console.print(f"[green]Total tokens: {total_tokens} (index {tokens.get('index', 0)} + full {tokens.get('full', 0)})[/green]")
                    if scoreboard:
                        table = Table(title="Scoreboard (top 5)")
                        table.add_column("ID", style="dim")
                        table.add_column("FTS", justify="right")
                        table.add_column("Vec", justify="right")
                        table.add_column("Rec", justify="right")
                        for obs_id, data in list(scoreboard.items())[:5]:
                            table.add_row(
                                obs_id,
                                f"{(data.get('fts_score') or 0):.3f}",
                                f"{(data.get('vector_score') or 0):.3f}",
                                f"{(data.get('recency_factor') or 0):.3f}",
                            )
                        console.print(table)
                    console.print(f"[yellow]Cache hits: {cache.get('hits')} misses: {cache.get('misses')} (hit rate {round(((cache.get('hits') or 0) / max(1, (cache.get('hits', 0) + cache.get('misses', 0))))*100, 1)}%)[/yellow]")
                if token_limit:
                    if total_tokens > token_limit:
                        current_total = max(1, current_total - 1)
                    elif total_tokens < token_limit * 0.9:
                        current_total += 1
                await asyncio.sleep(max(5, interval))
        except asyncio.CancelledError:
             pass
        finally:
             await manager.close()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("[yellow]\nEndless Mode stopped.[/yellow]")


@app.command()
def hook(
    event: str = typer.Argument(
        ...,
        help="Hook event: session_start, user_prompt, assistant_response, tool_output, session_end",
    ),
    content: Optional[str] = typer.Option(None, "--content", "-c", help="Content to store"),
    content_file: Optional[str] = typer.Option(None, help="Read content from file (use '-' for stdin)"),
    project: Optional[str] = typer.Option(None, help="Project path override"),
    session_id: Optional[str] = typer.Option(None, help="Session ID override"),
    obs_type: Optional[str] = typer.Option(None, help="Observation type override"),
    tag: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tags to associate"),
    no_summary: Optional[bool] = typer.Option(
        None, "--no-summary/--summary", help="Disable or enable summarization"
    ),
    metadata: Optional[List[str]] = typer.Option(None, "--metadata", "-m", help="Metadata key=value pairs"),
    metadata_file: Optional[str] = typer.Option(None, help="JSON file with metadata"),
    asset_file: Optional[List[str]] = typer.Option(None, "--file", "-f", help="Attach file asset paths"),
    asset_diff: Optional[List[str]] = typer.Option(None, "--diff", help="Attach diff/patch asset paths"),
    query: Optional[str] = typer.Option(None, help="Context query (session_start)"),
    total: Optional[int] = typer.Option(None, help="Context index count"),
    full: Optional[int] = typer.Option(None, help="Context full count"),
    full_field: Optional[str] = typer.Option(None, help="Context full field (content or summary)"),
    context_tag: Optional[List[str]] = typer.Option(
        None, "--context-tag", help="Context tag filters (repeatable)"
    ),
    no_wrap: bool = typer.Option(False, help="Disable <ai-mem-context> wrapper"),
    session_tracking: Optional[bool] = typer.Option(
        None, "--session-tracking/--no-session-tracking", help="Start/end sessions automatically"
    ),
    summary_on_end: Optional[bool] = typer.Option(
        None, "--summary-on-end/--no-summary-on-end", help="Summarize on session_end"
    ),
    summary_count: int = typer.Option(20, help="Summary count"),
    summary_obs_type: Optional[str] = typer.Option(None, help="Summary observation type filter"),
):
    """Run hook logic without shell scripts."""
    async def _run():
        manager = get_memory_manager()
        await manager.initialize()
        try:
            normalized = event.strip().lower().replace("-", "_")

            project_name = project or os.environ.get("AI_MEM_PROJECT") or os.getcwd()
            session_value = session_id or os.environ.get("AI_MEM_SESSION_ID") or None
            tags = tag if tag is not None else _env_list("AI_MEM_TAGS")

            env_tracking = _env_bool("AI_MEM_SESSION_TRACKING") or False
            track_sessions = session_tracking if session_tracking is not None else env_tracking

            if normalized == "session_start":
                if track_sessions:
                    await manager.start_session(project=project_name, goal="", session_id=session_value)

                query_value = query or os.environ.get("AI_MEM_QUERY") or None
                total_value = total if total is not None else _env_int("AI_MEM_CONTEXT_TOTAL")
                full_value = full if full is not None else _env_int("AI_MEM_CONTEXT_FULL")
                full_field_value = full_field or os.environ.get("AI_MEM_CONTEXT_FULL_FIELD") or None
                context_tags = context_tag if context_tag is not None else _env_list("AI_MEM_CONTEXT_TAGS")
                if context_tag is None and not context_tags:
                    context_tags = None
                env_no_wrap = _env_bool("AI_MEM_CONTEXT_NO_WRAP") or False
                wrap = not (no_wrap or env_no_wrap)

                context_text, _ = await build_context(
                    manager,
                    project=None if session_value else project_name,
                    session_id=session_value,
                    query=query_value,
                    obs_type=obs_type,
                    tag_filters=context_tags,
                    total_count=total_value,
                    full_count=full_value,
                    full_field=full_field_value,
                    wrap=wrap,
                )
                sys.stdout.write(context_text)
                return

            content_text = _read_content(content, content_file)
            if not content_text.strip():
                return

            default_types = {
                "user_prompt": "interaction",
                "assistant_response": "interaction",
                "tool_output": "tool_output",
                "session_end": "note",
            }
            default_tags = {
                "user_prompt": "user",
                "assistant_response": "assistant",
                "tool_output": "tool",
                "session_end": "session_end",
            }
            if normalized not in default_types:
                console.print(f"[red]Unknown hook event: {event}[/red]")
                raise typer.Exit(1)

            obs_type_value = obs_type or os.environ.get("AI_MEM_OBS_TYPE") or default_types[normalized]
            env_no_summary = _env_bool("AI_MEM_NO_SUMMARY")
            if no_summary is None:
                if env_no_summary is None:
                    no_summary_value = normalized == "tool_output"
                else:
                    no_summary_value = env_no_summary
            else:
                no_summary_value = no_summary

            event_tag = default_tags[normalized]
            all_tags = [event_tag] if event_tag else []
            all_tags.extend(tags or [])
            deduped_tags = []
            seen = set()
            for item in all_tags:
                if item not in seen:
                    seen.add(item)
                    deduped_tags.append(item)

            env_metadata = _parse_metadata_value(os.environ.get("AI_MEM_METADATA"))
            metadata_payload: Dict[str, Any] = {}
            metadata_payload.update(env_metadata)
            for item in metadata or []:
                metadata_payload.update(_parse_metadata_pair(item))
            if metadata_file:
                metadata_payload.update(_load_json_file(metadata_file))
            assets: List[Dict[str, Any]] = []
            assets.extend(_build_asset_entries(_env_list("AI_MEM_ASSET_FILES"), "file"))
            assets.extend(_build_asset_entries(_env_list("AI_MEM_ASSET_DIFFS"), "diff"))
            assets.extend(_build_asset_entries(asset_file, "file"))
            assets.extend(_build_asset_entries(asset_diff, "diff"))

            await manager.add_observation(
                content=content_text,
                obs_type=obs_type_value,
                project=None if session_value else project_name,
                session_id=session_value,
                tags=deduped_tags,
                summarize=not no_summary_value,
                metadata=metadata_payload or None,
                assets=assets or None,
            )

            if normalized == "session_end":
                if track_sessions:
                    if session_value:
                        await manager.end_session(session_value)
                    else:
                        await manager.end_latest_session(project_name)

                env_summary_on_end = _env_bool("AI_MEM_SUMMARY_ON_END") or False
                should_summarize = summary_on_end if summary_on_end is not None else env_summary_on_end
                if should_summarize:
                    env_summary_count = _env_int("AI_MEM_SUMMARY_COUNT")
                    if env_summary_count is not None and summary_count == 20:
                        summary_count = env_summary_count
                    summary_type = summary_obs_type or os.environ.get("AI_MEM_SUMMARY_OBS_TYPE")
                    await manager.summarize_project(
                        project=None if session_value else project_name,
                        session_id=session_value,
                        limit=summary_count,
                        obs_type=summary_type,
                        store=True,
                    )
        finally:
            await manager.close()

    asyncio.run(_run())


@app.command()
def stats(
    project: Optional[str] = typer.Option(None, help="Project path filter"),
    all_projects: bool = typer.Option(False, help="Include all projects"),
    obs_type: Optional[str] = typer.Option(None, help="Observation type filter"),
    session_id: Optional[str] = typer.Option(None, help="Session ID filter"),
    date_start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD or epoch)"),
    date_end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD or epoch)"),
    since: Optional[str] = typer.Option(None, help="Relative start window (e.g. 24h, 7d)"),
    tag: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tag filters (any match)"),
    tag_limit: int = typer.Option(10, help="Max tags to include"),
    day_limit: int = typer.Option(14, help="Max days to include"),
    type_tag_limit: int = typer.Option(3, help="Max tags per type to include"),
    output: str = typer.Option("text", "--format", "-f", help="Output format: text, json"),
):
    """Show observation counts grouped by type and project."""
    async def _run():
        manager = get_memory_manager()
        await manager.initialize()
        try:
            nonlocal project
            if session_id:
                project = None
            elif not project and not all_projects:
                project = os.getcwd()
            if all_projects:
                project = None
            data = await manager.get_stats(
                project=project,
                obs_type=obs_type,
                session_id=session_id,
                date_start=date_start,
                date_end=date_end,
                since=since,
                tag_filters=tag,
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
        finally:
            await manager.close()

    asyncio.run(_run())


@app.command()
def tags(
    project: Optional[str] = typer.Option(None, help="Project path filter"),
    all_projects: bool = typer.Option(False, help="Include all projects"),
    session_id: Optional[str] = typer.Option(None, help="Session ID filter"),
    obs_type: Optional[str] = typer.Option(None, help="Observation type filter"),
    date_start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD or epoch)"),
    date_end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD or epoch)"),
    tag: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tags to filter by (any match)"),
    limit: int = typer.Option(50, help="Limit number of tags returned"),
    output: str = typer.Option("text", "--format", "-f", help="Output format: text, json"),
):
    """List tags with usage counts."""
    async def _run():
        manager = get_memory_manager()
        await manager.initialize()
        try:
            nonlocal project
            if session_id:
                project = None
            elif not project and not all_projects:
                project = os.getcwd()
            if all_projects:
                project = None
            tags_list = await manager.list_tags(
                project=project,
                session_id=session_id,
                obs_type=obs_type,
                date_start=date_start,
                date_end=date_end,
                tag_filters=tag,
                limit=limit,
            )
            if output == "json":
                print(json.dumps(tags_list, indent=2))
                return
            if not tags_list:
                console.print("[yellow]No tags found.[/yellow]")
                return
            table = Table(title="Tags")
            table.add_column("Tag", style="cyan")
            table.add_column("Count", style="green")
            for item in tags_list:
                table.add_row(item.get("tag", "-"), str(item.get("count", 0)))
            console.print(table)
        finally:
            await manager.close()

    asyncio.run(_run())


@app.command(name="tag-add")
def tag_add(
    tag_value: str = typer.Argument(..., help="Tag to add"),
    project: Optional[str] = typer.Option(None, help="Project path filter"),
    all_projects: bool = typer.Option(False, help="Include all projects"),
    session_id: Optional[str] = typer.Option(None, help="Session ID filter"),
    obs_type: Optional[str] = typer.Option(None, help="Observation type filter"),
    date_start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD or epoch)"),
    date_end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD or epoch)"),
    filter_tag: Optional[List[str]] = typer.Option(None, "--filter-tag", help="Additional tag filter (any match)"),
):
    """Add a tag across matching observations."""
    async def _run():
        manager = get_memory_manager()
        await manager.initialize()
        try:
            nonlocal project
            if session_id:
                project = None
            elif not project and not all_projects:
                project = os.getcwd()
            if all_projects:
                project = None
            updated = await manager.add_tag(
                tag=tag_value,
                project=project,
                session_id=session_id,
                obs_type=obs_type,
                date_start=date_start,
                date_end=date_end,
                tag_filters=filter_tag,
            )
            console.print(f"[green]Added tag to {updated} observations[/green]")
        finally:
            await manager.close()

    asyncio.run(_run())


@app.command(name="tag-rename")
def tag_rename(
    old_tag: str = typer.Argument(..., help="Tag to rename"),
    new_tag: str = typer.Argument(..., help="New tag value"),
    project: Optional[str] = typer.Option(None, help="Project path filter"),
    all_projects: bool = typer.Option(False, help="Include all projects"),
    session_id: Optional[str] = typer.Option(None, help="Session ID filter"),
    obs_type: Optional[str] = typer.Option(None, help="Observation type filter"),
    date_start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD or epoch)"),
    date_end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD or epoch)"),
    filter_tag: Optional[List[str]] = typer.Option(None, "--filter-tag", help="Additional tag filter (any match)"),
):
    """Rename a tag across matching observations."""
    async def _run():
        manager = get_memory_manager()
        await manager.initialize()
        try:
            nonlocal project
            if session_id:
                project = None
            elif not project and not all_projects:
                project = os.getcwd()
            if all_projects:
                project = None
            updated = await manager.rename_tag(
                old_tag=old_tag,
                new_tag=new_tag,
                project=project,
                session_id=session_id,
                obs_type=obs_type,
                date_start=date_start,
                date_end=date_end,
                tag_filters=filter_tag,
            )
            console.print(f"[green]Renamed tag in {updated} observations[/green]")
        finally:
            await manager.close()

    asyncio.run(_run())


@app.command(name="tag-delete")
def tag_delete(
    tag_value: str = typer.Argument(..., help="Tag to delete"),
    project: Optional[str] = typer.Option(None, help="Project path filter"),
    all_projects: bool = typer.Option(False, help="Include all projects"),
    session_id: Optional[str] = typer.Option(None, help="Session ID filter"),
    obs_type: Optional[str] = typer.Option(None, help="Observation type filter"),
    date_start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD or epoch)"),
    date_end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD or epoch)"),
    force: bool = typer.Option(False, help="Skip confirmation prompt"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without actually deleting"),
    filter_tag: Optional[List[str]] = typer.Option(None, "--filter-tag", help="Additional tag filter (any match)"),
):
    """Delete a tag across matching observations."""
    async def _run():
        manager = get_memory_manager()
        await manager.initialize()
        try:
            nonlocal project
            if session_id:
                project = None
            elif not project and not all_projects:
                project = os.getcwd()
            if all_projects:
                project = None

            if dry_run:
                # Query observations that have this tag to show what would be affected
                # We need to include the tag in filter_tag to find matching observations
                tag_filters_with_target = list(filter_tag) if filter_tag else []
                tag_filters_with_target.append(tag_value)
                results = await manager.list_observations(
                    project=project,
                    session_id=session_id,
                    obs_type=obs_type,
                    date_start=date_start,
                    date_end=date_end,
                    tags=tag_filters_with_target,
                    limit=1000,
                )
                count = len(results)
                console.print(f"[yellow]DRY RUN - Would remove tag '{tag_value}' from {count} observation(s)[/yellow]")
                if count > 0 and count <= 10:
                    console.print("[dim]Affected observations:[/dim]")
                    for obs in results:
                        console.print(f"  - {obs.get('id')}: {obs.get('type', 'unknown')} ({obs.get('project', 'no project')})")
                elif count > 10:
                    console.print(f"[dim]First 10 of {count} affected observations:[/dim]")
                    for obs in results[:10]:
                        console.print(f"  - {obs.get('id')}: {obs.get('type', 'unknown')} ({obs.get('project', 'no project')})")
                return

            if not force:
                confirm = typer.confirm(f"Delete tag '{tag_value}' from matching observations?")
                if not confirm:
                    raise typer.Exit(1)

            updated = await manager.delete_tag(
                tag=tag_value,
                project=project,
                session_id=session_id,
                obs_type=obs_type,
                date_start=date_start,
                date_end=date_end,
                tag_filters=filter_tag,
            )
            console.print(f"[green]Removed tag from {updated} observations[/green]")
        finally:
            await manager.close()

    asyncio.run(_run())


@app.command()
def summarize(
    project: Optional[str] = typer.Option(None, help="Project path filter"),
    all_projects: bool = typer.Option(False, help="Include all projects"),
    session_id: Optional[str] = typer.Option(None, help="Session ID filter"),
    count: int = typer.Option(20, help="Number of recent observations to summarize"),
    obs_type: Optional[str] = typer.Option(None, help="Observation type filter"),
    tag: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tags to attach to summary"),
    dry_run: bool = typer.Option(False, help="Generate summary without storing"),
    output: str = typer.Option("text", "--format", "-f", help="Output format: text, json"),
):
    """Summarize recent observations into a session summary."""
    async def _run():
        manager = get_memory_manager()
        await manager.initialize()
        try:
            nonlocal project
            if session_id:
                project = None
            elif not project and not all_projects:
                project = os.getcwd()
            if all_projects:
                project = None
            result = await manager.summarize_project(
                project=project,
                session_id=session_id,
                limit=count,
                obs_type=obs_type,
                store=not dry_run,
                tags=tag,
            )
            if not result:
                console.print("[yellow]No observations found to summarize.[/yellow]")
                return
            summary = result.get("summary", "")
            if output == "json":
                payload = {"summary": summary, "metadata": result.get("metadata")}
                obs = result.get("observation")
                if obs:
                    payload["observation"] = obs.model_dump()
                print(json.dumps(payload, indent=2))
                return
            console.print(summary)
            obs = result.get("observation")
            if obs:
                console.print(f"[green]Stored summary:[/green] {obs.id}")
        finally:
            await manager.close()

    asyncio.run(_run())


@app.command()
def get(
    ids: List[str] = typer.Argument(..., help="Observation IDs"),
    output: str = typer.Option("json", "--format", "-f", help="Output format: text, json"),
):
    """Fetch full observation details by ID."""
    async def _run():
        manager = get_memory_manager()
        await manager.initialize()
        try:
            results = await manager.get_observations(ids)
            if output == "json":
                print(json.dumps(results, indent=2))
                return

            for obs in results:
                console.print(f"[bold]{obs['id']}[/bold] {obs['summary']}")
                console.print(obs["content"])
                console.print("")
        finally:
            await manager.close()

    asyncio.run(_run())


@app.command()
def server(
    port: int = typer.Option(8000, help="Port to run the API server"),
):
    """Start the API + web viewer server."""
    from .server import start_server

    console.print(f"[green]Starting ai-mem server at http://localhost:{port}[/green]")
    start_server(port=port)


@app.command(name="sessions")
def list_sessions(
    project: Optional[str] = typer.Option(None, help="Project path filter"),
    all_projects: bool = typer.Option(False, help="Include all projects"),
    active_only: bool = typer.Option(False, help="Only sessions without an end time"),
    goal: Optional[str] = typer.Option(None, help="Filter by goal text"),
    date_start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD or epoch)"),
    date_end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD or epoch)"),
    limit: Optional[int] = typer.Option(None, help="Max sessions to return"),
    output: str = typer.Option("text", "--format", "-f", help="Output format: text, json"),
):
    """List known sessions."""
    async def _run():
        manager = get_memory_manager()
        await manager.initialize()
        try:
            nonlocal project
            if not project and not all_projects:
                project = os.getcwd()
            if all_projects:
                project = None
            sessions = await manager.list_sessions(
                project=project,
                active_only=active_only,
                goal_query=goal,
                date_start=date_start,
                date_end=date_end,
                limit=limit,
            )
            if output == "json":
                print(json.dumps(sessions, indent=2))
                return
            table = Table(title="Sessions")
            table.add_column("ID", style="dim")
            table.add_column("Project", style="green")
            table.add_column("Start", style="cyan")
            table.add_column("End", style="magenta")
            for item in sessions:
                table.add_row(
                    item.get("id", "-"),
                    item.get("project", "-"),
                    str(item.get("start_time", "-")),
                    str(item.get("end_time", "-")),
                )
            console.print(table)
        finally:
            await manager.close()

    asyncio.run(_run())


@app.command(name="session-start")
def session_start(
    project: Optional[str] = typer.Option(None, help="Project path override"),
    goal: Optional[str] = typer.Option(None, help="Session goal"),
    session_id: Optional[str] = typer.Option(None, "--id", "-i", help="Session ID override"),
):
    """Start a new session for a project."""
    async def _run():
        manager = get_memory_manager()
        await manager.initialize()
        try:
            project_name = project or os.getcwd()
            session = await manager.start_session(
                project=project_name,
                goal=goal or "",
                session_id=session_id,
            )
            console.print(f"[green]Session started:[/green] {session.id}")
        finally:
            await manager.close()

    asyncio.run(_run())


@app.command(name="session-end")
def session_end(
    session_id: Optional[str] = typer.Option(None, "--id", "-i", help="Session ID to close"),
    project: Optional[str] = typer.Option(None, help="Project path (used when no ID is provided)"),
):
    """End a session by ID (or close the current one)."""
    async def _run():
        manager = get_memory_manager()
        await manager.initialize()
        try:
            if session_id:
                session = await manager.end_session(session_id)
            else:
                project_name = project or os.getcwd()
                session = await manager.end_latest_session(project_name)
            if not session:
                console.print("[yellow]No matching session found.[/yellow]")
                raise typer.Exit(1)
            console.print(f"[green]Session ended:[/green] {session.id}")
        finally:
            await manager.close()

    asyncio.run(_run())


@app.command()
def proxy(
    host: str = typer.Option("0.0.0.0", help="Proxy host"),
    port: int = typer.Option(8081, help="Proxy port"),
    upstream: Optional[str] = typer.Option(None, help="Upstream OpenAI-compatible base URL"),
    upstream_key: Optional[str] = typer.Option(None, help="Upstream API key"),
    inject: bool = typer.Option(True, "--inject/--no-inject", help="Inject ai-mem context"),
    store: bool = typer.Option(True, "--store/--no-store", help="Store prompt/response pairs"),
    project: Optional[str] = typer.Option(None, help="Default project path"),
    summarize: bool = typer.Option(True, "--summarize/--no-summarize", help="Summarize stored content"),
):
    """Start an OpenAI-compatible proxy that injects context and stores interactions."""
    from .proxy import start_proxy

    cfg = load_config()
    upstream_base = upstream or cfg.llm.base_url or os.environ.get("AI_MEM_PROXY_UPSTREAM_BASE_URL")
    if not upstream_base:
        console.print("[red]Proxy requires --upstream or AI_MEM_PROXY_UPSTREAM_BASE_URL.[/red]")
        raise typer.Exit(1)
    upstream_secret = (
        upstream_key
        or os.environ.get("AI_MEM_PROXY_UPSTREAM_KEY")
        or os.environ.get("AI_MEM_PROXY_UPSTREAM_API_KEY")
        or cfg.llm.api_key
    )
    console.print(f"[green]Starting ai-mem proxy at http://{host}:{port}[/green]")
    start_proxy(
        host=host,
        port=port,
        upstream_base_url=upstream_base,
        upstream_api_key=upstream_secret,
        inject_context=inject,
        store_interactions=store,
        default_project=project,
        summarize=summarize,
    )


@app.command(name="gemini-proxy")
def gemini_proxy(
    host: str = typer.Option("0.0.0.0", help="Proxy host"),
    port: int = typer.Option(8090, help="Proxy port"),
    upstream: Optional[str] = typer.Option(None, help="Gemini API base URL"),
    upstream_key: Optional[str] = typer.Option(None, help="Gemini API key"),
    inject: bool = typer.Option(True, "--inject/--no-inject", help="Inject ai-mem context"),
    store: bool = typer.Option(True, "--store/--no-store", help="Store prompt/response pairs"),
    project: Optional[str] = typer.Option(None, help="Default project path"),
    summarize: bool = typer.Option(True, "--summarize/--no-summarize", help="Summarize stored content"),
):
    """Start a Gemini API proxy that injects context and stores interactions."""
    from .gemini_proxy import start_proxy as start_gemini_proxy

    base_url = (
        upstream
        or os.environ.get("AI_MEM_GEMINI_UPSTREAM_BASE_URL")
        or "https://generativelanguage.googleapis.com"
    )
    api_key = (
        upstream_key
        or os.environ.get("AI_MEM_GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        console.print("[yellow]Gemini proxy running without API key (pass --upstream-key or AI_MEM_GEMINI_API_KEY).[/yellow]")
    console.print(f"[green]Starting ai-mem Gemini proxy at http://{host}:{port}[/green]")
    start_gemini_proxy(
        host=host,
        port=port,
        upstream_base_url=base_url,
        upstream_api_key=api_key,
        inject_context=inject,
        store_interactions=store,
        default_project=project,
        summarize=summarize,
    )


@app.command()
def anthropic_proxy(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8095, help="Proxy port"),
    upstream: Optional[str] = typer.Option(None, help="Anthropic API base URL"),
    upstream_key: Optional[str] = typer.Option(None, help="Anthropic API key"),
    anthropic_version: Optional[str] = typer.Option(None, help="Anthropic API version header"),
    inject: bool = typer.Option(True, "--inject/--no-inject", help="Inject memory context"),
    store: bool = typer.Option(True, "--store/--no-store", help="Store interactions"),
    project: Optional[str] = typer.Option(None, help="Project path override"),
    summarize: bool = typer.Option(True, "--summarize/--no-summarize", help="Summarize stored content"),
):
    """Start an Anthropic API proxy that injects context and stores interactions."""
    from .anthropic_proxy import start_proxy as start_anthropic_proxy

    base_url = (
        upstream
        or os.environ.get("AI_MEM_ANTHROPIC_UPSTREAM_BASE_URL")
        or "https://api.anthropic.com"
    )
    api_key = (
        upstream_key
        or os.environ.get("AI_MEM_ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not api_key:
        console.print(
            "[yellow]Anthropic proxy running without API key (pass --upstream-key or AI_MEM_ANTHROPIC_API_KEY).[/yellow]"
        )
    version = (
        anthropic_version
        or os.environ.get("AI_MEM_ANTHROPIC_VERSION")
        or "2023-06-01"
    )
    console.print(f"[green]Starting ai-mem Anthropic proxy at http://{host}:{port}[/green]")
    start_anthropic_proxy(
        host=host,
        port=port,
        upstream_base_url=base_url,
        upstream_api_key=api_key,
        inject_context=inject,
        store_interactions=store,
        default_project=project,
        summarize=summarize,
        anthropic_version=version,
    )


@app.command(name="azure-proxy")
def azure_proxy(
    host: str = typer.Option("0.0.0.0", help="Proxy host"),
    port: int = typer.Option(8092, help="Proxy port"),
    upstream: Optional[str] = typer.Option(None, help="Azure OpenAI base URL"),
    upstream_key: Optional[str] = typer.Option(None, help="Azure OpenAI API key"),
    deployment: Optional[str] = typer.Option(None, help="Azure OpenAI deployment name"),
    api_version: Optional[str] = typer.Option(None, help="Azure OpenAI API version"),
    inject: bool = typer.Option(True, "--inject/--no-inject", help="Inject ai-mem context"),
    store: bool = typer.Option(True, "--store/--no-store", help="Store prompt/response pairs"),
    project: Optional[str] = typer.Option(None, help="Default project path"),
    summarize: bool = typer.Option(True, "--summarize/--no-summarize", help="Summarize stored content"),
):
    """Start an Azure OpenAI proxy that injects context and stores interactions."""
    from .azure_proxy import start_proxy as start_azure_proxy

    base_url = (
        upstream
        or os.environ.get("AI_MEM_AZURE_UPSTREAM_BASE_URL")
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
    )
    if not base_url:
        console.print("[red]Azure proxy requires --upstream or AZURE_OPENAI_ENDPOINT.[/red]")
        raise typer.Exit(1)
    api_key = (
        upstream_key
        or os.environ.get("AI_MEM_AZURE_API_KEY")
        or os.environ.get("AZURE_OPENAI_API_KEY")
    )
    if not api_key:
        console.print("[yellow]Azure proxy running without API key (pass --upstream-key or AZURE_OPENAI_API_KEY).[/yellow]")
    deployment_name = (
        deployment
        or os.environ.get("AI_MEM_AZURE_DEPLOYMENT")
        or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    )
    if not deployment_name:
        console.print("[red]Azure proxy requires a deployment name (use --deployment or AZURE_OPENAI_DEPLOYMENT).[/red]")
        raise typer.Exit(1)
    version = (
        api_version
        or os.environ.get("AI_MEM_AZURE_API_VERSION")
        or os.environ.get("AZURE_OPENAI_API_VERSION")
        or "2024-02-01"
    )
    console.print(f"[green]Starting ai-mem Azure proxy at http://{host}:{port}[/green]")
    start_azure_proxy(
        host=host,
        port=port,
        upstream_base_url=base_url,
        upstream_api_key=api_key,
        deployment_name=deployment_name,
        api_version=version,
        inject_context=inject,
        store_interactions=store,
        default_project=project,
        summarize=summarize,
    )


@app.command(name="bedrock-proxy")
def bedrock_proxy(
    host: str = typer.Option("0.0.0.0", help="Proxy host"),
    port: int = typer.Option(8094, help="Proxy port"),
    model: Optional[str] = typer.Option(None, help="Bedrock model id"),
    region: Optional[str] = typer.Option(None, help="AWS region override"),
    endpoint: Optional[str] = typer.Option(None, help="Bedrock runtime endpoint override"),
    profile: Optional[str] = typer.Option(None, help="AWS profile name"),
    anthropic_version: Optional[str] = typer.Option(None, help="Anthropic Bedrock version"),
    max_tokens: Optional[int] = typer.Option(None, help="Default max tokens"),
    inject: bool = typer.Option(True, "--inject/--no-inject", help="Inject ai-mem context"),
    store: bool = typer.Option(True, "--store/--no-store", help="Store prompt/response pairs"),
    project: Optional[str] = typer.Option(None, help="Default project path"),
    summarize: bool = typer.Option(True, "--summarize/--no-summarize", help="Summarize stored content"),
):
    """Start a Bedrock proxy that injects context and stores interactions."""
    from .bedrock_proxy import (
        DEFAULT_ANTHROPIC_VERSION,
        DEFAULT_MAX_TOKENS,
        start_proxy as start_bedrock_proxy,
    )

    model_id = model or os.environ.get("AI_MEM_BEDROCK_MODEL")
    if not model_id:
        console.print("[red]Bedrock proxy requires a model id (use --model or AI_MEM_BEDROCK_MODEL).[/red]")
        raise typer.Exit(1)
    region_name = (
        region
        or os.environ.get("AI_MEM_BEDROCK_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
    )
    profile_name = profile or os.environ.get("AI_MEM_BEDROCK_PROFILE") or os.environ.get("AWS_PROFILE")
    endpoint_url = endpoint or os.environ.get("AI_MEM_BEDROCK_ENDPOINT")
    version = (
        anthropic_version
        or os.environ.get("AI_MEM_BEDROCK_ANTHROPIC_VERSION")
        or DEFAULT_ANTHROPIC_VERSION
    )
    max_tokens_value: int
    if max_tokens is not None:
        max_tokens_value = max_tokens
    else:
        env_max = os.environ.get("AI_MEM_BEDROCK_MAX_TOKENS")
        if env_max:
            try:
                max_tokens_value = int(env_max)
            except ValueError:
                console.print("[yellow]Invalid AI_MEM_BEDROCK_MAX_TOKENS; using default.[/yellow]")
                max_tokens_value = DEFAULT_MAX_TOKENS
        else:
            max_tokens_value = DEFAULT_MAX_TOKENS

    console.print(f"[green]Starting ai-mem Bedrock proxy at http://{host}:{port}[/green]")
    start_bedrock_proxy(
        host=host,
        port=port,
        model_id=model_id,
        region=region_name,
        endpoint_url=endpoint_url,
        profile=profile_name,
        anthropic_version=version,
        max_tokens=max_tokens_value,
        inject_context=inject,
        store_interactions=store,
        default_project=project,
        summarize=summarize,
    )


@app.command()
def mcp():
    """Start MCP stdio server with search tools."""
    from .mcp_server import run_stdio

    run_stdio()


@app.command("mcp-config")
def mcp_config(
    bin_path: str = typer.Option("ai-mem", help="ai-mem binary path"),
    name: str = typer.Option("ai-mem", help="MCP server name"),
    project: Optional[str] = typer.Option(None, help="Default AI_MEM_PROJECT"),
    full: bool = typer.Option(False, help="Wrap in mcpServers object"),
    output: Optional[str] = typer.Option(None, help="Write JSON to file"),
):
    """Generate an MCP config snippet for external clients."""
    entry: Dict[str, object] = {"command": bin_path, "args": ["mcp"]}
    if project:
        entry["env"] = {"AI_MEM_PROJECT": project}
    payload: Dict[str, object]
    if full:
        payload = {"mcpServers": {name: entry}}
    else:
        payload = entry

    text = json.dumps(payload, indent=2)
    if output:
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        console.print(f"[green]Wrote MCP config to {output}[/green]")
        return
    print(text)


@app.command("hook-config")
def hook_config(
    bin_path: str = typer.Option("ai-mem", help="ai-mem binary path"),
    project: Optional[str] = typer.Option(None, help="Default project path"),
    session_tracking: bool = typer.Option(True, help="Enable session tracking for start/end"),
    summary_on_end: bool = typer.Option(False, help="Summarize on session_end"),
    summary_count: int = typer.Option(20, help="Summary count"),
    summary_obs_type: Optional[str] = typer.Option(None, help="Summary observation type filter"),
    context_total: Optional[int] = typer.Option(None, help="Context index count"),
    context_full: Optional[int] = typer.Option(None, help="Context full count"),
    context_full_field: Optional[str] = typer.Option(None, help="Context full field (content or summary)"),
    context_tags: Optional[str] = typer.Option(None, help="Context tags (comma-separated)"),
    no_wrap: bool = typer.Option(False, help="Disable <ai-mem-context> wrapper"),
    output: Optional[str] = typer.Option(None, help="Write JSON to file"),
):
    """Generate hook command config for clients that support shell hooks."""
    def build_args(event: str) -> List[str]:
        args = ["hook", event]
        if project:
            args += ["--project", project]
        if session_tracking and event in {"session_start", "session_end"}:
            args.append("--session-tracking")
        if event == "session_start":
            if context_total is not None:
                args += ["--total", str(context_total)]
            if context_full is not None:
                args += ["--full", str(context_full)]
            if context_full_field:
                args += ["--full-field", context_full_field]
            if context_tags:
                for tag in [item.strip() for item in context_tags.split(",") if item.strip()]:
                    args += ["--context-tag", tag]
            if no_wrap:
                args.append("--no-wrap")
        if event == "session_end":
            if summary_on_end:
                args.append("--summary-on-end")
                args += ["--summary-count", str(summary_count)]
                if summary_obs_type:
                    args += ["--summary-obs-type", summary_obs_type]
        return args

    payload = {
        "session_start": {"command": bin_path, "args": build_args("session_start")},
        "user_prompt": {"command": bin_path, "args": build_args("user_prompt")},
        "assistant_response": {"command": bin_path, "args": build_args("assistant_response")},
        "tool_output": {"command": bin_path, "args": build_args("tool_output")},
        "session_end": {"command": bin_path, "args": build_args("session_end")},
    }

    text = json.dumps(payload, indent=2)
    if output:
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        console.print(f"[green]Wrote hook config to {output}[/green]")
        return
    print(text)


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
            import shlex

            console.print(f"[green]Running and watching:[/green] {command}")
            process = subprocess.Popen(
                shlex.split(command),
                shell=False,
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
    path: str = typer.Argument(..., help="Path to write exported data"),
    project: Optional[str] = typer.Option(None, help="Project path filter"),
    session_id: Optional[str] = typer.Option(None, help="Session ID filter"),
    obs_type: Optional[str] = typer.Option(None, help="Observation type filter"),
    date_start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD or epoch)"),
    date_end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD or epoch)"),
    since: Optional[str] = typer.Option(None, help="Relative start window (e.g. 24h, 7d)"),
    tag: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tag filters (any match)"),
    limit: Optional[int] = typer.Option(None, help="Limit number of observations"),
    output: Optional[str] = typer.Option(None, "--format", "-f", help="Output format: json, jsonl, csv"),
):
    """Export observations to a file."""
    manager = get_memory_manager()
    fmt = _infer_format(path, output, "json")
    if since and not date_start:
        date_start = since
    data = manager.export_observations(
        project=project,
        session_id=session_id,
        obs_type=obs_type,
        date_start=date_start,
        date_end=date_end,
        tag_filters=tag,
        limit=limit,
    )
    if fmt == "json":
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
    elif fmt in {"jsonl", "ndjson"}:
        with open(path, "w", encoding="utf-8") as handle:
            for item in data:
                handle.write(json.dumps(item, ensure_ascii=True) + "\n")
    elif fmt == "csv":
        fields = [
            "id",
            "session_id",
            "project",
            "type",
            "title",
            "summary",
            "content",
            "created_at",
            "importance_score",
            "tags",
            "metadata",
        ]
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in data:
                writer.writerow(
                    {
                        "id": item.get("id"),
                        "session_id": item.get("session_id"),
                        "project": item.get("project"),
                        "type": item.get("type"),
                        "title": item.get("title"),
                        "summary": item.get("summary"),
                        "content": item.get("content"),
                        "created_at": item.get("created_at"),
                        "importance_score": item.get("importance_score"),
                        "tags": json.dumps(item.get("tags") or [], ensure_ascii=True),
                        "metadata": json.dumps(item.get("metadata") or {}, ensure_ascii=True),
                    }
                )
    else:
        console.print(f"[red]Unsupported format: {fmt}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Exported {len(data)} observations to {path}[/green]")


@app.command("import")
def import_memories(
    path: str = typer.Argument(..., help="Path to import file"),
    project: Optional[str] = typer.Option(None, help="Override project for imported items"),
    dedupe: bool = typer.Option(True, help="Skip duplicates by content hash"),
    input_format: Optional[str] = typer.Option(None, "--format", "-f", help="Input format: json, jsonl, csv"),
):
    """Import observations from an export file."""
    manager = get_memory_manager()
    fmt = _infer_format(path, input_format, "json")
    items, metadata = _read_export_file(path, fmt)
    imported = _import_items(manager, items, project, dedupe)
    if metadata:
        meta_parts = []
        since = metadata.get("since")
        if since:
            meta_parts.append(f"since={since}")
        meta_parts.append(f"snapshot_id={metadata.get('snapshot_id')}")
        console.print(f"[green]Snapshot metadata:{' '.join(meta_parts)}[/green]")
    console.print(f"[green]Imported {imported} observations[/green]")


@snapshot_app.command("export")
def snapshot_export(
    path: str = typer.Argument(..., help="Path to write snapshot data"),
    project: Optional[str] = typer.Option(None, help="Project path filter"),
    session_id: Optional[str] = typer.Option(None, help="Session ID filter"),
    obs_type: Optional[str] = typer.Option(None, help="Observation type filter"),
    since: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD or relative)"),
    tag: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tag filters (any match)"),
    limit: Optional[int] = typer.Option(None, help="Limit number of observations"),
    output: Optional[str] = typer.Option(None, "--format", "-f", help="Output format: json, jsonl"),
    note: Optional[str] = typer.Option(None, help="Optional note to include in snapshot metadata"),
):
    """Export a snapshot (JSONL) for incremental syncing."""
    manager = get_memory_manager()
    fmt = _infer_format(path, output, "jsonl")
    data = manager.export_observations(
        project=project,
        session_id=session_id,
        obs_type=obs_type,
        date_start=since,
        date_end=None,
        tag_filters=tag,
        limit=limit,
    )
    data = sorted(data, key=lambda item: float(item.get("created_at") or 0))
    now = time.time()
    meta = {
        "snapshot_id": str(uuid4()),
        "created_at": now,
        "project": project,
        "session_id": session_id,
        "obs_type": obs_type,
        "since": since,
        "limit": limit,
        "note": note,
        "last_observation": data[-1]["created_at"] if data else None,
        "count": len(data),
    }
    if fmt == "json":
        payload = {"snapshot": meta, "observations": data}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    elif fmt in {"jsonl", "ndjson"}:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"snapshot": meta}, ensure_ascii=True))
            handle.write("\n")
            for item in data:
                handle.write(json.dumps(item, ensure_ascii=True))
                handle.write("\n")
    else:
        console.print(f"[red]Unsupported format: {fmt}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Snapshot exported with {len(data)} observations to {path}[/green]")
    console.print(f"[green]Snapshot id: {meta['snapshot_id']} created at {meta['created_at']}[/green]")


@snapshot_app.command("import")
def snapshot_import(
    path: str = typer.Argument(..., help="Path to snapshot file"),
    project: Optional[str] = typer.Option(None, help="Override project for imported items"),
    dedupe: bool = typer.Option(True, help="Skip duplicates by content hash"),
    input_format: Optional[str] = typer.Option(None, "--format", "-f", help="Input format: json, jsonl, ndjson"),
):
    """Import a snapshot file (JSONL or JSON) and merge into the local store."""
    manager = get_memory_manager()
    fmt = _infer_format(path, input_format, "jsonl")
    items, metadata = _read_export_file(path, fmt)
    imported = _import_items(manager, items, project, dedupe)
    if metadata:
        meta_parts = []
        snapshot_id = metadata.get("snapshot_id")
        if snapshot_id:
            meta_parts.append(f"id={snapshot_id}")
        if metadata.get("note"):
            meta_parts.append(f"note={metadata['note']}")
    console.print(f"[green]Snapshot metadata:{' '.join(meta_parts)}[/green]")
    console.print(f"[green]Imported {imported} observations from snapshot[/green]")


@snapshot_app.command("merge")
def snapshot_merge(
    output: str = typer.Argument(..., help="Path to write merged snapshot"),
    inputs: List[str] = typer.Argument(..., help="Snapshot files to merge"),
    dedupe: bool = typer.Option(True, help="Skip duplicates by content hash"),
    output_format: Optional[str] = typer.Option(None, "--format", "-f", help="Output format: json, jsonl, csv"),
):
    """Merge multiple snapshot exports into one file (dedupe & reorder)."""
    if not inputs:
        console.print("[red]Provide at least one input snapshot file.[/red]")
        raise typer.Exit(1)
    all_items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for path in inputs:
        fmt = _infer_format(path, None, "jsonl")
        items, _ = _read_export_file(path, fmt)
        for item in items:
            if not isinstance(item, dict):
                continue
            key = _snapshot_key(item)
            if dedupe and key in seen:
                continue
            seen.add(key)
            all_items.append(item)
    all_items.sort(key=lambda obs: float(obs.get("created_at") or 0))
    fmt_out = _infer_format(output, output_format, "jsonl")
    meta = {
        "snapshot_id": str(uuid4()),
        "created_at": time.time(),
        "sources": inputs,
        "count": len(all_items),
        "dedupe": dedupe,
    }
    _write_snapshot_file(output, fmt_out, meta, all_items)
    console.print(f"[green]Merged {len(all_items)} observations into {output} ({fmt_out})[/green]")


@app.command()
def delete(
    obs_id: str = typer.Argument(..., help="Observation ID to delete"),
    force: bool = typer.Option(False, help="Skip confirmation prompt"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without actually deleting"),
):
    """Delete a single observation."""
    async def _run():
        manager = get_memory_manager()
        await manager.initialize()
        try:
            results = await manager.get_observations([obs_id])
            if not results:
                console.print(f"[yellow]Observation not found: {obs_id}[/yellow]")
                raise typer.Exit(1)

            if dry_run:
                obs = results[0]
                console.print("[yellow]DRY RUN - Would delete:[/yellow]")
                console.print(f"  ID: {obs.get('id')}")
                console.print(f"  Type: {obs.get('type')}")
                console.print(f"  Title: {obs.get('title', 'N/A')}")
                console.print(f"  Project: {obs.get('project')}")
                console.print(f"  Created: {obs.get('created_at')}")
                return

            nonlocal force
            if not force:
                confirm = typer.confirm(f"Delete observation {obs_id}?")
                if not confirm:
                    raise typer.Exit(1)

            if await manager.delete_observation(obs_id):
                console.print(f"[green]Deleted observation {obs_id}[/green]")
            else:
                console.print(f"[yellow]Observation not found: {obs_id}[/yellow]")
        finally:
            await manager.close()

    asyncio.run(_run())


@app.command(name="update-tags")
def update_tags(
    obs_id: str = typer.Argument(..., help="Observation ID to update"),
    tag: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tags to set (replaces existing)"),
    clear: bool = typer.Option(False, help="Clear all tags"),
):
    """Replace tags on a single observation."""
    if clear:
        tags = []
    else:
        tags = []
        for item in tag or []:
            tags.extend([part.strip() for part in str(item).split(",") if part.strip()])
        if not tags:
            console.print("[yellow]No tags provided. Use --tag or --clear.[/yellow]")
            raise typer.Exit(1)
    manager = get_memory_manager()
    if manager.update_observation_tags(obs_id, tags):
        console.print(f"[green]Updated tags for {obs_id}[/green]")
    else:
        console.print(f"[yellow]Observation not found: {obs_id}[/yellow]")


@app.command()
def delete_project(
    project: str = typer.Argument(..., help="Project path to delete"),
    force: bool = typer.Option(False, help="Skip confirmation prompt"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without actually deleting"),
):
    """Delete all observations for a project."""
    async def _run():
        manager = get_memory_manager()
        await manager.initialize()
        try:
            # Count observations that would be affected
            stats = await manager.db.get_stats(project=project)
            count = stats.get("total", 0)

            if dry_run:
                console.print("[yellow]DRY RUN - Would delete:[/yellow]")
                console.print(f"  Project: {project}")
                console.print(f"  Observations: {count}")
                return

            if count == 0:
                console.print(f"[yellow]No observations found for project: {project}[/yellow]")
                return

            nonlocal force
            if not force:
                confirm = typer.confirm(f"Delete {count} observations for {project}?")
                if not confirm:
                    raise typer.Exit(1)

            deleted = await manager.delete_project(project)
            console.print(f"[green]Deleted {deleted} observations for {project}[/green]")
        finally:
            await manager.close()

    asyncio.run(_run())


@app.command()
def cleanup_events(
    max_age_days: int = typer.Option(30, help="Delete event IDs older than this many days"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be cleaned up without actually deleting"),
):
    """Clean up old event idempotency records.

    The event_idempotency table tracks processed events to prevent duplicates.
    This command removes records older than max_age_days to prevent table growth.
    This is automatically run on server startup, but can be run manually.
    """
    async def _run():
        manager = get_memory_manager()
        await manager.initialize()
        try:
            if dry_run:
                # Count records that would be deleted
                import time
                cutoff = time.time() - (max_age_days * 86400)
                async with manager.db.conn.execute(
                    "SELECT COUNT(*) FROM event_idempotency WHERE processed_at < ?",
                    (cutoff,),
                ) as cursor:
                    row = await cursor.fetchone()
                    count = row[0] if row else 0
                console.print("[yellow]DRY RUN - Would clean up:[/yellow]")
                console.print(f"  Records older than: {max_age_days} days")
                console.print(f"  Records to delete: {count}")
                return

            deleted = await manager.db.cleanup_old_event_ids(max_age_days=max_age_days)
            console.print(f"[green]Cleaned up {deleted} old event idempotency records (older than {max_age_days} days)[/green]")
        finally:
            await manager.close()

    asyncio.run(_run())


if __name__ == "__main__":
    app()

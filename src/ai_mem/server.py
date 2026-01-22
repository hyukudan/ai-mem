import asyncio
import csv
import io
import json
import os
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any

import uvicorn

from .context import build_context, estimate_tokens
from . import __version__
from .memory import MemoryManager
from .adapters import get_adapter
from .events import ToolEvent, UserPromptEvent, SessionEvent
from .exceptions import (
    AiMemError,
    ValidationError,
    InvalidUUIDError,
    ResourceNotFoundError,
    ObservationNotFoundError,
    SessionNotFoundError,
    ProjectNotFoundError,
    InvalidTokenError,
    UnauthorizedError,
    DatabaseError,
    AdapterError,
    UnknownHostError,
    PayloadParseError,
)

import logging

logger = logging.getLogger("ai_mem.server")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if not _api_token:
        logger.warning("SECURITY WARNING: AI_MEM_API_TOKEN is not set. The API is accessible without authentication.")

    manager = get_manager()
    await manager.initialize()

    # Cleanup old event IDs to prevent table growth
    try:
        max_age_days = int(os.environ.get("AI_MEM_EVENT_ID_MAX_AGE_DAYS", "30"))
        deleted = await manager.db.cleanup_old_event_ids(max_age_days=max_age_days)
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old event idempotency records (older than {max_age_days} days)")
    except Exception as e:
        logger.warning(f"Failed to cleanup old event IDs: {e}")

    yield
    # Shutdown
    if _manager:
        await _manager.close()


app = FastAPI(title="ai-mem Server", lifespan=lifespan)


# =============================================================================
# Web UI Routes
# =============================================================================

from pathlib import Path

UI_DIR = Path(__file__).parent / "ui"


@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to dashboard."""
    return """
    <html>
        <head><meta http-equiv="refresh" content="0; url=/ui" /></head>
        <body>Redirecting to <a href="/ui">dashboard</a>...</body>
    </html>
    """


# Cache for dashboard HTML (read once, serve many)
_dashboard_cache: Optional[str] = None
_dashboard_cache_time: float = 0


@app.get("/ui", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard UI with caching."""
    global _dashboard_cache, _dashboard_cache_time
    import time

    index_path = UI_DIR / "index.html"

    # Check cache validity (refresh every hour in production)
    current_time = time.time()
    if _dashboard_cache is None or (current_time - _dashboard_cache_time) > CACHE_MAX_AGE:
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Dashboard UI not found")
        _dashboard_cache = index_path.read_text()
        _dashboard_cache_time = current_time

    return _dashboard_cache


@app.get("/ui/{path:path}", response_class=HTMLResponse)
async def serve_ui_files(path: str):
    """Serve UI static files with path traversal protection."""
    # Security: validate path BEFORE constructing file path
    # Reject any path with traversal attempts
    if ".." in path or path.startswith("/") or "\\" in path:
        raise HTTPException(status_code=403, detail="Access denied: invalid path")

    file_path = UI_DIR / path
    resolved_path = file_path.resolve()

    # Security: ensure resolved path is within UI_DIR
    try:
        resolved_path.relative_to(UI_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: path outside UI directory")

    if not resolved_path.exists() or not resolved_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return resolved_path.read_text()


# =============================================================================
# Citation URLs - Observation Viewer
# =============================================================================

@app.get("/view/{observation_id}", response_class=HTMLResponse)
async def view_observation_html(observation_id: str, request: Request):
    """View an observation as a rendered HTML page.

    This provides a human-readable view of an observation with all its details.
    Accessible via citation URLs like /view/{id}.
    """
    _validate_uuid(observation_id)
    try:
        results = await get_manager().get_observations([observation_id])
        if not results:
            raise ObservationNotFoundError(observation_id)
        obs = results[0]

        # Format metadata
        from datetime import datetime
        from .models import CONCEPT_ICONS, TYPE_ICONS

        created = datetime.fromtimestamp(obs.get("created_at", 0)).strftime("%Y-%m-%d %H:%M:%S")
        obs_type = obs.get("type", "note")
        concept = obs.get("concept")
        tags = obs.get("tags") or []
        project = obs.get("project", "-")
        session_id = obs.get("session_id", "-")

        # Get icons
        type_icon = TYPE_ICONS.get(obs_type, "") if hasattr(__import__('ai_mem.models', fromlist=['TYPE_ICONS']), 'TYPE_ICONS') else ""
        concept_icon = CONCEPT_ICONS.get(concept, "") if concept else ""
        icon = concept_icon or type_icon

        # Content
        summary = obs.get("summary") or ""
        content = obs.get("content") or ""
        metadata = obs.get("metadata") or {}

        # Escape HTML
        import html
        summary_html = html.escape(summary)
        content_html = html.escape(content).replace("\n", "<br>")
        metadata_html = html.escape(json.dumps(metadata, indent=2))

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Observation {observation_id[:8]} | ai-mem</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e4e4e7;
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(90deg, #4f46e5, #7c3aed);
            padding: 1.5rem 2rem;
        }}
        .header h1 {{
            font-size: 1.5rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .header .id {{
            font-family: monospace;
            font-size: 0.9rem;
            opacity: 0.8;
            margin-top: 0.5rem;
        }}
        .meta {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            padding: 1.5rem 2rem;
            background: rgba(0,0,0,0.2);
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .meta-item {{
            display: flex;
            flex-direction: column;
        }}
        .meta-item label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #a1a1aa;
            margin-bottom: 0.25rem;
        }}
        .meta-item span {{
            font-weight: 500;
        }}
        .tags {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}
        .tag {{
            background: rgba(79, 70, 229, 0.3);
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
        }}
        .section {{
            padding: 1.5rem 2rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .section:last-child {{
            border-bottom: none;
        }}
        .section h2 {{
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #a1a1aa;
            margin-bottom: 1rem;
        }}
        .section p {{
            line-height: 1.6;
        }}
        .content {{
            background: rgba(0,0,0,0.2);
            padding: 1rem;
            border-radius: 8px;
            font-family: monospace;
            font-size: 0.9rem;
            white-space: pre-wrap;
            overflow-x: auto;
        }}
        .footer {{
            padding: 1rem 2rem;
            text-align: center;
            font-size: 0.8rem;
            color: #71717a;
        }}
        .footer a {{
            color: #818cf8;
            text-decoration: none;
        }}
        .footer a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{icon} Observation</h1>
            <div class="id">{observation_id}</div>
        </div>

        <div class="meta">
            <div class="meta-item">
                <label>Type</label>
                <span>{obs_type}</span>
            </div>
            <div class="meta-item">
                <label>Concept</label>
                <span>{concept or '-'}</span>
            </div>
            <div class="meta-item">
                <label>Created</label>
                <span>{created}</span>
            </div>
            <div class="meta-item">
                <label>Project</label>
                <span>{project}</span>
            </div>
        </div>

        {"<div class='section'><h2>Tags</h2><div class='tags'>" + "".join(f"<span class='tag'>{html.escape(t)}</span>" for t in tags) + "</div></div>" if tags else ""}

        <div class="section">
            <h2>Summary</h2>
            <p>{summary_html}</p>
        </div>

        {"<div class='section'><h2>Content</h2><div class='content'>" + content_html + "</div></div>" if content else ""}

        <div class="footer">
            <a href="/ui">← Back to Dashboard</a> |
            <a href="/api/observation/{observation_id}">View JSON</a>
        </div>
    </div>
</body>
</html>"""
    except (HTTPException, AiMemError):
        raise
    except Exception as exc:
        logger.error(f"Error viewing observation: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/obs/{short_id}")
async def redirect_short_observation(short_id: str, request: Request):
    """Redirect from short observation ID to full view.

    Short IDs are the first 8 characters of the observation UUID.
    This enables compact citation URLs like /obs/abc12345.
    """
    if len(short_id) < 4 or len(short_id) > 36:
        raise HTTPException(status_code=400, detail="Invalid short ID")

    # Search for observation by prefix
    try:
        observations = await get_manager().db.list_observations(limit=100)
        for obs in observations:
            if obs.id.startswith(short_id):
                from fastapi.responses import RedirectResponse
                return RedirectResponse(url=f"/view/{obs.id}", status_code=302)

        raise HTTPException(status_code=404, detail=f"No observation found with ID prefix: {short_id}")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error finding observation: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(InvalidUUIDError)
async def invalid_uuid_handler(request: Request, exc: InvalidUUIDError):
    return JSONResponse(
        status_code=400,
        content={"detail": exc.message, "field": exc.field, "value": exc.value},
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": exc.message, "field": exc.field},
    )


@app.exception_handler(ResourceNotFoundError)
async def not_found_handler(request: Request, exc: ResourceNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"detail": exc.message, "type": exc.resource_type, "id": exc.resource_id},
    )


@app.exception_handler(UnauthorizedError)
async def unauthorized_handler(request: Request, exc: UnauthorizedError):
    return JSONResponse(
        status_code=401,
        content={"detail": exc.message},
    )


@app.exception_handler(InvalidTokenError)
async def invalid_token_handler(request: Request, exc: InvalidTokenError):
    return JSONResponse(
        status_code=401,
        content={"detail": exc.message},
    )


@app.exception_handler(AdapterError)
async def adapter_error_handler(request: Request, exc: AdapterError):
    return JSONResponse(
        status_code=400,
        content={"detail": exc.message, "details": exc.details},
    )


@app.exception_handler(DatabaseError)
async def database_error_handler(request: Request, exc: DatabaseError):
    logger.error(f"Database error: {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Database error occurred"},
    )


@app.exception_handler(AiMemError)
async def ai_mem_error_handler(request: Request, exc: AiMemError):
    logger.error(f"Application error: {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": exc.message},
    )


# CORS Configuration - secure by default
# Set AI_MEM_ALLOWED_ORIGINS="*" to allow all origins (NOT recommended for production)
# Example: AI_MEM_ALLOWED_ORIGINS="http://localhost:3000,https://myapp.com"
_cors_env = os.environ.get("AI_MEM_ALLOWED_ORIGINS", "")
if _cors_env:
    _allowed_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
else:
    # Secure default: only localhost variations
    _allowed_origins = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    
    # 🔐 Security: Warn if wildcard is in origins
    if "*" in _allowed_origins:
        logger.warning(
            "SECURITY WARNING: CORS allow_origins contains '*' - "
            "this allows any website to access the API. "
            "Set AI_MEM_CORS_ORIGINS for production."
        )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-AI-Mem-Token"],
    max_age=3600,  # Cache preflight for 1 hour
)

# 🔐 Rate limiting for DoS protection
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    
    _rate_limit_default = os.environ.get("AI_MEM_RATE_LIMIT", "100 per minute")
    
    limiter = Limiter(key_func=get_remote_address, default_limits=[_rate_limit_default])
    app.state.limiter = limiter
    
    @app.exception_handler(RateLimitExceeded)
    async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "message": "Too many requests. Please try again later.",
                "detail": str(exc.detail) if exc.detail else None,
            }
        )
    
    logger.info(f"🔐 Rate limiting enabled: {_rate_limit_default}")
except ImportError:
    logger.warning("slowapi not installed. Rate limiting disabled. Run: pip install slowapi")
    limiter = None

_manager: Optional[MemoryManager] = None

# 🔐 Get API token from secure storage (keyring) or environment
from .secrets import SecretManager
_api_token = SecretManager.get_api_token()

# Constants
MAX_LIMIT = 1000
DEFAULT_LIMIT = 50
CACHE_MAX_AGE = 3600  # 1 hour for static content


def get_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager


def _check_token(request: Request) -> None:
    """Validate API token from headers only (not query params for security).

    Tokens in query parameters are logged in server access logs, which is a security risk.
    Use Authorization header or X-AI-Mem-Token header instead.
    """
    if not _api_token:
        return
    auth = request.headers.get("authorization") or ""
    token = ""
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
    if not token:
        token = request.headers.get("x-ai-mem-token", "")
    if token != _api_token:
        raise InvalidTokenError()


def _parse_list_param(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


class MemoryInput(BaseModel):
    content: str
    obs_type: str = "note"
    project: Optional[str] = None
    session_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    title: Optional[str] = None
    summarize: bool = True


class EventIngestRequest(BaseModel):
    """Request to ingest a raw event from any LLM host."""
    host: str = Field(default="generic", description="Host identifier: claude-code, gemini, cursor, etc.", min_length=1, max_length=50)
    payload: Dict[str, Any] = Field(..., description="Raw event payload from the host")
    session_id: Optional[str] = Field(default=None, description="Session ID override")
    project: Optional[str] = Field(default=None, description="Project path override")
    tags: List[str] = Field(default_factory=list, description="Additional tags")
    summarize: bool = Field(default=True, description="Enable summarization")
    
    class Config:
        extra = "forbid"  # 🔐 Security: Reject unknown fields
    
    @field_validator('host')
    @classmethod
    def validate_host_whitelist(cls, v: str) -> str:
        """🔐 Validate host is in allowed list."""
        ALLOWED_HOSTS = {"claude", "gemini", "vscode", "cursor", "generic", "claude-code", "claude-desktop", "anthropic"}
        if v not in ALLOWED_HOSTS:
            raise ValueError(f"Host '{v}' not allowed. Valid hosts: {', '.join(ALLOWED_HOSTS)}")
        return v


class ObservationIds(BaseModel):
    ids: List[str]


class BatchMemoryInput(BaseModel):
    """Input for batch add memories."""
    observations: List[MemoryInput] = Field(..., description="List of observations to add")


class BatchDeleteInput(BaseModel):
    """Input for batch delete observations."""
    ids: List[str] = Field(..., description="List of observation IDs to delete")


class ObservationUpdate(BaseModel):
    tags: Optional[List[str]] = None


class ProjectDelete(BaseModel):
    project: str


class ImportPayload(BaseModel):
    items: List[Dict[str, Any]]
    project: Optional[str] = None


class SummarizeRequest(BaseModel):
    project: Optional[str] = None
    session_id: Optional[str] = None
    count: int = 20
    obs_type: Optional[str] = None
    store: bool = True
    tags: List[str] = Field(default_factory=list)


class SessionStartRequest(BaseModel):
    project: Optional[str] = None
    goal: Optional[str] = None
    session_id: Optional[str] = None


def _validate_uuid(uid: str) -> None:
    try:
        uuid.UUID(uid)
    except ValueError:
        raise InvalidUUIDError(uid)


class SessionEndRequest(BaseModel):
    session_id: Optional[str] = None
    project: Optional[str] = None


class TagRenameRequest(BaseModel):
    old_tag: str
    new_tag: str
    project: Optional[str] = None
    session_id: Optional[str] = None
    obs_type: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    tags: Optional[str] = None


class TagDeleteRequest(BaseModel):
    tag: str
    project: Optional[str] = None
    session_id: Optional[str] = None
    obs_type: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    tags: Optional[str] = None


class TagAddRequest(BaseModel):
    tag: str
    project: Optional[str] = None
    session_id: Optional[str] = None
    obs_type: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    tags: Optional[str] = None


class ContextRequest(BaseModel):
    project: Optional[str] = None
    session_id: Optional[str] = None
    query: Optional[str] = None
    obs_type: Optional[str] = None
    obs_types: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    total: Optional[int] = None
    full: Optional[int] = None
    full_field: Optional[str] = None
    show_tokens: Optional[bool] = None
    wrap: Optional[bool] = None


class ConsolidateRequest(BaseModel):
    """Request to consolidate similar memories."""
    project: str = Field(..., description="Project to consolidate")
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="Similarity threshold (0.0-1.0)")
    keep_strategy: str = Field(default="newest", description="Strategy: newest, oldest, highest_importance")
    obs_type: Optional[str] = Field(default=None, description="Filter by observation type")
    dry_run: bool = Field(default=False, description="Preview without making changes")
    limit: int = Field(default=100, ge=1, le=1000, description="Max observations to analyze")


class CleanupRequest(BaseModel):
    """Request to cleanup stale memories."""
    project: str = Field(..., description="Project to clean up")
    max_age_days: int = Field(default=90, ge=1, description="Only consider observations older than this")
    min_access_count: int = Field(default=0, ge=0, description="Only consider observations with access count <= this")
    dry_run: bool = Field(default=False, description="Preview without deleting")
    limit: int = Field(default=100, ge=1, le=1000, description="Max observations to consider")


@app.get("/", response_class=HTMLResponse)
def read_root():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/memories")
async def add_memory(mem: MemoryInput, request: Request):
    _check_token(request)
    try:
        manager = get_manager()
        obs = await manager.add_observation(
            content=mem.content,
            obs_type=mem.obs_type,
            project=mem.project,
            session_id=mem.session_id,
            tags=mem.tags,
            metadata=mem.metadata,
            title=mem.title,
            summarize=mem.summarize,
        )
        if not obs:
            return {"status": "skipped", "reason": "private"}
        return obs.model_dump()
    except Exception as exc:
        logger.error(f"Internal error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/memories/batch")
async def add_memories_batch(batch: BatchMemoryInput, request: Request):
    """Add multiple memories in a single batch request.

    This is more efficient than making multiple individual requests when adding
    many observations at once.

    Returns:
        - added: Number of observations successfully added
        - skipped: Number skipped (e.g., private content)
        - failed: Number that failed to add
        - ids: List of added observation IDs
        - errors: List of error details for failed items
    """
    _check_token(request)
    try:
        manager = get_manager()
        observations = [
            {
                "content": mem.content,
                "obs_type": mem.obs_type,
                "project": mem.project,
                "session_id": mem.session_id,
                "tags": mem.tags,
                "metadata": mem.metadata,
                "title": mem.title,
                "summarize": mem.summarize,
            }
            for mem in batch.observations
        ]
        result = await manager.add_observations_batch(observations)
        return result
    except Exception as exc:
        logger.error(f"Batch add error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/observations/batch/delete")
async def delete_observations_batch(batch: BatchDeleteInput, request: Request):
    """Delete multiple observations in a single batch request.

    This is more efficient than making multiple individual delete requests.

    Returns:
        - deleted: Number of observations successfully deleted
        - not_found: Number that were not found
        - ids: List of deleted observation IDs
    """
    _check_token(request)
    try:
        # Validate all IDs are valid UUIDs
        for obs_id in batch.ids:
            _validate_uuid(obs_id)

        manager = get_manager()
        result = await manager.delete_observations_batch(batch.ids)
        return result
    except (HTTPException, AiMemError):
        raise
    except Exception as exc:
        logger.error(f"Batch delete error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/events")
async def ingest_event(req: EventIngestRequest, request: Request):
    """Ingest a raw event from any LLM host.

    This endpoint uses host adapters to parse the payload into a canonical
    ToolEvent, then converts it to an observation. Supports idempotency via
    event_id to prevent duplicates from retried hooks.

    Example:
        POST /api/events
        {
            "host": "gemini",
            "payload": {
                "tool_name": "Read",
                "tool_input": {"path": "/src/main.py"},
                "tool_response": "file contents..."
            }
        }
    """
    _check_token(request)
    try:
        # Get the appropriate adapter for this host
        adapter = get_adapter(req.host)

        # Try to parse as tool event first (most common)
        event = adapter.parse_tool_event(req.payload)

        if not event:
            # Try user prompt event
            prompt_event = adapter.parse_user_prompt(req.payload)
            if prompt_event:
                # Convert user prompt to observation
                manager = get_manager()
                obs = await manager.add_observation(
                    content=prompt_event.content,
                    obs_type="user_prompt",
                    project=req.project or prompt_event.context.project,
                    session_id=req.session_id or prompt_event.session_id,
                    tags=req.tags or ["user-prompt", req.host],
                    metadata=prompt_event.metadata,
                    summarize=req.summarize,
                    event_id=prompt_event.event_id,
                    host=req.host,
                )
                if not obs:
                    return {"status": "skipped", "reason": "private"}
                return {"status": "ok", "observation": obs.model_dump(), "event_type": "user_prompt"}

            # Try session event
            session_event = adapter.parse_session_event(req.payload)
            if session_event:
                manager = get_manager()
                obs = await manager.add_observation(
                    content=f"Session: {session_event.goal or 'started'}",
                    obs_type="session",
                    project=req.project or session_event.context.project,
                    session_id=req.session_id or session_event.session_id,
                    tags=req.tags or ["session", req.host],
                    metadata=session_event.metadata,
                    summarize=False,
                    event_id=session_event.event_id,
                    host=req.host,
                )
                if not obs:
                    return {"status": "skipped", "reason": "private"}
                return {"status": "ok", "observation": obs.model_dump(), "event_type": "session"}

            # Could not parse the payload
            raise PayloadParseError(
                req.host,
                f"Could not parse payload with {req.host} adapter. Check payload format."
            )

        # Build content from tool event
        parts = []
        if event.tool.name:
            parts.append(f"Tool: {event.tool.name}")
        if event.tool.input:
            input_str = json.dumps(event.tool.input) if isinstance(event.tool.input, dict) else str(event.tool.input)
            parts.append(f"Input: {input_str}")

        # Get tool output
        output_str = ""
        if event.tool.output:
            output_str = json.dumps(event.tool.output) if isinstance(event.tool.output, dict) else str(event.tool.output)

        # Endless Mode: compress output if enabled
        endless_metadata = None
        from .endless import get_endless_manager, compress_if_endless_mode

        endless_manager = get_endless_manager()
        if endless_manager.enabled and output_str and len(output_str) >= endless_manager.config.min_output_for_compression:
            # Generate observation ID early for archive linking
            obs_id = str(uuid.uuid4())
            compressed_output, endless_metadata = await compress_if_endless_mode(
                content=output_str,
                tool_name=event.tool.name or "unknown",
                tool_input=event.tool.input,
                observation_id=obs_id,
                session_id=req.session_id or event.session_id or "",
                project=req.project or event.context.project or "",
                metadata={"original_length": len(output_str)},
            )
            if compressed_output:
                output_str = compressed_output
                logger.debug(f"Endless Mode compressed output: {endless_metadata}")

        if output_str:
            parts.append(f"Output: {output_str}")

        content = "\n".join(parts)
        if not content.strip():
            return {"status": "skipped", "reason": "empty_content"}

        # Merge tags: event tags + request tags + host
        tags = list(set(req.tags + [req.host, "tool", "auto-ingested"]))
        if endless_metadata:
            tags.append("endless-compressed")

        # Build metadata
        obs_metadata = {
            "tool_name": event.tool.name,
            "tool_success": event.tool.success,
            "tool_latency_ms": event.tool.latency_ms,
            "source_host": event.source.host,
            **event.metadata,
        }
        if endless_metadata:
            obs_metadata["endless"] = endless_metadata

        # Create observation with idempotency
        manager = get_manager()
        obs = await manager.add_observation(
            content=content,
            obs_type="tool_output",
            project=req.project or event.context.project,
            session_id=req.session_id or event.session_id,
            tags=tags,
            metadata=obs_metadata,
            summarize=req.summarize,
            event_id=event.event_id,
            host=req.host,
        )

        if not obs:
            return {"status": "skipped", "reason": "private"}

        response_data = {
            "status": "ok",
            "observation": obs.model_dump(),
            "event_type": "tool_use",
            "event_id": event.event_id,
        }
        if endless_metadata:
            response_data["endless_mode"] = {
                "compressed": True,
                "compression_ratio": endless_metadata.get("compression_ratio"),
                "tokens_saved": endless_metadata.get("original_tokens", 0) - endless_metadata.get("compressed_tokens", 0),
            }
        return response_data

    except (HTTPException, AiMemError):
        raise
    except Exception as exc:
        logger.error(f"Event ingestion error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/search")
async def search_memories(
    request: Request,
    response: Response,
    query: str,
    limit: int = 10,
    project: Optional[str] = None,
    obs_type: Optional[str] = None,
    session_id: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    since: Optional[str] = None,
    tags: Optional[str] = None,
    show_tokens: Optional[bool] = None,
):
    _check_token(request)
    if date_start is None and since is not None:
        date_start = since
    try:
        limit = min(max(1, limit), MAX_LIMIT)
        manager = get_manager()
        results = await manager.search(
            query,
            limit=limit,
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=date_start,
            date_end=date_end,
            since=since,
            tag_filters=_parse_list_param(tags),
        )
        payload = [item.model_dump() for item in results]
        cache_hit = manager.search_cache_hit
        response.headers["X-AI-MEM-Search-Cache"] = "hit" if cache_hit else "miss"
        if show_tokens:
            for row, item in zip(payload, results):
                row["token_estimate"] = estimate_tokens(item.summary or "")
        return payload
    except Exception as exc:
        logger.error(f"Internal error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/timeline")
async def get_timeline(
    request: Request,
    anchor_id: Optional[str] = None,
    query: Optional[str] = None,
    depth_before: int = 3,
    depth_after: int = 3,
    project: Optional[str] = None,
    obs_type: Optional[str] = None,
    session_id: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    since: Optional[str] = None,
    tags: Optional[str] = None,
    show_tokens: Optional[bool] = None,
):
    _check_token(request)
    if date_start is None and since is not None:
        date_start = since
    try:
        manager = get_manager()
        results = await manager.timeline(
            anchor_id=anchor_id,
            query=query,
            depth_before=depth_before,
            depth_after=depth_after,
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=date_start,
            date_end=date_end,
            since=since,
            tag_filters=_parse_list_param(tags),
        )
        payload = [item.model_dump() for item in results]
        if show_tokens:
            for row, item in zip(payload, results):
                row["token_estimate"] = estimate_tokens(item.summary or "")
        return payload
    except Exception as exc:
        logger.error(f"Internal error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/context")
async def context_alias(
    request: Request,
    project: Optional[str] = None,
    session_id: Optional[str] = None,
    query: Optional[str] = None,
    obs_type: Optional[str] = None,
    obs_types: Optional[str] = None,
    tags: Optional[str] = None,
    total: Optional[int] = None,
    full: Optional[int] = None,
    full_field: Optional[str] = None,
    show_tokens: Optional[bool] = None,
    wrap: Optional[bool] = None,
):
    return await inject_context(
        request=request,
        project=project,
        session_id=session_id,
        query=query,
        obs_type=obs_type,
        obs_types=obs_types,
        tags=tags,
        total=total,
        full=full,
        full_field=full_field,
        show_tokens=show_tokens,
        wrap=wrap,
    )


@app.post("/api/context")
async def context_alias_post(payload: ContextRequest, request: Request):
    return await inject_context_post(payload, request)


@app.get("/api/context/config")
def context_config(request: Request):
    _check_token(request)
    return get_manager().config.context.model_dump()


@app.get("/api/context/preview")
async def preview_context(
    request: Request,
    project: Optional[str] = None,
    session_id: Optional[str] = None,
    query: Optional[str] = None,
    obs_type: Optional[str] = None,
    obs_types: Optional[str] = None,
    tags: Optional[str] = None,
    total: Optional[int] = None,
    full: Optional[int] = None,
    full_field: Optional[str] = None,
    show_tokens: Optional[bool] = None,
    wrap: Optional[bool] = None,
):
    _check_token(request)
    context_text, _ = await build_context(
        get_manager(),
        project=project,
        session_id=session_id,
        query=query,
        obs_type=obs_type,
        obs_types=_parse_list_param(obs_types),
        tag_filters=_parse_list_param(tags),
        total_count=total,
        full_count=full,
        full_field=full_field,
        show_tokens=show_tokens,
        wrap=False if wrap is None else wrap,
    )
    return Response(context_text, media_type="text/plain")


@app.post("/api/context/preview")
async def preview_context_post(payload: ContextRequest, request: Request):
    _check_token(request)
    context_text, meta = await build_context(
        get_manager(),
        project=payload.project,
        session_id=payload.session_id,
        query=payload.query,
        obs_type=payload.obs_type,
        obs_types=payload.obs_types,
        tag_filters=payload.tags,
        total_count=payload.total,
        full_count=payload.full,
        full_field=payload.full_field,
        show_tokens=payload.show_tokens,
        wrap=False if payload.wrap is None else payload.wrap,
    )
    return {"context": context_text, "metadata": meta}


@app.get("/api/stats/sessions")
async def get_session_stats_api(
    request: Request,
    project: Optional[str] = None,
    limit: int = 50,
):
    _check_token(request)
    limit = min(max(1, limit), MAX_LIMIT)
    stats = await get_manager().db.get_session_stats(project=project, limit=limit)
    return stats


@app.get("/api/context/inject")
async def inject_context(
    request: Request,
    project: Optional[str] = None,
    session_id: Optional[str] = None,
    query: Optional[str] = None,
    obs_type: Optional[str] = None,
    obs_types: Optional[str] = None,
    tags: Optional[str] = None,
    total: Optional[int] = None,
    full: Optional[int] = None,
    full_field: Optional[str] = None,
    show_tokens: Optional[bool] = None,
    wrap: Optional[bool] = None,
):
    _check_token(request)
    context_text, _ = await build_context(
        get_manager(),
        project=project,
        session_id=session_id,
        query=query,
        obs_type=obs_type,
        obs_types=_parse_list_param(obs_types),
        tag_filters=_parse_list_param(tags),
        total_count=total,
        full_count=full,
        full_field=full_field,
        show_tokens=show_tokens,
        wrap=wrap,
    )
    return Response(context_text, media_type="text/plain")


@app.post("/api/context/inject")
async def inject_context_post(payload: ContextRequest, request: Request):
    _check_token(request)
    context_text, meta = await build_context(
        get_manager(),
        project=payload.project,
        session_id=payload.session_id,
        query=payload.query,
        obs_type=payload.obs_type,
        obs_types=payload.obs_types,
        tag_filters=payload.tags,
        total_count=payload.total,
        full_count=payload.full,
        full_field=payload.full_field,
        show_tokens=payload.show_tokens,
        wrap=payload.wrap,
    )
    return {"context": context_text, "metadata": meta}


@app.post("/api/observations")
async def get_observations(payload: ObservationIds, request: Request):
    _check_token(request)
    try:
        return await get_manager().get_observations(payload.ids)
    except Exception as exc:
        logger.error(f"Internal error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/observations/{obs_id}")
async def get_observation(obs_id: str, request: Request):
    _check_token(request)
    _validate_uuid(obs_id)
    try:
        results = await get_manager().get_observations([obs_id])
        if not results:
            raise ObservationNotFoundError(obs_id)
        return results[0]
    except (HTTPException, AiMemError):
        raise
    except Exception as exc:
        logger.error(f"Internal error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.patch("/api/observations/{obs_id}")
async def update_observation(obs_id: str, payload: ObservationUpdate, request: Request):
    _check_token(request)
    _validate_uuid(obs_id)
    if payload.tags is None:
        raise ValidationError(field="tags", message="No updates provided")
    updated = await get_manager().update_observation_tags(obs_id, payload.tags)
    if not updated:
        raise ObservationNotFoundError(obs_id)
    return {"success": True}


@app.get("/api/observation/{obs_id}")
async def get_observation_alias(obs_id: str, request: Request):
    _validate_uuid(obs_id)
    return await get_observation(obs_id, request)


@app.get("/api/projects")
async def list_projects(request: Request):
    _check_token(request)
    return await get_manager().list_projects()


@app.get("/api/sessions")
async def list_sessions(
    request: Request,
    project: Optional[str] = None,
    active_only: bool = False,
    goal: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    limit: Optional[int] = None,
):
    _check_token(request)
    return await get_manager().list_sessions(
        project=project,
        active_only=active_only,
        goal_query=goal,
        date_start=date_start,
        date_end=date_end,
        limit=min(max(1, limit), MAX_LIMIT) if limit is not None else None,
    )


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str, request: Request):
    _check_token(request)
    session = await get_manager().get_session(session_id)
    if not session:
        raise SessionNotFoundError(session_id)
    return session


@app.get("/api/sessions/{session_id}/observations")
async def get_session_observations(
    session_id: str,
    request: Request,
    limit: Optional[int] = None,
):
    _check_token(request)
    session = await get_manager().get_session(session_id)
    if not session:
        raise SessionNotFoundError(session_id)
    limit = min(max(1, limit), MAX_LIMIT) if limit is not None else None
    return await get_manager().export_observations(session_id=session_id, limit=limit)


@app.post("/api/sessions/start")
async def start_session(payload: SessionStartRequest, request: Request):
    _check_token(request)
    project = payload.project or os.getcwd()
    session = await get_manager().start_session(
        project=project,
        goal=payload.goal or "",
        session_id=payload.session_id,
    )
    return session.model_dump()


@app.post("/api/sessions/end")
async def end_session(payload: SessionEndRequest, request: Request):
    _check_token(request)
    if payload.session_id:
        session = await get_manager().end_session(payload.session_id)
    else:
        session = await get_manager().end_latest_session(payload.project)
    if not session:
        raise SessionNotFoundError(payload.session_id or "latest")
    return session.model_dump()


@app.get("/api/stats")
async def get_stats(
    request: Request,
    project: Optional[str] = None,
    obs_type: Optional[str] = None,
    session_id: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    since: Optional[str] = None,
    tags: Optional[str] = None,
    tag_limit: Optional[int] = None,
    day_limit: Optional[int] = None,
    type_tag_limit: Optional[int] = None,
):
    _check_token(request)
    if date_start is None and since is not None:
        date_start = since
    manager = get_manager()
    payload = await manager.get_stats(
        project=project,
        obs_type=obs_type,
        session_id=session_id,
        date_start=date_start,
        date_end=date_end,
        since=since,
        tag_filters=_parse_list_param(tags),
        tag_limit=min(max(1, tag_limit if tag_limit is not None else 10), MAX_LIMIT),
        day_limit=min(max(1, day_limit if day_limit is not None else 14), MAX_LIMIT),
        type_tag_limit=min(max(1, type_tag_limit if type_tag_limit is not None else 3), MAX_LIMIT),
    )
    payload["search_cache"] = manager.search_cache_summary()
    return payload


@app.get("/api/cache")
async def get_cache_metrics(request: Request):
    """Get current search cache metrics.

    Returns cache configuration, current entries, hit/miss counts, and hit rate.
    """
    _check_token(request)
    return get_manager().search_cache_summary()


@app.post("/api/cache/clear")
async def clear_cache(request: Request):
    """Clear the search cache and reset metrics.

    Returns the number of entries that were cleared and previous hit/miss counts.
    """
    _check_token(request)
    return get_manager().clear_search_cache()


@app.get("/api/tags")
async def list_tags(
    request: Request,
    project: Optional[str] = None,
    obs_type: Optional[str] = None,
    session_id: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    tags: Optional[str] = None,
    limit: Optional[int] = 50,
):
    _check_token(request)
    return await get_manager().list_tags(
        project=project,
        obs_type=obs_type,
        session_id=session_id,
        date_start=date_start,
        date_end=date_end,
        tag_filters=_parse_list_param(tags),
        limit=limit,
    )


@app.post("/api/tags/rename")
async def rename_tag(payload: TagRenameRequest, request: Request):
    _check_token(request)
    old_tag = payload.old_tag.strip()
    new_tag = payload.new_tag.strip()
    if not old_tag or not new_tag:
        raise HTTPException(status_code=400, detail="Both old_tag and new_tag are required")
    updated = await get_manager().rename_tag(
        old_tag=old_tag,
        new_tag=new_tag,
        project=payload.project,
        obs_type=payload.obs_type,
        session_id=payload.session_id,
        date_start=payload.date_start,
        date_end=payload.date_end,
        tag_filters=_parse_list_param(payload.tags),
    )
    return {"success": True, "updated": updated}


@app.post("/api/tags/add")
async def add_tag(payload: TagAddRequest, request: Request):
    _check_token(request)
    value = payload.tag.strip()
    if not value:
        raise HTTPException(status_code=400, detail="tag is required")
    updated = await get_manager().add_tag(
        tag=value,
        project=payload.project,
        obs_type=payload.obs_type,
        session_id=payload.session_id,
        date_start=payload.date_start,
        date_end=payload.date_end,
        tag_filters=_parse_list_param(payload.tags),
    )
    return {"success": True, "updated": updated}


@app.post("/api/tags/delete")
async def delete_tag(payload: TagDeleteRequest, request: Request):
    _check_token(request)
    value = payload.tag.strip()
    if not value:
        raise HTTPException(status_code=400, detail="tag is required")
    updated = await get_manager().delete_tag(
        tag=value,
        project=payload.project,
        obs_type=payload.obs_type,
        session_id=payload.session_id,
        date_start=payload.date_start,
        date_end=payload.date_end,
        tag_filters=_parse_list_param(payload.tags),
    )
    return {"success": True, "updated": updated}


@app.get("/api/observations")
async def list_observations(
    request: Request,
    project: Optional[str] = None,
    session_id: Optional[str] = None,
    obs_type: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    since: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 50,
):
    _check_token(request)
    if date_start is None and since is not None:
        date_start = since
    return await get_manager().export_observations(
        project=project,
        session_id=session_id,
        obs_type=obs_type,
        date_start=date_start,
        date_end=date_end,
        tag_filters=_parse_list_param(tags),
        limit=min(limit, MAX_LIMIT),
    )


@app.delete("/api/observations/{obs_id}")
async def delete_observation(obs_id: str, request: Request):
    _check_token(request)
    _validate_uuid(obs_id)
    if await get_manager().delete_observation(obs_id):
        return {"success": True}
    raise ObservationNotFoundError(obs_id)


@app.post("/api/projects/delete")
async def delete_project(payload: ProjectDelete, request: Request):
    _check_token(request)
    deleted = await get_manager().delete_project(payload.project)
    return {"success": True, "deleted": deleted}


@app.get("/api/export")
async def export_observations(
    request: Request,
    project: Optional[str] = None,
    session_id: Optional[str] = None,
    obs_type: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    since: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 1000,
    format: Optional[str] = None,
):
    _check_token(request)
    if date_start is None and since is not None:
        date_start = since
    data = await get_manager().export_observations(
        project=project,
        session_id=session_id,
        obs_type=obs_type,
        date_start=date_start,
        date_end=date_end,
        tag_filters=_parse_list_param(tags),
        limit=min(limit, MAX_LIMIT),
    )
    fmt = (format or "json").lower()
    if fmt == "json":
        return data
    if fmt in {"jsonl", "ndjson"}:
        lines = "\n".join(json.dumps(item, ensure_ascii=True) for item in data)
        return Response(lines, media_type="application/x-ndjson")
    if fmt == "csv":
        output = io.StringIO()
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
        writer = csv.DictWriter(output, fieldnames=fields)
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
        return Response(output.getvalue(), media_type="text/csv")
    raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")


@app.post("/api/import")
async def import_observations(payload: ImportPayload, request: Request):
    _check_token(request)
    manager = get_manager()
    imported = 0
    for item in payload.items:
        content = item.get("content")
        obs_type = item.get("type") or "note"
        if not content:
            continue
        item_project = item.get("project")
        session_id = item.get("session_id")
        if payload.project and item_project and payload.project != item_project:
            session_id = None
        result = await manager.add_observation(
            content=content,
            obs_type=obs_type,
            project=payload.project or item_project,
            session_id=session_id,
            tags=item.get("tags") or [],
            metadata=item.get("metadata") or {},
            title=item.get("title"),
            summarize=False,
            dedupe=True,
            summary=item.get("summary"),
            created_at=item.get("created_at"),
            importance_score=item.get("importance_score", 0.5),
        )
        if result:
            imported += 1
    return {"success": True, "imported": imported}


@app.post("/api/summarize")
async def summarize_project(payload: SummarizeRequest, request: Request):
    _check_token(request)
    result = await get_manager().summarize_project(
        project=payload.project,
        session_id=payload.session_id,
        limit=min(max(1, payload.count), MAX_LIMIT),
        obs_type=payload.obs_type,
        store=payload.store,
        tags=payload.tags or None,
    )
    if not result:
        return {"status": "empty"}
    obs = result.get("observation")
    return {
        "status": "ok",
        "summary": result.get("summary", ""),
        "metadata": result.get("metadata"),
        "observation": obs.model_dump() if obs else None,
    }


@app.get("/api/health")
def health(request: Request):
    _check_token(request)
    return {"status": "ok"}


@app.get("/api/readiness")
def readiness(request: Request):
    _check_token(request)
    return {"status": "ok"}


@app.get("/api/version")
def version(request: Request):
    _check_token(request)
    return {"version": __version__}


# =============================================================================
# Endless Mode Endpoints
# =============================================================================

class EndlessModeRequest(BaseModel):
    """Request body for enabling Endless Mode."""
    session_id: Optional[str] = None


@app.get("/api/endless/stats")
async def get_endless_stats(request: Request):
    """Get Endless Mode statistics."""
    _check_token(request)
    from .endless import get_endless_manager
    manager = get_endless_manager()
    stats = manager.get_stats()
    return {
        "enabled": manager.enabled,
        "stats": stats.to_dict(),
        "config": {
            "target_observation_tokens": manager.config.target_observation_tokens,
            "compression_ratio": manager.config.compression_ratio,
            "compression_method": manager.config.compression_method,
            "enable_archive": manager.config.enable_archive,
        },
    }


@app.post("/api/endless/enable")
async def enable_endless_mode(payload: EndlessModeRequest, request: Request):
    """Enable Endless Mode for extended sessions."""
    _check_token(request)
    from .endless import get_endless_manager
    manager = get_endless_manager()
    manager.enable(session_id=payload.session_id)
    return {
        "status": "enabled",
        "session_id": payload.session_id,
        "config": {
            "target_observation_tokens": manager.config.target_observation_tokens,
            "compression_ratio": manager.config.compression_ratio,
        },
    }


@app.post("/api/endless/disable")
async def disable_endless_mode(request: Request):
    """Disable Endless Mode."""
    _check_token(request)
    from .endless import get_endless_manager
    manager = get_endless_manager()
    final_stats = manager.get_stats()
    manager.disable()
    return {
        "status": "disabled",
        "final_stats": final_stats.to_dict(),
    }


@app.get("/api/endless/archive/{observation_id}")
async def get_archive_entry(
    observation_id: str,
    session_id: str,
    project: str,
    request: Request,
):
    """Get the full (uncompressed) output from archive memory."""
    _check_token(request)
    from .endless import get_endless_manager
    manager = get_endless_manager()

    if not manager.archive:
        raise HTTPException(status_code=404, detail="Archive memory not enabled")

    full_output = manager.get_full_output(observation_id, session_id, project)
    if not full_output:
        raise HTTPException(status_code=404, detail="Archive entry not found")

    return {
        "observation_id": observation_id,
        "full_output": full_output,
        "session_id": session_id,
        "project": project,
    }


@app.get("/api/stream")
async def stream_memories(
    request: Request,
    project: Optional[str] = None,
    session_id: Optional[str] = None,
    query: Optional[str] = None,
    obs_type: Optional[str] = None,
    tags: Optional[str] = None,
):
    """Stream memories in real-time via Server-Sent Events.

    Authentication: Use Authorization header (Bearer token) or X-AI-Mem-Token header.
    """
    _check_token(request)
    manager = get_manager()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
    tag_filters = _parse_list_param(tags) or []
    query_text = (query or "").strip().lower()

    def _matches(obs: Dict[str, Any]) -> bool:
        if session_id and obs.get("session_id") != session_id:
            return False
        if project and obs.get("project") != project:
            return False
        if obs_type and obs.get("type") != obs_type:
            return False
        if tag_filters:
            obs_tags = obs.get("tags") or []
            if not any(tag in obs_tags for tag in tag_filters):
                return False
        if query_text:
            haystack_parts = [
                obs.get("summary") or "",
                obs.get("content") or "",
                obs.get("title") or "",
                " ".join(obs.get("tags") or []),
            ]
            haystack = " ".join(haystack_parts).lower()
            if query_text not in haystack:
                return False
        return True

    def _listener(obs_obj: Any) -> None:
        obs = obs_obj.model_dump() if hasattr(obs_obj, "model_dump") else {}
        if not obs or not _matches(obs):
            return
        summary_text = obs.get("summary") or obs.get("content") or ""
        token_estimate = estimate_tokens(summary_text)
        payload = {
            "id": obs.get("id"),
            "summary": summary_text,
            "project": obs.get("project") or "",
            "type": obs.get("type") or "",
            "created_at": obs.get("created_at"),
            "session_id": obs.get("session_id"),
            "tags": obs.get("tags") or [],
            "token_estimate": token_estimate,
        }
        loop.call_soon_threadsafe(queue.put_nowait, payload)

    remove_listener = manager.add_listener(_listener)

    async def _event_stream():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(item, ensure_ascii=True)}\n\n"
        finally:
            remove_listener()

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


# =============================================================================
# Memory Maintenance Endpoints (Phase 3)
# =============================================================================

@app.post("/api/memory/consolidate")
async def consolidate_memories(payload: ConsolidateRequest, request: Request):
    """Consolidate similar memories by marking duplicates as superseded.

    This helps reduce redundancy and keep memory lean over time.
    Uses embedding similarity to find similar observations and marks
    the older/less important ones as superseded based on the strategy.

    Strategies:
        - newest: Keep the most recent observation
        - oldest: Keep the oldest observation
        - highest_importance: Keep the one with highest importance_score

    Returns:
        - analyzed: Number of observations analyzed
        - pairs_found: Number of similar pairs found
        - consolidated: Number marked as superseded
        - kept_ids: IDs of observations kept
        - superseded_ids: IDs of observations marked as superseded
        - dry_run: Whether this was a preview
    """
    _check_token(request)
    try:
        manager = get_manager()
        result = await manager.consolidate_memories(
            project=payload.project,
            similarity_threshold=payload.similarity_threshold,
            keep_strategy=payload.keep_strategy,
            obs_type=payload.obs_type,
            dry_run=payload.dry_run,
            limit=payload.limit,
        )
        return result
    except Exception as exc:
        logger.error(f"Consolidation error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/memory/cleanup")
async def cleanup_stale_memories(payload: CleanupRequest, request: Request):
    """Remove old, rarely-accessed memories.

    This implements memory decay - observations that are old and haven't
    been accessed will be removed to keep the memory store efficient.

    Returns:
        - candidates_found: Number of stale observations found
        - deleted: Number actually deleted (or would be if dry_run)
        - deleted_ids: IDs of deleted observations
        - dry_run: Whether this was a preview
    """
    _check_token(request)
    try:
        manager = get_manager()
        result = await manager.cleanup_stale_memories(
            project=payload.project,
            max_age_days=payload.max_age_days,
            min_access_count=payload.min_access_count,
            dry_run=payload.dry_run,
            limit=payload.limit,
        )
        return result
    except Exception as exc:
        logger.error(f"Cleanup error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/memory/similar")
async def find_similar_observations(
    request: Request,
    project: str,
    similarity_threshold: float = 0.85,
    obs_type: Optional[str] = None,
    limit: int = 100,
):
    """Find pairs of similar observations that might be duplicates.

    Useful to preview what would be consolidated before running consolidate.

    Returns:
        List of (obs_id_1, obs_id_2, similarity_score) pairs
    """
    _check_token(request)
    try:
        manager = get_manager()
        pairs = await manager.find_similar_observations(
            project=project,
            similarity_threshold=similarity_threshold,
            use_embeddings=True,
            obs_type=obs_type,
            limit=min(limit, MAX_LIMIT),
        )
        return [
            {"obs_id_1": p[0], "obs_id_2": p[1], "similarity": p[2]}
            for p in pairs
        ]
    except Exception as exc:
        logger.error(f"Find similar error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/search/rerank")
async def search_with_rerank(
    request: Request,
    response: Response,
    query: str,
    limit: int = 10,
    project: Optional[str] = None,
    obs_type: Optional[str] = None,
    session_id: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    since: Optional[str] = None,
    tags: Optional[str] = None,
    stage1_limit: int = 50,
    show_tokens: Optional[bool] = None,
):
    """Two-stage search with reranking for improved precision.

    Stage 1: BM25 + Vector search (high recall) - fetches stage1_limit candidates
    Stage 2: Embedding-based reranking (high precision) - returns top 'limit' results

    This provides better ranking than standard search at the cost of slightly
    more computation. Use for queries where precision matters.

    Returns:
        List of observations with rerank_score field added
    """
    _check_token(request)
    if date_start is None and since is not None:
        date_start = since
    try:
        limit = min(max(1, limit), MAX_LIMIT)
        stage1_limit = min(max(1, stage1_limit), MAX_LIMIT)
        manager = get_manager()
        results = await manager.search_with_rerank(
            query=query,
            limit=limit,
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=date_start,
            date_end=date_end,
            since=since,
            tag_filters=_parse_list_param(tags),
            stage1_limit=stage1_limit,
        )
        payload = [item.model_dump() for item in results]
        response.headers["X-AI-MEM-Search-Type"] = "reranked"
        if show_tokens:
            for row, item in zip(payload, results):
                row["token_estimate"] = estimate_tokens(item.summary or "")
        return payload
    except Exception as exc:
        logger.error(f"Rerank search error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# Entity Graph Endpoints (Phase 4)
# =============================================================================

@app.get("/api/entities")
async def search_entities(
    request: Request,
    query: str,
    project: Optional[str] = None,
    entity_types: Optional[str] = None,
    limit: int = 20,
):
    """Search for entities by name.

    Args:
        query: Search query
        project: Filter by project
        entity_types: Comma-separated entity types (file, function, class, etc.)
        limit: Maximum entities to return

    Returns:
        List of matching entities
    """
    _check_token(request)
    try:
        manager = get_manager()
        if not manager.entity_graph:
            return {"error": "Entity graph not enabled"}

        types = _parse_list_param(entity_types)
        results = await manager.entity_graph.find_entities(
            query=query,
            project=project,
            entity_types=types,
            limit=min(limit, MAX_LIMIT),
        )
        return results
    except Exception as exc:
        logger.error(f"Entity search error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/entities/{entity_id}/related")
async def get_related_entities(
    entity_id: str,
    request: Request,
    relation_types: Optional[str] = None,
    max_depth: int = 1,
    limit: int = 20,
):
    """Get entities related to a given entity.

    Supports multi-hop traversal for finding indirect relationships.

    Args:
        entity_id: Starting entity ID
        relation_types: Comma-separated relation types to filter
        max_depth: Maximum traversal depth (1 = direct, 2+ = multi-hop)
        limit: Maximum entities to return

    Returns:
        List of related entities with relationship info
    """
    _check_token(request)
    try:
        manager = get_manager()
        if not manager.entity_graph:
            return {"error": "Entity graph not enabled"}

        types = _parse_list_param(relation_types)
        results = await manager.entity_graph.get_related_entities(
            entity_id=entity_id,
            relation_types=types,
            max_depth=min(max_depth, 3),  # Limit depth to prevent expensive queries
            limit=min(limit, MAX_LIMIT),
        )
        return results
    except Exception as exc:
        logger.error(f"Related entities error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/entities/{entity_id}/observations")
async def get_entity_observations(
    entity_id: str,
    request: Request,
    limit: int = 20,
):
    """Get observation IDs that mention an entity.

    Args:
        entity_id: Entity ID
        limit: Maximum observations to return

    Returns:
        List of observation IDs
    """
    _check_token(request)
    try:
        manager = get_manager()
        if not manager.entity_graph:
            return {"error": "Entity graph not enabled"}

        obs_ids = await manager.entity_graph.get_entity_observations(
            entity_id=entity_id,
            limit=min(limit, MAX_LIMIT),
        )

        # Optionally fetch full observations
        if obs_ids:
            observations = await manager.get_observations(obs_ids)
            return {"observation_ids": obs_ids, "observations": observations}

        return {"observation_ids": [], "observations": []}
    except Exception as exc:
        logger.error(f"Entity observations error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/graph/stats")
async def get_graph_stats(
    request: Request,
    project: Optional[str] = None,
):
    """Get statistics about the entity graph.

    Returns:
        - entities: Total entity count
        - relations: Total relation count
        - entities_by_type: Count by entity type
        - relations_by_type: Count by relation type
    """
    _check_token(request)
    try:
        manager = get_manager()
        if not manager.entity_graph:
            return {"error": "Entity graph not enabled", "entities": 0, "relations": 0}

        stats = await manager.entity_graph.get_graph_stats(project=project)
        return stats
    except Exception as exc:
        logger.error(f"Graph stats error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# User Memory Endpoints (Phase 4)
# =============================================================================


class UserMemoryInput(BaseModel):
    """Input for adding a user memory."""
    content: str = Field(..., description="Memory content")
    obs_type: str = Field(default="preference", description="Observation type")
    tags: List[str] = Field(default_factory=list, description="Tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    title: Optional[str] = Field(default=None, description="Optional title")
    summarize: bool = Field(default=True, description="Whether to summarize")


@app.post("/api/user/memories")
async def add_user_memory(
    request: Request,
    data: UserMemoryInput,
):
    """Add a user-level memory.

    User memories are global and not tied to any project.
    Use for user preferences, conventions, and cross-project context.
    """
    _check_token(request)
    try:
        manager = get_manager()
        obs = await manager.add_user_memory(
            content=data.content,
            obs_type=data.obs_type,
            tags=data.tags,
            metadata=data.metadata,
            title=data.title,
            summarize=data.summarize,
        )
        if obs is None:
            return {"ok": False, "message": "Content filtered"}
        return {"ok": True, "id": obs.id}
    except Exception as exc:
        logger.error(f"Add user memory error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/user/memories")
async def list_user_memories(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    obs_type: Optional[str] = None,
    tag: Optional[List[str]] = Query(default=None),
):
    """List user-level memories.

    Returns all memories in user scope (not tied to any project).
    """
    _check_token(request)
    try:
        manager = get_manager()
        results = await manager.get_user_memories(
            limit=limit,
            obs_type=obs_type,
            tag_filters=tag,
        )
        return {"memories": [r.model_dump() for r in results], "count": len(results)}
    except Exception as exc:
        logger.error(f"List user memories error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/user/memories/search")
async def search_user_memories(
    request: Request,
    q: str = Query(..., description="Search query"),
    limit: int = Query(default=10, ge=1, le=100),
    obs_type: Optional[str] = None,
    tag: Optional[List[str]] = Query(default=None),
):
    """Search user-level memories.

    Searches across user-scoped memories using hybrid FTS + vector search.
    """
    _check_token(request)
    try:
        manager = get_manager()
        results = await manager.search_user_memories(
            query=q,
            limit=limit,
            obs_type=obs_type,
            tag_filters=tag,
        )
        return {"results": [r.model_dump() for r in results], "count": len(results)}
    except Exception as exc:
        logger.error(f"Search user memories error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/user/memories/export")
async def export_user_memories(
    request: Request,
    path: Optional[str] = Query(default=None, description="Export path"),
):
    """Export user memories to JSON file.

    Default path: ~/.config/ai-mem/user-memory.json
    """
    _check_token(request)
    try:
        manager = get_manager()
        result = await manager.export_user_memories(output_path=path)
        return result
    except Exception as exc:
        logger.error(f"Export user memories error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/user/memories/import")
async def import_user_memories(
    request: Request,
    path: Optional[str] = Query(default=None, description="Import path"),
    merge: bool = Query(default=True, description="Merge with existing"),
):
    """Import user memories from JSON file.

    Default path: ~/.config/ai-mem/user-memory.json
    Set merge=false to replace all existing user memories.
    """
    _check_token(request)
    try:
        manager = get_manager()
        result = await manager.import_user_memories(input_path=path, merge=merge)
        return result
    except Exception as exc:
        logger.error(f"Import user memories error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/user/memories/count")
async def get_user_memory_count(request: Request):
    """Get count of user-level memories."""
    _check_token(request)
    try:
        manager = get_manager()
        count = await manager.get_user_memory_count()
        return {"count": count}
    except Exception as exc:
        logger.error(f"Get user memory count error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# Query Expansion & Intent Detection Endpoints
# =============================================================================


@app.post("/api/intent/detect")
async def detect_intent_endpoint(
    request: Request,
    prompt: str = Query(..., description="Prompt to analyze"),
):
    """Detect intent from a user prompt.

    Analyzes the prompt to extract:
    - Primary action (create, fix, update, etc.)
    - Technical entities (files, functions, classes)
    - Technical concepts
    - Important keywords
    - Generated search query
    """
    _check_token(request)
    try:
        from .intent import detect_intent, should_inject_context

        intent = detect_intent(prompt)
        should_inject, reason = should_inject_context(prompt)

        return {
            "action": intent.action,
            "entities": intent.entities,
            "concepts": intent.concepts,
            "keywords": intent.keywords,
            "query": intent.query,
            "should_inject_context": should_inject,
            "inject_reason": reason,
        }
    except Exception as exc:
        logger.error(f"Detect intent error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/query/expand")
async def expand_query_endpoint(
    request: Request,
    query: str = Query(..., description="Query to expand"),
    max_synonyms: int = Query(2, description="Max synonyms per term"),
    max_terms: int = Query(15, description="Max total expanded terms"),
):
    """Expand a search query with synonyms and variations.

    This LLM-agnostic expansion:
    - Adds technical synonyms (e.g., "auth" → "authentication", "login")
    - Adds case variations (e.g., "get_user" → "getUser", "GetUser")
    - Preserves original terms as highest priority
    """
    _check_token(request)
    try:
        from .intent import expand_query

        result = expand_query(
            query,
            max_synonyms_per_term=max_synonyms,
            max_total_terms=max_terms,
        )

        return {
            "original": result.original,
            "expanded_terms": result.expanded_terms,
            "all_queries": result.all_queries,
            "expansion_count": result.expansion_count,
        }
    except Exception as exc:
        logger.error(f"Expand query error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/query/generate")
async def generate_query_endpoint(
    request: Request,
    prompt: str = Query(..., description="Prompt to generate query from"),
    expand: bool = Query(False, description="Also expand the generated query"),
    max_queries: int = Query(3, description="Max query variants if expanding"),
):
    """Generate search query from a user prompt.

    Optionally expands the query with synonyms for better recall.
    """
    _check_token(request)
    try:
        from .intent import generate_context_query, generate_expanded_queries

        if expand:
            queries = generate_expanded_queries(prompt, max_queries=max_queries)
            return {
                "prompt": prompt,
                "queries": queries,
                "expanded": True,
            }
        else:
            query = generate_context_query(prompt)
            return {
                "prompt": prompt,
                "query": query,
                "expanded": False,
            }
    except Exception as exc:
        logger.error(f"Generate query error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# Inline Tags Endpoints
# =============================================================================


@app.post("/api/tags/parse")
async def parse_tags_endpoint(
    request: Request,
    prompt: str = Query(..., description="Prompt with <mem> tags to parse"),
):
    """Parse inline <mem> tags in a prompt without expanding them.

    Recognizes tags like:
    - <mem query="auth"/>
    - <mem query="database" limit="5" mode="compact"/>
    - <mem query="user" type="decision" tags="api,backend"/>
    """
    _check_token(request)
    try:
        from .inline_tags import parse_prompt

        parsed = parse_prompt(prompt)

        return {
            "original": parsed.original,
            "has_tags": parsed.has_tags,
            "cleaned": parsed.cleaned,
            "tags": [
                {
                    "raw": t.raw,
                    "query": t.query,
                    "limit": t.limit,
                    "project": t.project,
                    "obs_type": t.obs_type,
                    "tags": t.tags,
                    "mode": t.mode,
                }
                for t in parsed.tags
            ],
        }
    except Exception as exc:
        logger.error(f"Parse tags error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/tags/expand")
async def expand_tags_endpoint(
    request: Request,
    prompt: str = Query(..., description="Prompt with <mem> tags to expand"),
    project: str = Query(None, description="Default project for queries"),
):
    """Expand inline <mem> tags in a prompt with actual memory content.

    Replaces <mem> tags with retrieved memory context.
    """
    _check_token(request)
    try:
        from .inline_tags import expand_mem_tags, has_mem_tags

        if not has_mem_tags(prompt):
            return {
                "original": prompt,
                "expanded": prompt,
                "expansions": [],
                "has_tags": False,
            }

        manager = get_manager()
        expanded, expansions = await expand_mem_tags(
            prompt,
            manager,
            default_project=project,
        )

        return {
            "original": prompt,
            "expanded": expanded,
            "expansions": expansions,
            "has_tags": True,
        }
    except Exception as exc:
        logger.error(f"Expand tags error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# Token Budget Endpoints
# =============================================================================


@app.get("/api/tokens/budget")
async def check_token_budget_endpoint(
    request: Request,
    tokens_used: int = Query(..., description="Number of tokens used"),
    model: str = Query(None, description="Model for context window lookup"),
    max_budget: int = Query(None, description="Override max budget"),
):
    """Check if token usage exceeds recommended thresholds.

    Returns warning info if usage is above thresholds:
    - info: >15% of budget
    - warning: >30% of budget
    - critical: >50% of budget
    """
    _check_token(request)
    try:
        from .context import check_token_budget, get_model_context_window

        warning = check_token_budget(
            tokens_used,
            model=model,
            max_budget=max_budget,
        )

        context_window = get_model_context_window(model)

        if warning:
            return {
                "has_warning": True,
                "level": warning.level,
                "message": warning.message,
                "tokens_used": warning.tokens_used,
                "tokens_budget": warning.tokens_budget,
                "percentage": warning.percentage,
                "recommendations": warning.recommendations,
                "model_context_window": context_window,
            }
        else:
            return {
                "has_warning": False,
                "tokens_used": tokens_used,
                "model_context_window": context_window,
            }
    except Exception as exc:
        logger.error(f"Check token budget error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/tokens/models")
async def list_model_context_windows(request: Request):
    """List known model context window sizes.

    Returns a mapping of model names to their context window sizes.
    """
    _check_token(request)
    try:
        from .context import MODEL_CONTEXT_WINDOWS

        return {
            "models": MODEL_CONTEXT_WINDOWS,
            "default": MODEL_CONTEXT_WINDOWS.get("default", 16000),
        }
    except Exception as exc:
        logger.error(f"List models error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


def start_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)

import asyncio
import csv
import io
import json
import os
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
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


_allowed_origins = os.environ.get("AI_MEM_ALLOWED_ORIGINS", "*").split(",")
_allowed_origins = [o.strip() for o in _allowed_origins if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting (optional - requires slowapi)
_rate_limit = os.environ.get("AI_MEM_RATE_LIMIT", "")
if _rate_limit:
    try:
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.util import get_remote_address
        from slowapi.errors import RateLimitExceeded
        
        limiter = Limiter(key_func=get_remote_address, default_limits=[_rate_limit])
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        logger.info(f"Rate limiting enabled: {_rate_limit}")
    except ImportError:
        logger.warning("Rate limiting requested but slowapi is not installed. Run: pip install slowapi")

_manager: Optional[MemoryManager] = None
_manager: Optional[MemoryManager] = None
_api_token = os.environ.get("AI_MEM_API_TOKEN")
MAX_LIMIT = 1000
MAX_LIMIT = 1000


def get_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager


def _check_token(request: Request, query_token: Optional[str] = None) -> None:
    if not _api_token:
        return
    auth = request.headers.get("authorization") or ""
    token = ""
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
    if not token:
        token = request.headers.get("x-ai-mem-token", "")
    if not token and query_token:
        token = query_token
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
    host: str = Field(default="generic", description="Host identifier: claude-code, gemini, cursor, etc.")
    payload: Dict[str, Any] = Field(..., description="Raw event payload from the host")
    session_id: Optional[str] = Field(default=None, description="Session ID override")
    project: Optional[str] = Field(default=None, description="Project path override")
    tags: List[str] = Field(default_factory=list, description="Additional tags")
    summarize: bool = Field(default=True, description="Enable summarization")


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
        if event.tool.output:
            output_str = json.dumps(event.tool.output) if isinstance(event.tool.output, dict) else str(event.tool.output)
            parts.append(f"Output: {output_str}")

        content = "\n".join(parts)
        if not content.strip():
            return {"status": "skipped", "reason": "empty_content"}

        # Merge tags: event tags + request tags + host
        tags = list(set(req.tags + [req.host, "tool", "auto-ingested"]))

        # Create observation with idempotency
        manager = get_manager()
        obs = await manager.add_observation(
            content=content,
            obs_type="tool_output",
            project=req.project or event.context.project,
            session_id=req.session_id or event.session_id,
            tags=tags,
            metadata={
                "tool_name": event.tool.name,
                "tool_success": event.tool.success,
                "tool_latency_ms": event.tool.latency_ms,
                "source_host": event.source.host,
                **event.metadata,
            },
            summarize=req.summarize,
            event_id=event.event_id,
            host=req.host,
        )

        if not obs:
            return {"status": "skipped", "reason": "private"}

        return {
            "status": "ok",
            "observation": obs.model_dump(),
            "event_type": "tool_use",
            "event_id": event.event_id,
        }

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


@app.get("/api/stream")
async def stream_memories(
    request: Request,
    project: Optional[str] = None,
    session_id: Optional[str] = None,
    query: Optional[str] = None,
    obs_type: Optional[str] = None,
    tags: Optional[str] = None,
    token: Optional[str] = None,
):
    _check_token(request, query_token=token)
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


def start_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)

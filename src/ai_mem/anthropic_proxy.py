import json
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from .context import build_context
from .memory import MemoryManager

DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
            elif isinstance(item, str):
                chunks.append(item)
        if chunks:
            return "".join(chunks)
    return json.dumps(content, ensure_ascii=True)


def _messages_have_context_tag(messages: List[Dict[str, Any]]) -> bool:
    for msg in messages:
        content = _stringify_content(msg.get("content"))
        if "<ai-mem-context>" in content:
            return True
    return False


def _system_has_context_tag(system: Any) -> bool:
    if system is None:
        return False
    return "<ai-mem-context>" in _stringify_content(system)


def _extract_last_user(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return _stringify_content(msg.get("content"))
    return ""


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    text = value.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_tags(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_project(request: Request, default_project: Optional[str]) -> str:
    header = request.headers.get("x-ai-mem-project")
    if header:
        return header
    return default_project or os.getcwd()


def _resolve_session_id(request: Request) -> Optional[str]:
    header = request.headers.get("x-ai-mem-session-id")
    if header:
        return header
    return None


def _context_overrides(request: Request) -> Dict[str, Any]:
    return {
        "obs_type": request.headers.get("x-ai-mem-obs-type"),
        "obs_types": _parse_tags(request.headers.get("x-ai-mem-obs-types")),
        "tag_filters": _parse_tags(request.headers.get("x-ai-mem-tags")),
        "total_count": _parse_int(request.headers.get("x-ai-mem-total")),
        "full_count": _parse_int(request.headers.get("x-ai-mem-full")),
        "full_field": request.headers.get("x-ai-mem-full-field"),
        "show_tokens": _parse_bool(request.headers.get("x-ai-mem-show-tokens")),
        "wrap": _parse_bool(request.headers.get("x-ai-mem-wrap")),
    }


def _merge_system(existing: Any, context_text: str) -> Any:
    if not context_text:
        return existing
    if existing is None:
        return context_text
    if isinstance(existing, list):
        return [{"type": "text", "text": context_text}] + existing
    existing_text = _stringify_content(existing).strip()
    if not existing_text:
        return context_text
    return f"{context_text}\n\n{existing_text}"


def _extract_assistant_text(payload: Dict[str, Any]) -> str:
    content = payload.get("content") or []
    chunks: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks)


def create_app(
    upstream_base_url: str,
    upstream_api_key: Optional[str] = None,
    inject_context: bool = True,
    store_interactions: bool = True,
    default_project: Optional[str] = None,
    summarize: bool = True,
    anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
) -> FastAPI:
    app = FastAPI(title="ai-mem Anthropic Proxy")
    app.state.manager = MemoryManager()
    app.state.upstream_base_url = upstream_base_url.rstrip("/")
    app.state.upstream_api_key = upstream_api_key
    app.state.inject_context = inject_context
    app.state.store_interactions = store_interactions
    app.state.default_project = default_project
    app.state.summarize = summarize
    app.state.anthropic_version = anthropic_version
    app.state.client = None

    @app.on_event("startup")
    async def _startup() -> None:
        app.state.client = httpx.AsyncClient(timeout=120.0)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        client = app.state.client
        if client:
            await client.aclose()

    def _build_headers(request: Request) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        api_key = app.state.upstream_api_key or request.headers.get("x-api-key")
        if api_key:
            headers["x-api-key"] = api_key
        version = request.headers.get("anthropic-version") or app.state.anthropic_version
        if version:
            headers["anthropic-version"] = version
        beta = request.headers.get("anthropic-beta")
        if beta:
            headers["anthropic-beta"] = beta
        return headers

    async def _store_pair(
        project: str,
        session_id: Optional[str],
        user_text: str,
        assistant_text: str,
    ) -> None:
        if not app.state.store_interactions:
            return
        manager: MemoryManager = app.state.manager
        if user_text.strip():
            manager.add_observation(
                content=user_text,
                obs_type="interaction",
                project=project,
                session_id=session_id,
                tags=["user"],
                metadata={"source": "anthropic-proxy"},
                summarize=app.state.summarize,
            )
        if assistant_text.strip():
            manager.add_observation(
                content=assistant_text,
                obs_type="interaction",
                project=project,
                session_id=session_id,
                tags=["assistant"],
                metadata={"source": "anthropic-proxy"},
                summarize=app.state.summarize,
            )

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        client: httpx.AsyncClient = app.state.client
        if not client:
            raise HTTPException(status_code=500, detail="Proxy not ready")

        payload = await request.json()
        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            messages = []
        system = payload.get("system")

        project = _resolve_project(request, app.state.default_project)
        session_id = _resolve_session_id(request)
        context_project = None if session_id else project
        user_text = _extract_last_user(messages)
        query_override = request.headers.get("x-ai-mem-query")
        query_text = (query_override or user_text).strip()
        if len(query_text) > 500:
            query_text = query_text[:500]

        inject_override = _parse_bool(request.headers.get("x-ai-mem-inject"))
        should_inject = app.state.inject_context if inject_override is None else inject_override
        if should_inject and not _messages_have_context_tag(messages) and not _system_has_context_tag(system):
            overrides = _context_overrides(request)
            if overrides.get("wrap") is None:
                overrides["wrap"] = True
            context_text, _ = build_context(
                app.state.manager,
                project=context_project,
                session_id=session_id,
                query=query_text or None,
                **overrides,
            )
            if context_text.strip():
                payload["system"] = _merge_system(system, context_text)

        store_override = _parse_bool(request.headers.get("x-ai-mem-store"))
        should_store = app.state.store_interactions if store_override is None else store_override

        url = f"{app.state.upstream_base_url}/v1/messages"
        headers = _build_headers(request)
        stream = bool(payload.get("stream"))

        if not stream:
            upstream = await client.post(url, headers=headers, json=payload)
            if upstream.status_code >= 400:
                return Response(
                    content=upstream.content,
                    status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type", "application/json"),
                )
            assistant_text = ""
            try:
                data = upstream.json()
                assistant_text = _extract_assistant_text(data)
            except Exception:
                assistant_text = ""
            if should_store:
                await _store_pair(project, session_id, user_text, assistant_text)
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json"),
            )

        async def _streamer() -> Any:
            assistant_chunks: List[str] = []
            async with client.stream("POST", url, headers=headers, json=payload) as upstream:
                if upstream.status_code >= 400:
                    body = await upstream.aread()
                    yield body
                    return
                async for line in upstream.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str and data_str != "[DONE]":
                            try:
                                event = json.loads(data_str)
                                delta = event.get("delta") or {}
                                text = delta.get("text")
                                if isinstance(text, str):
                                    assistant_chunks.append(text)
                                else:
                                    block = event.get("content_block") or {}
                                    text = block.get("text")
                                    if isinstance(text, str):
                                        assistant_chunks.append(text)
                            except Exception:
                                pass
                    yield (line + "\n").encode("utf-8")
            if should_store and assistant_chunks:
                await _store_pair(project, session_id, user_text, "".join(assistant_chunks))

        return StreamingResponse(_streamer(), media_type="text/event-stream")

    return app


def start_proxy(
    host: str = "0.0.0.0",
    port: int = 8095,
    upstream_base_url: Optional[str] = None,
    upstream_api_key: Optional[str] = None,
    inject_context: bool = True,
    store_interactions: bool = True,
    default_project: Optional[str] = None,
    summarize: bool = True,
    anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
) -> None:
    import uvicorn

    if not upstream_base_url:
        raise ValueError("Proxy requires upstream_base_url.")
    app = create_app(
        upstream_base_url=upstream_base_url,
        upstream_api_key=upstream_api_key,
        inject_context=inject_context,
        store_interactions=store_interactions,
        default_project=default_project,
        summarize=summarize,
        anthropic_version=anthropic_version,
    )
    uvicorn.run(app, host=host, port=port)

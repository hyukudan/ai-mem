import json
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from .context import build_context
from .memory import MemoryManager


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True)


def _extract_text_from_parts(parts: Any) -> str:
    if not isinstance(parts, list):
        return ""
    chunks: List[str] = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            chunks.append(part["text"])
    return "".join(chunks)


def _extract_last_user(contents: Any) -> str:
    if not isinstance(contents, list):
        return ""
    for item in reversed(contents):
        if not isinstance(item, dict):
            continue
        if str(item.get("role", "")).lower() == "user":
            return _extract_text_from_parts(item.get("parts"))
    return ""


def _content_has_context_tag(contents: Any, system_instruction: Any) -> bool:
    if isinstance(system_instruction, dict):
        if "<ai-mem-context>" in _extract_text_from_parts(system_instruction.get("parts")):
            return True
    if isinstance(system_instruction, str) and "<ai-mem-context>" in system_instruction:
        return True
    if not isinstance(contents, list):
        return False
    for item in contents:
        if not isinstance(item, dict):
            continue
        if "<ai-mem-context>" in _extract_text_from_parts(item.get("parts")):
            return True
    return False


def _inject_context(contents: Any, context_text: str) -> List[Dict[str, Any]]:
    if not isinstance(contents, list):
        contents = []

    for item in reversed(contents):
        if not isinstance(item, dict):
            continue
        if str(item.get("role", "")).lower() != "user":
            continue
        parts = item.get("parts")
        if not isinstance(parts, list):
            parts = []
        inserted = False
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                part["text"] = f"{context_text}\n\n{part['text']}".strip()
                inserted = True
                break
        if not inserted:
            parts.insert(0, {"text": context_text})
        item["parts"] = parts
        return contents

    contents.append({"role": "user", "parts": [{"text": context_text}]})
    return contents


def _merge_system_instruction(system_instruction: Any, context_text: str) -> Any:
    if isinstance(system_instruction, dict):
        merged = dict(system_instruction)
        parts = merged.get("parts")
        if not isinstance(parts, list):
            parts = []
        inserted = False
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                part["text"] = f"{context_text}\n\n{part['text']}".strip()
                inserted = True
                break
        if not inserted:
            parts.insert(0, {"text": context_text})
        merged["parts"] = parts
        merged.setdefault("role", "system")
        return merged
    if isinstance(system_instruction, str):
        return f"{context_text}\n\n{system_instruction}".strip()
    return {"role": "system", "parts": [{"text": context_text}]}


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


def _build_upstream_url(
    base_url: str,
    path: str,
    query: Dict[str, str],
    api_key: Optional[str],
) -> str:
    params = dict(query)
    if api_key:
        params["key"] = api_key
    query_text = httpx.QueryParams(params)
    suffix = f"?{query_text}" if query_text else ""
    return f"{base_url.rstrip('/')}{path}{suffix}"


def _build_headers(request: Request) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for key, value in request.headers.items():
        lower = key.lower()
        if lower in {"host", "content-length"}:
            continue
        if lower.startswith("x-ai-mem-"):
            continue
        headers[key] = value
    return headers


def _extract_candidate_text(payload: Dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    first = candidates[0]
    if not isinstance(first, dict):
        return ""
    content = first.get("content") or {}
    return _extract_text_from_parts(content.get("parts"))


def _parse_stream_line(line: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    raw = line.rstrip("\n")
    text = raw.strip()
    if not text:
        return raw, None
    if text.startswith("data:"):
        text = text[5:].strip()
    if not text:
        return raw, None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return raw, None
    return raw, payload


def create_app(
    upstream_base_url: str,
    upstream_api_key: Optional[str] = None,
    inject_context: bool = True,
    store_interactions: bool = True,
    default_project: Optional[str] = None,
    summarize: bool = True,
) -> FastAPI:
    app = FastAPI(title="ai-mem Gemini Proxy")
    app.state.manager = MemoryManager()
    app.state.upstream_base_url = upstream_base_url.rstrip("/")
    app.state.upstream_api_key = upstream_api_key
    app.state.inject_context = inject_context
    app.state.store_interactions = store_interactions
    app.state.default_project = default_project
    app.state.summarize = summarize
    app.state.summarize = summarize
    app.state.client = None

    _allowed_origins = os.environ.get("AI_MEM_ALLOWED_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def check_auth(request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        api_token = os.environ.get("AI_MEM_API_TOKEN")
        if api_token:
            auth = request.headers.get("authorization") or ""
            token = ""
            if auth.lower().startswith("bearer "):
                token = auth.split(" ", 1)[1].strip()
            if not token:
                token = request.headers.get("x-ai-mem-token", "")
            if token != api_token:
                return Response(content="Unauthorized", status_code=401)
        
        return await call_next(request)

    @app.on_event("startup")
    async def _startup() -> None:
        await app.state.manager.initialize()
        app.state.client = httpx.AsyncClient(timeout=120.0)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await app.state.manager.close()
        client = app.state.client
        if client:
            await client.aclose()

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
            await manager.add_observation(
                content=user_text,
                obs_type="interaction",
                project=project,
                session_id=session_id,
                tags=["user"],
                metadata={"source": "gemini-proxy"},
                summarize=app.state.summarize,
            )
        if assistant_text.strip():
            await manager.add_observation(
                content=assistant_text,
                obs_type="interaction",
                project=project,
                session_id=session_id,
                tags=["assistant"],
                metadata={"source": "gemini-proxy"},
                summarize=app.state.summarize,
            )

    def _should_inject(request: Request) -> bool:
        inject_override = _parse_bool(request.headers.get("x-ai-mem-inject"))
        return app.state.inject_context if inject_override is None else inject_override

    def _should_store(request: Request) -> bool:
        store_override = _parse_bool(request.headers.get("x-ai-mem-store"))
        return app.state.store_interactions if store_override is None else store_override

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def proxy(path: str, request: Request) -> Response:
        client: httpx.AsyncClient = app.state.client
        if not client:
            raise HTTPException(status_code=500, detail="Proxy not ready")

        upstream_url = _build_upstream_url(
            app.state.upstream_base_url,
            request.url.path,
            dict(request.query_params),
            app.state.upstream_api_key,
        )
        headers = _build_headers(request)

        if request.method != "POST":
            upstream = await client.request(
                request.method,
                upstream_url,
                headers=headers,
                content=await request.body(),
            )
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json"),
            )

        raw_body = await request.body()
        if not raw_body:
            upstream = await client.post(upstream_url, headers=headers, content=raw_body)
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json"),
            )

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            upstream = await client.post(upstream_url, headers=headers, content=raw_body)
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json"),
            )

        contents = payload.get("contents")
        system_instruction = payload.get("systemInstruction") or payload.get("system_instruction")
        project = _resolve_project(request, app.state.default_project)
        session_id = _resolve_session_id(request)
        context_project = None if session_id else project
        user_text = _extract_last_user(contents)
        query_override = request.headers.get("x-ai-mem-query")
        query_text = (query_override or user_text).strip()
        if len(query_text) > 500:
            query_text = query_text[:500]

        if _should_inject(request) and not _content_has_context_tag(contents, system_instruction):
            overrides = _context_overrides(request)
            if overrides.get("wrap") is None:
                overrides["wrap"] = True
            context_text, _ = await build_context(
                app.state.manager,
                project=context_project,
                session_id=session_id,
                query=query_text or None,
                **overrides,
            )
            if context_text.strip():
                merged_instruction = _merge_system_instruction(system_instruction, context_text)
                payload["systemInstruction"] = merged_instruction
                if "system_instruction" in payload:
                    payload["system_instruction"] = merged_instruction

        store = _should_store(request)
        is_stream = ":streamGenerateContent" in request.url.path

        if not is_stream:
            upstream = await client.post(upstream_url, headers=headers, json=payload)
            if upstream.status_code >= 400:
                return Response(
                    content=upstream.content,
                    status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type", "application/json"),
                )
            assistant_text = ""
            try:
                assistant_text = _extract_candidate_text(upstream.json())
            except Exception:
                assistant_text = ""
            if store:
                await _store_pair(project, session_id, user_text, assistant_text)
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json"),
            )

        async def _streamer() -> Any:
            assistant_chunks: List[str] = []
            async with client.stream("POST", upstream_url, headers=headers, json=payload) as upstream:
                if upstream.status_code >= 400:
                    body = await upstream.aread()
                    yield body
                    return
                async for line in upstream.aiter_lines():
                    raw, event = _parse_stream_line(line)
                    if event:
                        chunk_text = _extract_candidate_text(event)
                        if chunk_text:
                            assistant_chunks.append(chunk_text)
                    if raw is None:
                        continue
                    yield (raw + "\n").encode("utf-8")
            if store and assistant_chunks:
                await _store_pair(project, session_id, user_text, "".join(assistant_chunks))

        return StreamingResponse(_streamer(), media_type="application/json")

    return app


def start_proxy(
    host: str = "0.0.0.0",
    port: int = 8090,
    upstream_base_url: Optional[str] = None,
    upstream_api_key: Optional[str] = None,
    inject_context: bool = True,
    store_interactions: bool = True,
    default_project: Optional[str] = None,
    summarize: bool = True,
) -> None:
    import uvicorn

    base_url = upstream_base_url or os.environ.get("AI_MEM_GEMINI_UPSTREAM_BASE_URL") or "https://generativelanguage.googleapis.com"
    app = create_app(
        upstream_base_url=base_url,
        upstream_api_key=upstream_api_key or os.environ.get("AI_MEM_GEMINI_API_KEY"),
        inject_context=inject_context,
        store_interactions=store_interactions,
        default_project=default_project,
        summarize=summarize,
    )
    uvicorn.run(app, host=host, port=port)

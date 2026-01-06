import json
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from .context import build_context
from .memory import MemoryManager

DEFAULT_AZURE_API_VERSION = "2024-02-01"


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=True)


def _extract_last_user(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return _stringify_content(msg.get("content"))
    return ""


def _messages_have_context_tag(messages: List[Dict[str, Any]]) -> bool:
    for msg in messages:
        content = _stringify_content(msg.get("content"))
        if "<ai-mem-context>" in content:
            return True
    return False


def _prompt_has_context_tag(prompt: Any) -> bool:
    if isinstance(prompt, list):
        for item in prompt:
            if "<ai-mem-context>" in _stringify_content(item):
                return True
        return False
    return "<ai-mem-context>" in _stringify_content(prompt)


def _normalize_prompt(prompt: Any) -> List[str]:
    if isinstance(prompt, list):
        return [_stringify_content(item) for item in prompt]
    if prompt is None:
        return []
    return [_stringify_content(prompt)]


def _inject_prompt(prompt: Any, context_text: str) -> Any:
    prefix = f"{context_text}\n\n" if context_text else ""
    if isinstance(prompt, list):
        return [prefix + _stringify_content(item) for item in prompt]
    return prefix + _stringify_content(prompt)


def _extract_prompt_text(prompt: Any) -> str:
    prompts = _normalize_prompt(prompt)
    if prompts:
        return prompts[0]
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


def _resolve_deployment(
    request: Request, payload: Dict[str, Any], default_deployment: Optional[str]
) -> Optional[str]:
    header = request.headers.get("x-ai-mem-azure-deployment")
    if header:
        return header
    model = payload.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return default_deployment


def create_app(
    upstream_base_url: str,
    upstream_api_key: Optional[str] = None,
    deployment_name: Optional[str] = None,
    api_version: str = DEFAULT_AZURE_API_VERSION,
    inject_context: bool = True,
    store_interactions: bool = True,
    default_project: Optional[str] = None,
    summarize: bool = True,
) -> FastAPI:
    app = FastAPI(title="ai-mem Azure OpenAI Proxy")
    app.state.manager = MemoryManager()
    app.state.upstream_base_url = upstream_base_url.rstrip("/")
    app.state.upstream_api_key = upstream_api_key
    app.state.deployment_name = deployment_name
    app.state.api_version = api_version
    app.state.inject_context = inject_context
    app.state.store_interactions = store_interactions
    app.state.default_project = default_project
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

    def _build_headers(request: Request) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        api_key = app.state.upstream_api_key or request.headers.get("api-key")
        if not api_key:
            auth = request.headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                api_key = auth.split(" ", 1)[1].strip()
        if api_key:
            headers["api-key"] = api_key
        return headers

    def _deployment_or_error(payload: Dict[str, Any], request: Request) -> str:
        deployment = _resolve_deployment(request, payload, app.state.deployment_name)
        if not deployment:
            raise HTTPException(status_code=400, detail="Azure deployment name is required.")
        return deployment

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
                metadata={"source": "azure-proxy"},
                summarize=app.state.summarize,
            )
        if assistant_text.strip():
            await manager.add_observation(
                content=assistant_text,
                obs_type="interaction",
                project=project,
                session_id=session_id,
                tags=["assistant"],
                metadata={"source": "azure-proxy"},
                summarize=app.state.summarize,
            )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        client: httpx.AsyncClient = app.state.client
        if not client:
            raise HTTPException(status_code=500, detail="Proxy not ready")

        payload = await request.json()
        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            messages = []

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
        if should_inject and not _messages_have_context_tag(messages):
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
                payload["messages"] = [{"role": "system", "content": context_text}] + messages

        store_override = _parse_bool(request.headers.get("x-ai-mem-store"))
        should_store = app.state.store_interactions if store_override is None else store_override

        deployment = _deployment_or_error(payload, request)
        payload.pop("model", None)
        url = f"{app.state.upstream_base_url}/openai/deployments/{deployment}/chat/completions"
        headers = _build_headers(request)
        params = {"api-version": app.state.api_version}
        stream = bool(payload.get("stream"))

        if not stream:
            upstream = await client.post(url, headers=headers, params=params, json=payload)
            if upstream.status_code >= 400:
                return Response(
                    content=upstream.content,
                    status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type", "application/json"),
                )
            assistant_text = ""
            try:
                data = upstream.json()
                choices = data.get("choices") or []
                if choices:
                    message = choices[0].get("message") or {}
                    assistant_text = _stringify_content(message.get("content"))
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
            async with client.stream(
                "POST", url, headers=headers, params=params, json=payload
            ) as upstream:
                if upstream.status_code >= 400:
                    body = await upstream.aread()
                    yield body
                    return
                async for line in upstream.aiter_lines():
                    if line is None:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str and data_str != "[DONE]":
                            try:
                                event = json.loads(data_str)
                                choices = event.get("choices") or []
                                if choices:
                                    delta = choices[0].get("delta") or {}
                                    content_piece = delta.get("content")
                                    if isinstance(content_piece, str):
                                        assistant_chunks.append(content_piece)
                            except Exception:
                                pass
                    yield (line + "\n").encode("utf-8")
            if should_store and assistant_chunks:
                await _store_pair(project, session_id, user_text, "".join(assistant_chunks))

        return StreamingResponse(_streamer(), media_type="text/event-stream")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        client: httpx.AsyncClient = app.state.client
        if not client:
            raise HTTPException(status_code=500, detail="Proxy not ready")

        payload = await request.json()
        prompt = payload.get("prompt")

        project = _resolve_project(request, app.state.default_project)
        session_id = _resolve_session_id(request)
        context_project = None if session_id else project
        prompt_text = _extract_prompt_text(prompt)
        query_override = request.headers.get("x-ai-mem-query")
        query_text = (query_override or prompt_text).strip()
        if len(query_text) > 500:
            query_text = query_text[:500]

        inject_override = _parse_bool(request.headers.get("x-ai-mem-inject"))
        should_inject = app.state.inject_context if inject_override is None else inject_override
        if should_inject and not _prompt_has_context_tag(prompt):
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
                payload["prompt"] = _inject_prompt(prompt, context_text)

        store_override = _parse_bool(request.headers.get("x-ai-mem-store"))
        should_store = app.state.store_interactions if store_override is None else store_override

        deployment = _deployment_or_error(payload, request)
        payload.pop("model", None)
        url = f"{app.state.upstream_base_url}/openai/deployments/{deployment}/completions"
        headers = _build_headers(request)
        params = {"api-version": app.state.api_version}
        stream = bool(payload.get("stream"))

        if not stream:
            upstream = await client.post(url, headers=headers, params=params, json=payload)
            if upstream.status_code >= 400:
                return Response(
                    content=upstream.content,
                    status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type", "application/json"),
                )
            assistant_text = ""
            try:
                data = upstream.json()
                choices = data.get("choices") or []
                if choices:
                    assistant_text = _stringify_content(choices[0].get("text"))
            except Exception:
                assistant_text = ""
            if should_store:
                await _store_pair(project, session_id, prompt_text, assistant_text)
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json"),
            )

        async def _streamer() -> Any:
            assistant_chunks: List[str] = []
            async with client.stream(
                "POST", url, headers=headers, params=params, json=payload
            ) as upstream:
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
                                choices = event.get("choices") or []
                                if choices:
                                    text_piece = choices[0].get("text")
                                    if isinstance(text_piece, str):
                                        assistant_chunks.append(text_piece)
                            except Exception:
                                pass
                    yield (line + "\n").encode("utf-8")
            if should_store and assistant_chunks:
                await _store_pair(project, session_id, prompt_text, "".join(assistant_chunks))

        return StreamingResponse(_streamer(), media_type="text/event-stream")

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> Response:
        client: httpx.AsyncClient = app.state.client
        if not client:
            raise HTTPException(status_code=500, detail="Proxy not ready")

        payload = await request.json()
        deployment = _deployment_or_error(payload, request)
        payload.pop("model", None)
        url = f"{app.state.upstream_base_url}/openai/deployments/{deployment}/embeddings"
        headers = _build_headers(request)
        params = {"api-version": app.state.api_version}
        upstream = await client.post(url, headers=headers, params=params, json=payload)
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    return app


def start_proxy(
    host: str = "0.0.0.0",
    port: int = 8092,
    upstream_base_url: Optional[str] = None,
    upstream_api_key: Optional[str] = None,
    deployment_name: Optional[str] = None,
    api_version: str = DEFAULT_AZURE_API_VERSION,
    inject_context: bool = True,
    store_interactions: bool = True,
    default_project: Optional[str] = None,
    summarize: bool = True,
) -> None:
    import uvicorn

    if not upstream_base_url:
        raise ValueError("Proxy requires upstream_base_url.")
    app = create_app(
        upstream_base_url=upstream_base_url,
        upstream_api_key=upstream_api_key,
        deployment_name=deployment_name,
        api_version=api_version,
        inject_context=inject_context,
        store_interactions=store_interactions,
        default_project=default_project,
        summarize=summarize,
    )
    uvicorn.run(app, host=host, port=port)

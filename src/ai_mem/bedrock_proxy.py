import json
import os
from typing import Any, Dict, List, Optional, Tuple

import boto3
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from .context import build_context
from .memory import MemoryManager

DEFAULT_ANTHROPIC_VERSION = "bedrock-2023-05-31"
DEFAULT_MAX_TOKENS = 1024


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


def _resolve_model_id(
    request: Request, payload: Dict[str, Any], default_model: Optional[str]
) -> Optional[str]:
    header = request.headers.get("x-ai-mem-bedrock-model")
    if header:
        return header
    model = payload.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return default_model


def _resolve_generation_params(payload: Dict[str, Any]) -> Tuple[int, float]:
    max_tokens = (
        payload.get("max_tokens")
        or payload.get("maxTokens")
        or payload.get("max_tokens_to_sample")
        or payload.get("maxTokenCount")
        or payload.get("max_gen_len")
        or DEFAULT_MAX_TOKENS
    )
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens = DEFAULT_MAX_TOKENS
    temperature = payload.get("temperature", 0.2)
    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        temperature = 0.2
    return max_tokens, temperature


def _messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
    lines = []
    for msg in messages:
        role = (msg.get("role") or "user").upper()
        content = _stringify_content(msg.get("content")).strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def _build_anthropic_payload(
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    anthropic_version: str,
) -> Dict[str, Any]:
    system_parts: List[str] = []
    converted: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role") or "user"
        content = _stringify_content(msg.get("content")).strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        converted.append(
            {"role": role, "content": [{"type": "text", "text": content}]}
        )
    payload: Dict[str, Any] = {
        "anthropic_version": anthropic_version,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": converted,
    }
    system_text = "\n\n".join(system_parts).strip()
    if system_text:
        payload["system"] = system_text
    return payload


def _build_payload(
    model_id: str,
    messages: Optional[List[Dict[str, Any]]],
    prompt: Optional[str],
    max_tokens: int,
    temperature: float,
    anthropic_version: str,
) -> Dict[str, Any]:
    if model_id.startswith("anthropic."):
        return _build_anthropic_payload(messages or [], max_tokens, temperature, anthropic_version)
    if prompt is None:
        prompt = _messages_to_prompt(messages or [])
    if model_id.startswith("amazon."):
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
            },
        }
    if model_id.startswith("cohere."):
        return {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    if model_id.startswith("meta."):
        return {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
        }
    return {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}


def _extract_anthropic_text(payload: Dict[str, Any]) -> str:
    content = payload.get("content") or []
    chunks: List[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks)


def _extract_generic_text(payload: Dict[str, Any]) -> str:
    if isinstance(payload.get("completion"), str):
        return payload["completion"]
    if isinstance(payload.get("generation"), str):
        return payload["generation"]
    if isinstance(payload.get("outputText"), str):
        return payload["outputText"]
    results = payload.get("results") or []
    if results and isinstance(results, list):
        value = results[0].get("outputText") if isinstance(results[0], dict) else None
        if isinstance(value, str):
            return value
    generations = payload.get("generations") or []
    if generations and isinstance(generations, list):
        value = generations[0].get("text") if isinstance(generations[0], dict) else None
        if isinstance(value, str):
            return value
    return ""


def create_app(
    model_id: Optional[str],
    region: Optional[str],
    endpoint_url: Optional[str],
    profile: Optional[str],
    anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    inject_context: bool = True,
    store_interactions: bool = True,
    default_project: Optional[str] = None,
    summarize: bool = True,
) -> FastAPI:
    app = FastAPI(title="ai-mem Bedrock Proxy")
    app.state.manager = MemoryManager()
    app.state.model_id = model_id
    app.state.anthropic_version = anthropic_version
    app.state.max_tokens = max_tokens
    app.state.inject_context = inject_context
    app.state.store_interactions = store_interactions
    app.state.default_project = default_project
    app.state.default_project = default_project
    app.state.summarize = summarize

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

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    app.state.client = session.client(
        "bedrock-runtime",
        region_name=region,
        endpoint_url=endpoint_url,
    )

    @app.on_event("startup")
    async def _startup() -> None:
        await app.state.manager.initialize()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await app.state.manager.close()

    def _invoke(payload: Dict[str, Any], model: str) -> Dict[str, Any]:
        response = app.state.client.invoke_model(
            modelId=model,
            body=json.dumps(payload),
            accept="application/json",
            contentType="application/json",
        )
        body = response.get("body")
        raw = body.read() if hasattr(body, "read") else body
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        return json.loads(raw or "{}")

    def _invoke_stream(payload: Dict[str, Any], model: str):
        return app.state.client.invoke_model_with_response_stream(
            modelId=model,
            body=json.dumps(payload),
            accept="application/json",
            contentType="application/json",
        )

    def _require_model(payload: Dict[str, Any], request: Request) -> str:
        model = _resolve_model_id(request, payload, app.state.model_id)
        if not model:
            raise HTTPException(status_code=400, detail="Bedrock model id is required.")
        return model

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
                metadata={"source": "bedrock-proxy"},
                summarize=app.state.summarize,
            )
        if assistant_text.strip():
            await manager.add_observation(
                content=assistant_text,
                obs_type="interaction",
                project=project,
                session_id=session_id,
                tags=["assistant"],
                metadata={"source": "bedrock-proxy"},
                summarize=app.state.summarize,
            )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
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
                messages = [{"role": "system", "content": context_text}] + messages

        store_override = _parse_bool(request.headers.get("x-ai-mem-store"))
        should_store = app.state.store_interactions if store_override is None else store_override

        model = _require_model(payload, request)
        max_tokens, temperature = _resolve_generation_params(payload)
        bedrock_payload = _build_payload(
            model_id=model,
            messages=messages,
            prompt=None,
            max_tokens=max_tokens,
            temperature=temperature,
            anthropic_version=app.state.anthropic_version,
        )
        stream = bool(payload.get("stream"))
        if stream:
            if not model.startswith("anthropic."):
                raise HTTPException(status_code=400, detail="Streaming is only supported for Anthropic Bedrock models.")

            async def _streamer():
                assistant_chunks: List[str] = []
                try:
                    response = _invoke_stream(bedrock_payload, model)
                except Exception as exc:
                    yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode("utf-8")
                    return
                for event in response.get("body", []):
                    chunk = event.get("chunk", {}).get("bytes")
                    if not chunk:
                        continue
                    try:
                        data = json.loads(chunk.decode("utf-8"))
                    except Exception:
                        continue
                    text = None
                    delta = data.get("delta") or {}
                    if isinstance(delta, dict):
                        text = delta.get("text")
                    if text is None:
                        block = data.get("content_block") or {}
                        if isinstance(block, dict):
                            text = block.get("text")
                    if isinstance(text, str) and text:
                        assistant_chunks.append(text)
                        event_payload = {
                            "choices": [{"delta": {"content": text}}]
                        }
                        yield f"data: {json.dumps(event_payload)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
                if should_store and assistant_chunks:
                    await _store_pair(project, session_id, user_text, "".join(assistant_chunks))

            return StreamingResponse(_streamer(), media_type="text/event-stream")

        try:
            data = _invoke(bedrock_payload, model)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        assistant_text = (
            _extract_anthropic_text(data) if model.startswith("anthropic.") else _extract_generic_text(data)
        )
        if should_store:
            await _store_pair(project, session_id, user_text, assistant_text)
        response_payload = {
            "choices": [{"message": {"role": "assistant", "content": assistant_text}}],
            "model": model,
        }
        return Response(
            content=json.dumps(response_payload, ensure_ascii=True),
            media_type="application/json",
        )

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        payload = await request.json()
        prompt = payload.get("prompt")
        prompt_text = _extract_prompt_text(prompt)

        project = _resolve_project(request, app.state.default_project)
        session_id = _resolve_session_id(request)
        context_project = None if session_id else project
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
                prompt = _inject_prompt(prompt, context_text)
                prompt_text = _extract_prompt_text(prompt)

        store_override = _parse_bool(request.headers.get("x-ai-mem-store"))
        should_store = app.state.store_interactions if store_override is None else store_override

        model = _require_model(payload, request)
        max_tokens, temperature = _resolve_generation_params(payload)
        if model.startswith("anthropic."):
            messages = [{"role": "user", "content": prompt_text}]
            bedrock_payload = _build_payload(
                model_id=model,
                messages=messages,
                prompt=None,
                max_tokens=max_tokens,
                temperature=temperature,
                anthropic_version=app.state.anthropic_version,
            )
        else:
            bedrock_payload = _build_payload(
                model_id=model,
                messages=None,
                prompt=prompt_text,
                max_tokens=max_tokens,
                temperature=temperature,
                anthropic_version=app.state.anthropic_version,
            )
        stream = bool(payload.get("stream"))
        if stream:
            if not model.startswith("anthropic."):
                raise HTTPException(status_code=400, detail="Streaming is only supported for Anthropic Bedrock models.")

            async def _streamer():
                assistant_chunks: List[str] = []
                try:
                    response = _invoke_stream(bedrock_payload, model)
                except Exception as exc:
                    yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode("utf-8")
                    return
                for event in response.get("body", []):
                    chunk = event.get("chunk", {}).get("bytes")
                    if not chunk:
                        continue
                    try:
                        data = json.loads(chunk.decode("utf-8"))
                    except Exception:
                        continue
                    text = None
                    delta = data.get("delta") or {}
                    if isinstance(delta, dict):
                        text = delta.get("text")
                    if text is None:
                        block = data.get("content_block") or {}
                        if isinstance(block, dict):
                            text = block.get("text")
                    if isinstance(text, str) and text:
                        assistant_chunks.append(text)
                        event_payload = {
                            "choices": [{"text": text}]
                        }
                        yield f"data: {json.dumps(event_payload)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
                if should_store and assistant_chunks:
                    await _store_pair(project, session_id, prompt_text, "".join(assistant_chunks))

            return StreamingResponse(_streamer(), media_type="text/event-stream")

        try:
            data = _invoke(bedrock_payload, model)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        assistant_text = (
            _extract_anthropic_text(data) if model.startswith("anthropic.") else _extract_generic_text(data)
        )
        if should_store:
            await _store_pair(project, session_id, prompt_text, assistant_text)
        response_payload = {
            "choices": [{"text": assistant_text}],
            "model": model,
        }
        return Response(
            content=json.dumps(response_payload, ensure_ascii=True),
            media_type="application/json",
        )

    return app


def start_proxy(
    host: str = "0.0.0.0",
    port: int = 8094,
    model_id: Optional[str] = None,
    region: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    profile: Optional[str] = None,
    anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    inject_context: bool = True,
    store_interactions: bool = True,
    default_project: Optional[str] = None,
    summarize: bool = True,
) -> None:
    import uvicorn

    app = create_app(
        model_id=model_id,
        region=region,
        endpoint_url=endpoint_url,
        profile=profile,
        anthropic_version=anthropic_version,
        max_tokens=max_tokens,
        inject_context=inject_context,
        store_interactions=store_interactions,
        default_project=default_project,
        summarize=summarize,
    )
    uvicorn.run(app, host=host, port=port)

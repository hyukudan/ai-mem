import hashlib
import json
import math
import os
import re
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .chunking import chunk_text
from .config import AppConfig, load_config, resolve_storage_paths
from .db import DatabaseManager
from .embeddings.base import EmbeddingProvider
from .models import Observation, ObservationAsset, ObservationIndex, Session
from .providers.base import ChatMessage, ChatProvider, NoOpChatProvider
from .privacy import strip_memory_tags
from .vector_store import build_vector_store


def _build_chat_provider(config: AppConfig) -> ChatProvider:
    provider = config.llm.provider.lower()
    if provider in {"none", "noop"}:
        return NoOpChatProvider()
    if provider == "gemini":
        if not config.llm.api_key:
            raise ValueError("Gemini provider requires an API key.")
        from .providers.gemini import GeminiProvider

        return GeminiProvider(
            api_key=config.llm.api_key,
            model_name=config.llm.model,
        )
    if provider == "anthropic":
        if not config.llm.api_key:
            raise ValueError("Anthropic provider requires an API key.")
        base_url = config.llm.base_url or "https://api.anthropic.com"
        model_name = config.llm.model
        if not model_name or model_name == "local-model":
            model_name = "claude-3-haiku-20240307"
        from .providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            api_key=config.llm.api_key,
            model_name=model_name,
            base_url=base_url,
            timeout_s=config.llm.timeout_s,
        )
    if provider == "bedrock":
        model_id = config.llm.model
        if not model_id or model_id == "local-model":
            raise ValueError("Bedrock provider requires a model id (llm model).")
        region = (
            os.environ.get("AI_MEM_BEDROCK_REGION")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
        )
        endpoint = config.llm.base_url or os.environ.get("AI_MEM_BEDROCK_ENDPOINT")
        profile = os.environ.get("AI_MEM_BEDROCK_PROFILE")
        max_tokens = int(os.environ.get("AI_MEM_BEDROCK_MAX_TOKENS", "1024"))
        anthropic_version = os.environ.get("AI_MEM_BEDROCK_ANTHROPIC_VERSION", "bedrock-2023-05-31")
        from .providers.bedrock import BedrockProvider

        return BedrockProvider(
            model_id=model_id,
            region=region,
            endpoint_url=endpoint,
            profile=profile,
            max_tokens=max_tokens,
            anthropic_version=anthropic_version,
        )
    if provider in {"azure", "azure-openai"}:
        api_key = config.llm.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Azure OpenAI provider requires an API key.")
        endpoint = config.llm.base_url or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError("Azure OpenAI provider requires a base_url or AZURE_OPENAI_ENDPOINT.")
        deployment = config.llm.model
        if not deployment or deployment == "local-model":
            raise ValueError("Azure OpenAI provider requires a deployment name (llm model).")
        api_version = (
            config.llm.api_version
            or os.environ.get("AI_MEM_AZURE_API_VERSION")
            or os.environ.get("AZURE_OPENAI_API_VERSION")
            or "2024-02-01"
        )
        from .providers.azure_openai import AzureOpenAIProvider

        return AzureOpenAIProvider(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            deployment_name=deployment,
            timeout_s=config.llm.timeout_s,
        )
    if provider in {"openai", "openai-compatible", "vllm", "local"}:
        base_url = config.llm.base_url or "http://localhost:8000/v1"
        model_name = config.llm.model or "local-model"
        from .providers.openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            base_url=base_url,
            api_key=config.llm.api_key,
            model_name=model_name,
            timeout_s=config.llm.timeout_s,
        )
    raise ValueError(f"Unsupported LLM provider: {config.llm.provider}")


def _build_embedding_provider(config: AppConfig) -> EmbeddingProvider:
    provider = config.embeddings.provider.lower()
    if provider in {"auto", "default"}:
        base_url = config.embeddings.base_url or config.llm.base_url
        model_name = config.embeddings.model
        if base_url and model_name:
            from .embeddings.openai_compatible import OpenAIEmbeddingProvider

            return OpenAIEmbeddingProvider(
                base_url=base_url,
                api_key=config.embeddings.api_key or config.llm.api_key,
                model_name=model_name,
            )
        from .embeddings.fastembed import FastEmbedProvider

        return FastEmbedProvider(model_name=config.embeddings.model)
    if provider == "fastembed":
        from .embeddings.fastembed import FastEmbedProvider

        return FastEmbedProvider(model_name=config.embeddings.model)
    if provider == "bedrock":
        model_id = config.embeddings.model
        if not model_id or model_id == "local-embedding":
            raise ValueError("Bedrock embeddings require a model id.")
        region = (
            os.environ.get("AI_MEM_BEDROCK_REGION")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
        )
        endpoint = config.embeddings.base_url or os.environ.get("AI_MEM_BEDROCK_ENDPOINT")
        profile = os.environ.get("AI_MEM_BEDROCK_PROFILE")
        input_type = os.environ.get("AI_MEM_BEDROCK_EMBED_INPUT_TYPE")
        from .embeddings.bedrock import BedrockEmbeddingProvider

        return BedrockEmbeddingProvider(
            model_id=model_id,
            region=region,
            endpoint_url=endpoint,
            profile=profile,
            input_type=input_type,
        )
    if provider in {"azure", "azure-openai"}:
        api_key = config.embeddings.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Azure OpenAI embeddings require an API key.")
        endpoint = config.embeddings.base_url or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError("Azure OpenAI embeddings require a base_url or AZURE_OPENAI_ENDPOINT.")
        deployment = config.embeddings.model
        if not deployment or deployment == "local-embedding":
            raise ValueError("Azure OpenAI embeddings require a deployment name (embeddings model).")
        api_version = (
            config.embeddings.api_version
            or os.environ.get("AI_MEM_AZURE_API_VERSION")
            or os.environ.get("AZURE_OPENAI_API_VERSION")
            or "2024-02-01"
        )
        from .embeddings.azure_openai import AzureOpenAIEmbeddingProvider

        return AzureOpenAIEmbeddingProvider(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            deployment_name=deployment,
        )
    if provider in {"openai", "openai-compatible", "vllm", "local"}:
        base_url = config.embeddings.base_url or "http://localhost:8000/v1"
        model_name = config.embeddings.model or "local-embedding"
        from .embeddings.openai_compatible import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            base_url=base_url,
            api_key=config.embeddings.api_key,
            model_name=model_name,
        )
    if provider == "gemini":
        if not config.embeddings.api_key:
            raise ValueError("Gemini embeddings require an API key.")
        model_name = config.embeddings.model or "models/text-embedding-004"
        from .embeddings.gemini import GeminiEmbeddingProvider

        return GeminiEmbeddingProvider(
            api_key=config.embeddings.api_key,
            model_name=model_name,
        )
    raise ValueError(f"Unsupported embedding provider: {config.embeddings.provider}")


_RELATIVE_DATE_RE = re.compile(r"^(\d+(?:\.\d+)?)([smhdw])$", re.IGNORECASE)


def _parse_date(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    match = _RELATIVE_DATE_RE.match(text)
    if match:
        amount = float(match.group(1))
        unit = match.group(2).lower()
        multipliers = {
            "s": 1,
            "m": 60,
            "h": 3600,
            "d": 86400,
            "w": 604800,
        }
        return time.time() - amount * multipliers[unit]
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _normalize_tags(tags: Optional[List[str]]) -> List[str]:
    if not tags:
        return []
    cleaned = []
    for tag in tags:
        value = str(tag).strip()
        if value:
            cleaned.append(value)
    return cleaned


def _match_tags(obs_tags: List[str], tag_filters: List[str]) -> bool:
    if not tag_filters:
        return True
    tag_set = set(obs_tags or [])
    return any(tag in tag_set for tag in tag_filters)


class MemoryManager:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        storage = resolve_storage_paths(self.config)
        os.makedirs(storage.data_dir, exist_ok=True)
        os.makedirs(storage.vector_dir, exist_ok=True)
        self.db = DatabaseManager(storage.sqlite_path)
        self.vector_store = build_vector_store(self.config, storage)
        self.chat_provider = _build_chat_provider(self.config)
        self.embedding_provider = _build_embedding_provider(self.config)
        self.allow_llm_summaries = self.chat_provider.get_name() != "none"
        self.current_session: Optional[Session] = None
        self._listeners: List[Callable[[Observation], None]] = []
        self._listener_lock = threading.Lock()
        self._search_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
        self._last_search_cache_hit: Optional[bool] = None
        self._search_cache_hits: int = 0
        self._search_cache_misses: int = 0

    async def initialize(self) -> None:
        await self.db.connect()
        await self.db.create_tables()

    async def close(self) -> None:
        await self.db.close()

    def add_listener(self, listener: Callable[[Observation], None]) -> Callable[[], None]:
        with self._listener_lock:
            self._listeners.append(listener)

        def _remove() -> None:
            with self._listener_lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return _remove

    def _notify_listeners(self, obs: Observation) -> None:
        with self._listener_lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(obs)
            except Exception:
                continue

    @property
    def search_cache_hit(self) -> Optional[bool]:
        return self._last_search_cache_hit

    def _is_search_cache_enabled(self) -> bool:
        return (
            self.config.search.cache_ttl_seconds > 0
            and self.config.search.cache_max_entries > 0
        )

    def _prune_search_cache(self) -> None:
        if not self._is_search_cache_enabled():
            return
        now = time.time()
        expired = [
            key
            for key, (ts, _) in self._search_cache.items()
            if now - ts > self.config.search.cache_ttl_seconds
        ]
        for key in expired:
            self._search_cache.pop(key, None)
        while len(self._search_cache) > self.config.search.cache_max_entries:
            oldest_key = min(self._search_cache.items(), key=lambda item: item[1][0])[0]
            self._search_cache.pop(oldest_key, None)

    def _build_search_cache_key(
        self,
        query: str,
        limit: int,
        project: Optional[str],
        obs_type: Optional[str],
        session_id: Optional[str],
        date_start: Optional[str],
        date_end: Optional[str],
        tags: List[str],
    ) -> str:
        key_data = {
            "query": query.strip(),
            "limit": limit,
            "project": project or "",
            "obs_type": obs_type or "",
            "session_id": session_id or "",
            "date_start": date_start or "",
            "date_end": date_end or "",
            "tags": sorted(tags),
        }
        return json.dumps(key_data, sort_keys=True)

    def _get_search_cache(self, key: str) -> Optional[List[ObservationIndex]]:
        if not self._is_search_cache_enabled():
            return None
        entry = self._search_cache.get(key)
        if not entry:
            return None
        timestamp, data = entry
        if time.time() - timestamp > self.config.search.cache_ttl_seconds:
            self._search_cache.pop(key, None)
            return None
        self._record_cache_hit()
        return [ObservationIndex.model_validate(item) for item in data]

    def _set_search_cache(self, key: str, results: List[ObservationIndex]) -> None:
        if not key or not self._is_search_cache_enabled():
            return
        payload = [item.model_dump() for item in results]
        self._search_cache[key] = (time.time(), payload)

    def _combine_scores(self, fts_score: float, vector_score: float) -> float:
        total_weight = self.config.search.fts_weight + self.config.search.vector_weight
        if total_weight <= 0:
            return max(fts_score, vector_score)
        fts_component = (fts_score or 0.0) * self.config.search.fts_weight
        vector_component = (vector_score or 0.0) * self.config.search.vector_weight
        return (fts_component + vector_component) / total_weight

    def _compute_recency_factor(self, created_at: Optional[float]) -> float:
        half_life_hours = self.config.search.recency_half_life_hours
        if half_life_hours <= 0 or not created_at:
            return 1.0
        age = max(0.0, time.time() - created_at)
        half_life_seconds = half_life_hours * 3600
        decay = math.exp(-age / half_life_seconds) if half_life_seconds > 0 else 0.0
        return 1.0 + self.config.search.recency_weight * decay

    def _record_cache_hit(self) -> None:
        self._search_cache_hits += 1

    def _record_cache_miss(self) -> None:
        self._search_cache_misses += 1

    def search_cache_summary(self) -> Dict[str, Any]:
        return {
            "enabled": self._is_search_cache_enabled(),
            "ttl_seconds": self.config.search.cache_ttl_seconds,
            "max_entries": self.config.search.cache_max_entries,
            "entries": len(self._search_cache),
            "hits": self._search_cache_hits,
            "misses": self._search_cache_misses,
            "fts_weight": self.config.search.fts_weight,
            "vector_weight": self.config.search.vector_weight,
            "recency_half_life_hours": self.config.search.recency_half_life_hours,
            "recency_weight": self.config.search.recency_weight,
        }

    async def start_session(
        self,
        project: str,
        goal: str = "",
        session_id: Optional[str] = None,
    ) -> Session:
        if self.current_session:
            await self.close_session()
        if session_id:
            existing = await self.db.get_session(session_id)
            if existing:
                session = Session.model_validate(existing)
                if goal and goal != session.goal:
                    session.goal = goal
                    await self.db.add_session(session)
                self.current_session = session
                return session
            self.current_session = Session(id=session_id, project=project, goal=goal)
        else:
            self.current_session = Session(project=project, goal=goal)
        await self.db.add_session(self.current_session)
        return self.current_session

    async def _ensure_session(self, project: str) -> Session:
        if self.current_session:
            if self.current_session.project == project:
                return self.current_session
            await self.close_session()
        existing = await self.db.list_sessions(project=project, active_only=True, limit=1)
        if existing:
            self.current_session = Session.model_validate(existing[0])
            return self.current_session
        return await self.start_session(project=project, goal="Auto-started session")

    async def close_session(self) -> None:
        if not self.current_session:
            return
        import time

        self.current_session.end_time = time.time()
        await self.db.add_session(self.current_session)
        self.current_session = None

    async def end_session(self, session_id: Optional[str] = None) -> Optional[Session]:
        if session_id:
            data = await self.db.get_session(session_id)
            if not data:
                return None
            session = Session.model_validate(data)
        else:
            session = self.current_session
            if not session:
                return None
        session.end_time = time.time()
        await self.db.add_session(session)
        if self.current_session and self.current_session.id == session.id:
            self.current_session = None
        return session

    async def end_latest_session(self, project: Optional[str] = None) -> Optional[Session]:
        project_name = project or (self.current_session.project if self.current_session else None)
        if not project_name:
            return None
        sessions = await self.db.list_sessions(project=project_name, active_only=True, limit=1)
        if not sessions:
            return None
        session = Session.model_validate(sessions[0])
        session.end_time = time.time()
        await self.db.add_session(session)
        if self.current_session and self.current_session.id == session.id:
            self.current_session = None
        return session

    async def add_observation(
        self,
        content: str,
        obs_type: str,
        project: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, object]] = None,
        title: Optional[str] = None,
        summarize: bool = True,
        dedupe: bool = True,
        summary: Optional[str] = None,
        created_at: Optional[float] = None,
        importance_score: float = 0.5,
        assets: Optional[List[Dict[str, Any]]] = None,
        diff: Optional[str] = None,
    ) -> Optional[Observation]:
        session = None
        if session_id:
            session_data = await self.db.get_session(session_id)
            if session_data:
                session = Session.model_validate(session_data)
        if session:
            project_name = session.project
        else:
            project_name = project or (os.getcwd() if session_id else (self.current_session.project if self.current_session else os.getcwd()))
            if session_id:
                session = Session(id=session_id, project=project_name)
                await self.db.add_session(session)
        cleaned_content, stripped = strip_memory_tags(content)
        if not cleaned_content:
            return None

        metadata = dict(metadata or {})
        if stripped:
            metadata["redacted"] = True

        content_hash = hashlib.sha256(cleaned_content.encode("utf-8")).hexdigest()
        if dedupe:
            existing = await self.db.find_observation_by_hash(content_hash, project_name)
            if existing:
                return Observation.model_validate(existing)

        if not session_id:
            if not self.current_session:
                await self._ensure_session(project_name)
            session = self.current_session

        summary_text = summary if summary is not None else cleaned_content
        if summary is None and len(cleaned_content) > 500:
            if summarize and self.allow_llm_summaries:
                try:
                    # TODO: Make chat_provider async in future refactor
                    summary_text = await asyncio.to_thread(self.chat_provider.summarize, cleaned_content)
                except Exception:
                    summary_text = cleaned_content[:500] + "..."
            else:
                summary_text = cleaned_content[:500] + "..."

        if not session:
            return None

        obs = Observation(
            session_id=session.id,
            project=project_name,
            type=obs_type,
            title=title,
            content=cleaned_content,
            summary=summary_text,
            created_at=created_at or time.time(),
            importance_score=importance_score,
            tags=tags or [],
            metadata=metadata,
            content_hash=content_hash,
            diff=diff,
        )
        normalized_assets = self._normalize_assets(assets or [])
        if normalized_assets:
            obs.assets = [
                ObservationAsset(
                    observation_id=obs.id,
                    type=asset["type"],
                    name=asset.get("name"),
                    path=asset.get("path"),
                    content=asset.get("content"),
                    metadata=asset.get("metadata") or {},
                    created_at=asset.get("created_at") or time.time(),
                )
                for asset in normalized_assets
            ]
        await self.db.add_observation(obs)
        if obs.assets:
            await self._store_assets(obs.id, obs.assets)
        self._index_observation(obs)
        self._notify_listeners(obs)
        return obs

    def _normalize_assets(self, assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for entry in assets:
            if not entry or not isinstance(entry, dict):
                continue
            asset_type = (entry.get("type") or entry.get("asset_type") or "file").lower()
            name = entry.get("name") or entry.get("filename")
            path = entry.get("path")
            content = entry.get("content")
            metadata = entry.get("metadata") or {}
            created = entry.get("created_at")
            normalized.append(
                {
                    "type": asset_type,
                    "name": name,
                    "path": path,
                    "content": content,
                    "metadata": metadata,
                    "created_at": created,
                }
            )
        return normalized

    async def _store_assets(self, observation_id: str, assets: List[ObservationAsset]) -> None:
        for asset in assets:
            await self.db.add_asset(
                observation_id=observation_id,
                asset_type=asset.type,
                name=asset.name,
                path=asset.path,
                content=asset.content,
                metadata=asset.metadata,
                asset_id=asset.id,
                created_at=asset.created_at,
            )

    def _index_observation(self, obs: Observation) -> None:
        chunks = chunk_text(
            obs.content,
            chunk_size=self.config.search.chunk_size,
            chunk_overlap=self.config.search.chunk_overlap,
        )
        if not chunks:
            return
        embeddings = self.embedding_provider.embed(chunks)
        ids = [f"{obs.id}:{idx}" for idx in range(len(chunks))]
        metadatas = [
            {
                "observation_id": obs.id,
                "project": obs.project,
                "session_id": obs.session_id,
                "type": obs.type,
                "created_at": obs.created_at,
                "chunk_index": idx,
            }
            for idx in range(len(chunks))
        ]
        self.vector_store.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids,
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        since: Optional[str] = None,
        tag_filters: Optional[List[str]] = None,
    ) -> List[ObservationIndex]:
        if date_start is None and since is not None:
            date_start = since
        start_ts = _parse_date(date_start)
        end_ts = _parse_date(date_end)
        tags = _normalize_tags(tag_filters)
        query_str = (query or "").strip()
        if not query_str:
            return await self.db.get_recent_observations(
                project,
                limit,
                obs_type=obs_type,
                session_id=session_id,
                tag_filters=tags,
                date_start=start_ts,
                date_end=end_ts,
            )

        cache_key = None
        cache_enabled = self._is_search_cache_enabled()
        self._last_search_cache_hit = False
        if cache_enabled:
            cache_key = self._build_search_cache_key(
                query_str,
                limit,
                project,
                obs_type,
                session_id,
                date_start,
                date_end,
                tags,
            )
            self._prune_search_cache()
            cached = self._get_search_cache(cache_key)
            if cached is not None:
                self._last_search_cache_hit = True
                return cached

        fts_results = await self.db.search_observations_fts(
            query_str,
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=start_ts,
            date_end=end_ts,
            tag_filters=tags,
            limit=self.config.search.fts_top_k,
        )
        vector_hits = await self._vector_search(
            query_str,
            project=project,
            session_id=session_id,
            date_start=start_ts,
            date_end=end_ts,
            limit=self.config.search.vector_top_k,
            tag_filters=tags,
        )

        obs_map: Dict[str, ObservationIndex] = {}
        for item in fts_results:
            item.fts_score = item.score
            item.vector_score = None
            obs_map[item.id] = item

        for obs_id, score in vector_hits.items():
            if obs_id in obs_map:
                existing = obs_map[obs_id]
                existing.vector_score = max(existing.vector_score or 0.0, score)
            else:
                obs = await self.db.get_observation(obs_id)
                if obs:
                    obs_map[obs_id] = ObservationIndex(
                        id=obs["id"],
                        summary=obs["summary"] or "",
                        project=obs["project"],
                        type=obs["type"],
                        created_at=obs["created_at"],
                        fts_score=None,
                        vector_score=score,
                    )

        combined_results: List[ObservationIndex] = []
        for item in obs_map.values():
            fts_score = item.fts_score or 0.0
            vector_score = item.vector_score or 0.0
            combined = self._combine_scores(fts_score, vector_score)
            recency_factor = self._compute_recency_factor(item.created_at)
            item.recency_factor = recency_factor
            item.score = combined * recency_factor
            combined_results.append(item)

        sorted_results = sorted(combined_results, key=lambda item: item.score, reverse=True)
        filtered = sorted_results
        if obs_type:
            filtered = [item for item in filtered if item.type == obs_type]
        final_results = filtered[:limit]
        if cache_key:
            self._record_cache_miss()
            self._set_search_cache(cache_key, final_results)
        return final_results

    async def _vector_search(
        self,
        query: str,
        project: Optional[str],
        session_id: Optional[str],
        date_start: Optional[float],
        date_end: Optional[float],
        limit: int,
        tag_filters: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        embedding = self.embedding_provider.embed([query])[0]
        where = {"project": project} if project else None
        results = self.vector_store.query(
            embedding=embedding,
            n_results=limit,
            where=where,
        )
        scores: Dict[str, float] = {}
        tags = _normalize_tags(tag_filters)
        if results.get("metadatas"):
            for idx, meta in enumerate(results["metadatas"][0]):
                obs_id = meta.get("observation_id")
                if not obs_id:
                    continue
                created_at = meta.get("created_at")
                if date_start is not None and created_at is not None and float(created_at) < date_start:
                    continue
                if date_end is not None and created_at is not None and float(created_at) > date_end:
                    continue
                obs = None
                if tags:
                    obs = await self.db.get_observation(obs_id)
                    if not obs:
                        continue
                    if not _match_tags(obs.get("tags") or [], tags):
                        continue
                if session_id:
                    if obs is None:
                        meta_session = meta.get("session_id")
                        if meta_session:
                            if meta_session != session_id:
                                continue
                        else:
                            obs = await self.db.get_observation(obs_id)
                            if not obs or obs.get("session_id") != session_id:
                                continue
                    elif obs.get("session_id") != session_id:
                        continue
                distance = 0.0
                if results.get("distances"):
                    distance = float(results["distances"][0][idx])
                score = 1.0 - distance
                scores[obs_id] = max(scores.get(obs_id, 0.0), score)
        return scores

    async def timeline(
        self,
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
        tag_filters: Optional[List[str]] = None,
    ) -> List[ObservationIndex]:
        tags = _normalize_tags(tag_filters)
        scoreboard_map: Dict[str, Dict[str, Optional[float]]] = {}
        if date_start is None and since is not None:
            date_start = since
        if not anchor_id and query:
            search_results = await self.search(
                query,
                limit=max(1, depth_before + depth_after + 5),
                project=project,
                obs_type=obs_type,
                session_id=session_id,
                date_start=date_start,
                date_end=date_end,
                since=since,
                tag_filters=tags,
            )
            if search_results:
                anchor_id = search_results[0].id
                scoreboard_map = {
                    item.id: {
                        "fts_score": item.fts_score,
                        "vector_score": item.vector_score,
                        "recency_factor": item.recency_factor,
                    }
                    for item in search_results
                }
        if not anchor_id:
            return []

        anchor = await self.db.get_observation(anchor_id)
        if not anchor:
            return []
        if tags and not _match_tags(anchor.get("tags") or [], tags):
            return []
        if session_id and anchor.get("session_id") != session_id:
            return []

        start_ts = _parse_date(date_start)
        end_ts = _parse_date(date_end)
        project_name = project or anchor["project"]
        before = await self.db.get_observations_before(
            project=project_name,
            anchor_time=anchor["created_at"],
            limit=depth_before,
            obs_type=obs_type,
            session_id=session_id,
            tag_filters=tags,
            date_start=start_ts,
            date_end=end_ts,
        )
        after = await self.db.get_observations_after(
            project=project_name,
            anchor_time=anchor["created_at"],
            limit=depth_after,
            obs_type=obs_type,
            session_id=session_id,
            tag_filters=tags,
            date_start=start_ts,
            date_end=end_ts,
        )
        def _build_index(entry: Dict[str, Any]) -> ObservationIndex:
            score_data = scoreboard_map.get(entry["id"], {})
            return ObservationIndex(
                id=entry["id"],
                summary=entry.get("summary") or "",
                project=entry.get("project"),
                type=entry.get("type"),
                created_at=entry.get("created_at"),
                score=0.0,
                fts_score=score_data.get("fts_score"),
                vector_score=score_data.get("vector_score"),
                recency_factor=score_data.get("recency_factor"),
            )

        def _apply_score(index: ObservationIndex) -> ObservationIndex:
            score_data = scoreboard_map.get(index.id, {})
            index.fts_score = score_data.get("fts_score")
            index.vector_score = score_data.get("vector_score")
            index.recency_factor = score_data.get("recency_factor")
            return index

        before = [_apply_score(item) for item in before]
        after = [_apply_score(item) for item in after]
        anchor_index = _apply_score(_build_index(anchor))
        timeline = list(reversed(before)) + [anchor_index] + after
        return timeline

    async def get_observations(self, ids: List[str]) -> List[Dict[str, object]]:
        return await self.db.get_observations(ids)

    async def list_projects(self) -> List[str]:
        return await self.db.list_projects()

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return await self.db.get_session(session_id)

    async def list_sessions(
        self,
        project: Optional[str] = None,
        active_only: bool = False,
        goal_query: Optional[str] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        start_ts = _parse_date(date_start)
        end_ts = _parse_date(date_end)
        return await self.db.list_sessions(
            project=project,
            active_only=active_only,
            goal_query=goal_query,
            date_start=start_ts,
            date_end=end_ts,
            limit=limit,
        )

    async def get_stats(
        self,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        since: Optional[str] = None,
        tag_filters: Optional[List[str]] = None,
        tag_limit: int = 10,
        day_limit: int = 14,
        type_tag_limit: int = 3,
    ) -> Dict[str, Any]:
        if date_start is None and since is not None:
            date_start = since
        start_ts = _parse_date(date_start)
        end_ts = _parse_date(date_end)
        return await self.db.get_stats(
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=start_ts,
            date_end=end_ts,
            tag_filters=_normalize_tags(tag_filters),
            tag_limit=tag_limit,
            day_limit=day_limit,
            type_tag_limit=type_tag_limit,
        )

    async def list_tags(
        self,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        tag_filters: Optional[List[str]] = None,
        limit: Optional[int] = 50,
    ) -> List[Dict[str, Any]]:
        start_ts = _parse_date(date_start)
        end_ts = _parse_date(date_end)
        return await self.db.get_tag_counts(
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=start_ts,
            date_end=end_ts,
            tag_filters=_normalize_tags(tag_filters),
            limit=limit,
        )

    async def rename_tag(
        self,
        old_tag: str,
        new_tag: str,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        tag_filters: Optional[List[str]] = None,
    ) -> int:
        start_ts = _parse_date(date_start)
        end_ts = _parse_date(date_end)
        return await self.db.replace_tag(
            old_tag=old_tag,
            new_tag=new_tag,
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=start_ts,
            date_end=end_ts,
            tag_filters=_normalize_tags(tag_filters),
        )

    async def delete_tag(
        self,
        tag: str,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        tag_filters: Optional[List[str]] = None,
    ) -> int:
        start_ts = _parse_date(date_start)
        end_ts = _parse_date(date_end)
        return await self.db.replace_tag(
            old_tag=tag,
            new_tag=None,
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=start_ts,
            date_end=end_ts,
            tag_filters=_normalize_tags(tag_filters),
        )

    async def add_tag(
        self,
        tag: str,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        tag_filters: Optional[List[str]] = None,
    ) -> int:
        start_ts = _parse_date(date_start)
        end_ts = _parse_date(date_end)
        return await self.db.add_tag(
            tag=tag,
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=start_ts,
            date_end=end_ts,
            tag_filters=_normalize_tags(tag_filters),
        )

    async def export_observations(
        self,
        project: Optional[str] = None,
        session_id: Optional[str] = None,
        obs_type: Optional[str] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        tag_filters: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        start_ts = _parse_date(date_start)
        end_ts = _parse_date(date_end)
        results = await self.db.list_observations(
            project=project,
            session_id=session_id,
            obs_type=obs_type,
            date_start=start_ts,
            date_end=end_ts,
            tag_filters=_normalize_tags(tag_filters),
            limit=limit,
        )
        return [obs.model_dump() for obs in results]

    async def summarize_project(
        self,
        project: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 20,
        obs_type: Optional[str] = None,
        store: bool = True,
        tags: Optional[List[str]] = None,
    ) -> Optional[Dict[str, object]]:
        session = None
        if session_id:
            session_data = await self.db.get_session(session_id)
            if not session_data:
                return None
            session = Session.model_validate(session_data)
        project_name = (
            session.project
            if session
            else project or (self.current_session.project if self.current_session else os.getcwd())
        )
        observations = await self.db.list_observations(
            project=project_name if not session_id else None,
            session_id=session_id,
            limit=limit,
        )
        if obs_type:
            observations = [obs for obs in observations if obs.type == obs_type]
        if not observations:
            return None

        observations = list(reversed(observations))
        lines = []
        for obs in observations:
            summary = obs.summary or ""
            summary = summary.strip()
            if summary:
                lines.append(f"- {summary}")
        source_text = "\n".join(lines)

        summary_text = ""
        if self.allow_llm_summaries:
            prompt = (
                "Summarize the following observations into a concise session summary. "
                "Focus on decisions, changes, and next steps. Keep it factual.\n\n"
                f"Observations:\n{source_text}"
            )
            try:
                # TODO: Make chat_provider async in future refactor
                summary_text = await asyncio.to_thread(
                    self.chat_provider.chat,
                    [ChatMessage(role="user", content=prompt)],
                    temperature=0.2,
                )
            except Exception:
                summary_text = ""

        if not summary_text.strip():
            summary_text = source_text
            if len(summary_text) > 1000:
                summary_text = summary_text[:1000] + "..."

        metadata = {
            "source_count": len(observations),
            "source_ids": [obs.id for obs in observations if obs.id],
        }
        if session_id:
            metadata["session_id"] = session_id

        if store:
            summary_tags = list(tags or [])
            if session_id and "session-summary" not in summary_tags:
                summary_tags.append("session-summary")
            obs = await self.add_observation(
                content=summary_text,
                obs_type="summary",
                project=project_name,
                session_id=session_id,
                summarize=False,
                summary=summary_text,
                metadata=metadata,
                tags=summary_tags,
            )
            if not obs:
                return None
            if session_id and session:
                session.summary = summary_text
                await self.db.add_session(session)
            return {"summary": summary_text, "observation": obs, "metadata": metadata}

        return {"summary": summary_text, "metadata": metadata}

    async def delete_observation(self, obs_id: str) -> bool:
        deleted = await self.db.delete_observation(obs_id)
        if deleted:
            self.vector_store.delete_where({"observation_id": obs_id})
        return bool(deleted)

    async def update_observation_tags(self, obs_id: str, tags: Optional[List[str]]) -> bool:
        if tags is None:
            return False
        normalized = _normalize_tags(tags)
        updated = await self.db.update_observation_tags(obs_id, normalized)
        return bool(updated)

    async def delete_project(self, project: str) -> int:
        deleted = await self.db.delete_project(project)
        if deleted:
            self.vector_store.delete_where({"project": project})
        return deleted

import asyncio
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
from .logging_config import get_logger
from .models import Observation, ObservationAsset, ObservationIndex, Session
from .providers.base import ChatMessage, ChatProvider, NoOpChatProvider
from .privacy import strip_memory_tags
from .structured import combine_extraction, StructuredData
from .vector_store import build_vector_store
from .performance import parallel_search

logger = get_logger("memory")

# User-level memory scope constant
# Used as a special "project" value to indicate user-global memories
USER_SCOPE_PROJECT = "_user"


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
        # Entity graph for knowledge graph features (Phase 4)
        self._entity_graph = None
        self._extract_entities: bool = getattr(self.config.context, 'enable_entity_extraction', True)

    async def initialize(self) -> None:
        logger.debug("Initializing MemoryManager")
        start = time.perf_counter()
        await self.db.connect()
        await self.db.create_tables()
        # Initialize entity graph
        if self._extract_entities:
            from .graph import EntityGraph
            self._entity_graph = EntityGraph(db=self.db)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(f"MemoryManager initialized in {duration_ms:.2f}ms")

    @property
    def entity_graph(self):
        """Get the entity graph manager."""
        return self._entity_graph

    async def close(self) -> None:
        logger.debug("Closing MemoryManager")
        await self.db.close()
        logger.info("MemoryManager closed")

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
            except Exception as e:
                # Log but don't propagate - one bad listener shouldn't break others
                logger.warning(f"Listener error: {type(e).__name__}: {e}")

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
        total_requests = self._search_cache_hits + self._search_cache_misses
        hit_rate = (
            (self._search_cache_hits / total_requests * 100)
            if total_requests > 0
            else 0.0
        )
        return {
            "enabled": self._is_search_cache_enabled(),
            "ttl_seconds": self.config.search.cache_ttl_seconds,
            "max_entries": self.config.search.cache_max_entries,
            "entries": len(self._search_cache),
            "hits": self._search_cache_hits,
            "misses": self._search_cache_misses,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "fts_weight": self.config.search.fts_weight,
            "vector_weight": self.config.search.vector_weight,
            "recency_half_life_hours": self.config.search.recency_half_life_hours,
            "recency_weight": self.config.search.recency_weight,
        }

    def clear_search_cache(self) -> Dict[str, Any]:
        """Clear the search cache and reset metrics."""
        cleared_entries = len(self._search_cache)
        previous_hits = self._search_cache_hits
        previous_misses = self._search_cache_misses
        self._search_cache.clear()
        self._search_cache_hits = 0
        self._search_cache_misses = 0
        logger.info(f"Search cache cleared: {cleared_entries} entries, {previous_hits} hits, {previous_misses} misses")
        return {
            "cleared_entries": cleared_entries,
            "previous_hits": previous_hits,
            "previous_misses": previous_misses,
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
        event_id: Optional[str] = None,
        host: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Observation]:
        logger.debug(f"Adding observation: type={obs_type}, project={project}, user_id={user_id}, event_id={event_id}")
        start = time.perf_counter()

        # Check idempotency: if event_id was already processed, return existing observation
        if event_id:
            existing_obs_id = await self.db.check_event_processed(event_id)
            if existing_obs_id:
                existing = await self.db.get_observation(existing_obs_id)
                if existing:
                    return Observation.model_validate(existing)

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
                    summary_text = await self.chat_provider.summarize(cleaned_content)
                except Exception as e:
                    # Graceful degradation: use truncation if LLM summarization fails
                    logger.debug(f"Summarization failed, using truncation: {type(e).__name__}")
                    summary_text = cleaned_content[:500] + "..."
            else:
                summary_text = cleaned_content[:500] + "..."

        if not session:
            return None

        # Extract structured data from content
        structured = combine_extraction(
            content=cleaned_content,
            tool_name=metadata.get("tool_name") if metadata else None,
            tool_input=metadata.get("tool_input") if metadata else None,
            tool_output=metadata.get("tool_output") if metadata else None,
        )
        if not structured.is_empty():
            metadata = dict(metadata) if metadata else {}
            metadata["structured"] = structured.to_dict()

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
        await self.db.add_observation(obs, user_id=user_id)
        if obs.assets:
            await self._store_assets(obs.id, obs.assets)
        self._index_observation(obs)
        self._notify_listeners(obs)

        # Extract entities for knowledge graph (Phase 4)
        if self._entity_graph and self._extract_entities:
            try:
                await self._entity_graph.extract_and_store(
                    text=cleaned_content,
                    observation_id=obs.id,
                    project=project_name,
                    created_at=obs.created_at,
                )
            except Exception as e:
                logger.warning(f"Entity extraction failed: {e}")

        # Record event_id for idempotency tracking
        if event_id:
            await self.db.record_event_processed(event_id, obs.id, host or "unknown")

        duration_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Observation added in {duration_ms:.2f}ms: id={obs.id}, type={obs.type}")
        return obs

    async def add_observations_batch(
        self,
        observations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Add multiple observations in a batch.

        Args:
            observations: List of observation dicts with keys:
                - content (required): Text content
                - obs_type: Observation type (default: "note")
                - project: Project path
                - session_id: Session ID
                - tags: List of tags
                - metadata: Dict of metadata
                - title: Optional title
                - summarize: Whether to summarize (default: True)

        Returns:
            Dict with:
                - added: Number of observations added
                - skipped: Number skipped (private content)
                - failed: Number that failed
                - ids: List of added observation IDs
                - errors: List of error messages for failed items
        """
        logger.info(f"Batch adding {len(observations)} observations")
        start = time.perf_counter()

        added = 0
        skipped = 0
        failed = 0
        ids: List[str] = []
        errors: List[Dict[str, Any]] = []

        for i, obs_data in enumerate(observations):
            try:
                content = obs_data.get("content")
                if not content:
                    errors.append({"index": i, "error": "Missing required field: content"})
                    failed += 1
                    continue

                obs = await self.add_observation(
                    content=content,
                    obs_type=obs_data.get("obs_type", "note"),
                    project=obs_data.get("project"),
                    session_id=obs_data.get("session_id"),
                    tags=obs_data.get("tags"),
                    metadata=obs_data.get("metadata"),
                    title=obs_data.get("title"),
                    summarize=obs_data.get("summarize", True),
                    event_id=obs_data.get("event_id"),
                    host=obs_data.get("host"),
                )

                if obs:
                    added += 1
                    ids.append(obs.id)
                else:
                    skipped += 1

            except Exception as e:
                errors.append({"index": i, "error": str(e)})
                failed += 1
                logger.warning(f"Batch add failed for item {i}: {e}")

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"Batch add complete in {duration_ms:.2f}ms: "
            f"added={added}, skipped={skipped}, failed={failed}"
        )

        return {
            "added": added,
            "skipped": skipped,
            "failed": failed,
            "ids": ids,
            "errors": errors,
        }

    async def delete_observations_batch(
        self,
        obs_ids: List[str],
    ) -> Dict[str, Any]:
        """Delete multiple observations in a batch.

        Args:
            obs_ids: List of observation IDs to delete

        Returns:
            Dict with:
                - deleted: Number of observations deleted
                - not_found: Number that were not found
                - ids: List of deleted observation IDs
        """
        logger.info(f"Batch deleting {len(obs_ids)} observations")
        start = time.perf_counter()

        deleted = 0
        not_found = 0
        deleted_ids: List[str] = []

        for obs_id in obs_ids:
            try:
                result = await self.delete_observation(obs_id)
                if result:
                    deleted += 1
                    deleted_ids.append(obs_id)
                else:
                    not_found += 1
            except Exception as e:
                not_found += 1
                logger.warning(f"Batch delete failed for {obs_id}: {e}")

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"Batch delete complete in {duration_ms:.2f}ms: "
            f"deleted={deleted}, not_found={not_found}"
        )

        return {
            "deleted": deleted,
            "not_found": not_found,
            "ids": deleted_ids,
        }

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
        user_id: Optional[str] = None,
    ) -> List[ObservationIndex]:
        logger.debug(f"Searching: query='{query}', project={project}, user_id={user_id}, limit={limit}")
        search_start = time.perf_counter()

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
                user_id=user_id,
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

        # ⚡ Parallel search: FTS and Vector run concurrently
        # Sequential: ~2-3s (800ms each), Parallel: ~800ms (both at same time)
        fts_results, vector_hits = await parallel_search(
            fts_search=lambda: self.db.search_observations_fts(
                query_str,
                project=project,
                obs_type=obs_type,
                session_id=session_id,
                date_start=start_ts,
                date_end=end_ts,
                tag_filters=tags,
                limit=self.config.search.fts_top_k,
                user_id=user_id,
            ),
            vector_search=lambda: self._vector_search(
                query_str,
                project=project,
                session_id=session_id,
                date_start=start_ts,
                date_end=end_ts,
                limit=self.config.search.vector_top_k,
                tag_filters=tags,
            ),
        )

        obs_map: Dict[str, ObservationIndex] = {}
        for item in fts_results:
            item.fts_score = item.score
            item.vector_score = None
            obs_map[item.id] = item

        # ⚡ Optimize: Batch fetch missing observations (avoid N+1)
        missing_ids = [obs_id for obs_id in vector_hits.keys() if obs_id not in obs_map]
        if missing_ids:
            # Batch fetch all missing observations at once
            missing_obs = await self.db.get_observations(missing_ids)
            for obs in missing_obs:
                score = vector_hits[obs["id"]]
                obs_map[obs["id"]] = ObservationIndex(
                    id=obs["id"],
                    summary=obs["summary"] or "",
                    project=obs["project"],
                    type=obs["type"],
                    created_at=obs["created_at"],
                    fts_score=None,
                    vector_score=score,
                )
        
        # Update scores for observations that appear in both searches
        for obs_id, score in vector_hits.items():
            if obs_id in obs_map:
                existing = obs_map[obs_id]
                existing.vector_score = max(existing.vector_score or 0.0, score)

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

        duration_ms = (time.perf_counter() - search_start) * 1000
        logger.debug(f"Search completed in {duration_ms:.2f}ms, found {len(final_results)} results")
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
                summary_text = await self.chat_provider.chat(
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

    # ==================== Memory Consolidation (Phase 3) ====================

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        # Tokenize into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _compute_embedding_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity between -1 and 1
        """
        if not embedding1 or not embedding2:
            return 0.0
        # Compute dot product and magnitudes
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        mag1 = math.sqrt(sum(a * a for a in embedding1))
        mag2 = math.sqrt(sum(b * b for b in embedding2))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot_product / (mag1 * mag2)

    async def find_similar_observations(
        self,
        project: str,
        similarity_threshold: float = 0.85,
        use_embeddings: bool = True,
        obs_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Tuple[str, str, float]]:
        """Find pairs of similar observations that might be duplicates.

        Args:
            project: Project to search in
            similarity_threshold: Minimum similarity to consider as duplicate
            use_embeddings: Use embedding similarity (more accurate but slower)
            obs_type: Optional type filter
            limit: Maximum observations to analyze

        Returns:
            List of (obs_id_1, obs_id_2, similarity_score) tuples
        """
        logger.info(f"Finding similar observations in project={project}, threshold={similarity_threshold}")
        start = time.perf_counter()

        observations = await self.db.get_similar_observations(
            project=project,
            obs_type=obs_type,
            exclude_superseded=True,
            limit=limit,
        )

        if len(observations) < 2:
            return []

        similar_pairs: List[Tuple[str, str, float]] = []

        # Compute embeddings if using embedding similarity
        embeddings_map: Dict[str, List[float]] = {}
        if use_embeddings:
            texts = [obs.get("summary") or obs.get("content", "")[:500] for obs in observations]
            try:
                all_embeddings = self.embedding_provider.embed(texts)
                for i, obs in enumerate(observations):
                    embeddings_map[obs["id"]] = all_embeddings[i]
            except Exception as e:
                logger.warning(f"Failed to compute embeddings for similarity: {e}")
                use_embeddings = False

        # Compare all pairs (O(n²) - consider optimization for large datasets)
        for i in range(len(observations)):
            for j in range(i + 1, len(observations)):
                obs1 = observations[i]
                obs2 = observations[j]

                if use_embeddings and obs1["id"] in embeddings_map and obs2["id"] in embeddings_map:
                    similarity = self._compute_embedding_similarity(
                        embeddings_map[obs1["id"]],
                        embeddings_map[obs2["id"]],
                    )
                else:
                    # Fallback to text similarity
                    text1 = obs1.get("summary") or obs1.get("content", "")
                    text2 = obs2.get("summary") or obs2.get("content", "")
                    similarity = self._compute_text_similarity(text1, text2)

                if similarity >= similarity_threshold:
                    similar_pairs.append((obs1["id"], obs2["id"], similarity))

        # Sort by similarity (highest first)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(f"Found {len(similar_pairs)} similar pairs in {duration_ms:.2f}ms")

        return similar_pairs

    async def consolidate_memories(
        self,
        project: str,
        similarity_threshold: float = 0.85,
        keep_strategy: str = "newest",
        obs_type: Optional[str] = None,
        dry_run: bool = False,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Consolidate similar memories by marking older ones as superseded.

        This helps reduce redundancy and keep memory lean over time.

        Args:
            project: Project to consolidate
            similarity_threshold: Minimum similarity to consider as duplicate (0.0-1.0)
            keep_strategy: "newest" keeps the most recent, "oldest" keeps the oldest,
                          "highest_importance" keeps highest importance_score
            obs_type: Optional type filter
            dry_run: If True, don't actually modify anything
            limit: Maximum observations to analyze

        Returns:
            Dict with consolidation stats:
                - analyzed: Number of observations analyzed
                - pairs_found: Number of similar pairs found
                - consolidated: Number of observations marked as superseded
                - kept_ids: IDs of observations that were kept
                - superseded_ids: IDs of observations marked as superseded
        """
        logger.info(f"Consolidating memories for project={project}, strategy={keep_strategy}, dry_run={dry_run}")
        start = time.perf_counter()

        similar_pairs = await self.find_similar_observations(
            project=project,
            similarity_threshold=similarity_threshold,
            use_embeddings=True,
            obs_type=obs_type,
            limit=limit,
        )

        if not similar_pairs:
            return {
                "analyzed": limit,
                "pairs_found": 0,
                "consolidated": 0,
                "kept_ids": [],
                "superseded_ids": [],
            }

        # Track which observations to keep and which to supersede
        kept_ids: set = set()
        superseded_ids: set = set()
        supersede_map: Dict[str, str] = {}  # obs_id -> superseded_by_id

        # Fetch full observation data for decision making
        all_obs_ids = set()
        for obs1_id, obs2_id, _ in similar_pairs:
            all_obs_ids.add(obs1_id)
            all_obs_ids.add(obs2_id)

        observations_data = await self.db.get_observations(list(all_obs_ids))
        obs_lookup = {obs["id"]: obs for obs in observations_data}

        for obs1_id, obs2_id, similarity in similar_pairs:
            # Skip if either observation is already superseded
            if obs1_id in superseded_ids or obs2_id in superseded_ids:
                continue

            obs1 = obs_lookup.get(obs1_id)
            obs2 = obs_lookup.get(obs2_id)

            if not obs1 or not obs2:
                continue

            # Decide which to keep based on strategy
            if keep_strategy == "newest":
                keep, supersede = (obs1, obs2) if obs1["created_at"] >= obs2["created_at"] else (obs2, obs1)
            elif keep_strategy == "oldest":
                keep, supersede = (obs1, obs2) if obs1["created_at"] <= obs2["created_at"] else (obs2, obs1)
            elif keep_strategy == "highest_importance":
                score1 = obs1.get("importance_score", 0.5)
                score2 = obs2.get("importance_score", 0.5)
                keep, supersede = (obs1, obs2) if score1 >= score2 else (obs2, obs1)
            else:
                # Default to newest
                keep, supersede = (obs1, obs2) if obs1["created_at"] >= obs2["created_at"] else (obs2, obs1)

            kept_ids.add(keep["id"])
            superseded_ids.add(supersede["id"])
            supersede_map[supersede["id"]] = keep["id"]

        # Apply changes if not dry run
        consolidated_count = 0
        if not dry_run:
            for obs_id, superseded_by in supersede_map.items():
                try:
                    result = await self.db.mark_superseded(obs_id, superseded_by)
                    if result:
                        consolidated_count += 1
                except Exception as e:
                    logger.warning(f"Failed to mark {obs_id} as superseded: {e}")

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(f"Consolidation complete in {duration_ms:.2f}ms: consolidated={consolidated_count}")

        return {
            "analyzed": len(all_obs_ids),
            "pairs_found": len(similar_pairs),
            "consolidated": consolidated_count if not dry_run else len(superseded_ids),
            "kept_ids": list(kept_ids),
            "superseded_ids": list(superseded_ids),
            "dry_run": dry_run,
        }

    async def cleanup_stale_memories(
        self,
        project: str,
        max_age_days: int = 90,
        min_access_count: int = 0,
        dry_run: bool = False,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Remove or flag old, rarely-accessed memories.

        Args:
            project: Project to clean up
            max_age_days: Only consider observations older than this
            min_access_count: Only consider observations with access count <= this
            dry_run: If True, don't actually delete
            limit: Maximum observations to consider

        Returns:
            Dict with cleanup stats
        """
        logger.info(f"Cleaning stale memories for project={project}, max_age={max_age_days}d, dry_run={dry_run}")

        stale = await self.db.get_stale_observations(
            project=project,
            max_age_days=max_age_days,
            min_access_count=min_access_count,
            exclude_superseded=True,
            limit=limit,
        )

        deleted_count = 0
        deleted_ids: List[str] = []

        if not dry_run:
            for obs in stale:
                try:
                    result = await self.delete_observation(obs["id"])
                    if result:
                        deleted_count += 1
                        deleted_ids.append(obs["id"])
                except Exception as e:
                    logger.warning(f"Failed to delete stale observation {obs['id']}: {e}")

        return {
            "candidates_found": len(stale),
            "deleted": deleted_count if not dry_run else len(stale),
            "deleted_ids": deleted_ids if not dry_run else [obs["id"] for obs in stale],
            "dry_run": dry_run,
        }

    # ==================== Two-Stage Retrieval (Phase 3) ====================

    def _rerank_results(
        self,
        query: str,
        results: List[ObservationIndex],
        top_k: int = 10,
        reranker_type: Optional[str] = None,
    ) -> List[ObservationIndex]:
        """Rerank search results using configurable strategy.

        Stage 1 (BM25 + vector) provides recall, Stage 2 (reranking) improves precision.

        Supported reranker types:
        - "biencoder": Embedding similarity (default, good balance)
        - "tfidf": TF-IDF similarity (fast, no external deps)
        - "crossencoder": Cross-encoder model (most accurate, requires model)

        Args:
            query: Original search query
            results: Results from Stage 1
            top_k: Number of results to return after reranking
            reranker_type: Override reranker type (default: from config)

        Returns:
            Reranked list of results
        """
        if not results or len(results) <= 1:
            return results[:top_k]

        # Get reranker type from config or parameter
        reranker = reranker_type or self.config.search.reranker_type
        rerank_weight = self.config.search.rerank_weight

        if reranker == "tfidf":
            return self._rerank_with_tfidf(query, results, top_k, rerank_weight)
        elif reranker == "crossencoder":
            return self._rerank_with_crossencoder(query, results, top_k, rerank_weight)
        else:
            # Default: biencoder
            return self._rerank_with_biencoder(query, results, top_k, rerank_weight)

    def _rerank_with_biencoder(
        self,
        query: str,
        results: List[ObservationIndex],
        top_k: int,
        rerank_weight: float,
    ) -> List[ObservationIndex]:
        """Rerank using bi-encoder (separate query and document embeddings)."""
        try:
            # Get query embedding
            query_embedding = self.embedding_provider.embed([query])[0]

            # Get embeddings for result summaries
            texts = [r.summary or "" for r in results]
            result_embeddings = self.embedding_provider.embed(texts)

            # Compute rerank scores
            rerank_scores: List[Tuple[ObservationIndex, float]] = []
            original_weight = 1.0 - rerank_weight

            for i, result in enumerate(results):
                similarity = self._compute_embedding_similarity(
                    query_embedding,
                    result_embeddings[i],
                )
                combined = (result.score or 0.0) * original_weight + similarity * rerank_weight
                result.rerank_score = combined
                rerank_scores.append((result, combined))

            # Sort by combined score
            rerank_scores.sort(key=lambda x: x[1], reverse=True)
            return [item[0] for item in rerank_scores[:top_k]]

        except Exception as e:
            logger.warning(f"Bi-encoder reranking failed: {e}")
            return results[:top_k]

    def _rerank_with_tfidf(
        self,
        query: str,
        results: List[ObservationIndex],
        top_k: int,
        rerank_weight: float,
    ) -> List[ObservationIndex]:
        """Rerank using TF-IDF similarity (fast, no external deps)."""
        try:
            # Build document frequency from results
            query_terms = set(query.lower().split())
            doc_freq: Dict[str, int] = {}

            texts = [r.summary or "" for r in results]
            for text in texts:
                unique_terms = set(text.lower().split())
                for term in unique_terms:
                    doc_freq[term] = doc_freq.get(term, 0) + 1

            n_docs = len(texts)
            rerank_scores: List[Tuple[ObservationIndex, float]] = []
            original_weight = 1.0 - rerank_weight

            for i, result in enumerate(results):
                text = texts[i]
                text_terms = text.lower().split()

                # Compute TF-IDF similarity with query
                tfidf_score = 0.0
                term_freq: Dict[str, int] = {}
                for term in text_terms:
                    term_freq[term] = term_freq.get(term, 0) + 1

                for term in query_terms:
                    if term in term_freq:
                        tf = term_freq[term] / len(text_terms) if text_terms else 0
                        idf = math.log((n_docs + 1) / (doc_freq.get(term, 0) + 1))
                        tfidf_score += tf * idf

                # Normalize
                max_score = len(query_terms) * math.log(n_docs + 1) if query_terms else 1
                normalized = tfidf_score / max_score if max_score > 0 else 0

                combined = (result.score or 0.0) * original_weight + normalized * rerank_weight
                result.rerank_score = combined
                rerank_scores.append((result, combined))

            rerank_scores.sort(key=lambda x: x[1], reverse=True)
            return [item[0] for item in rerank_scores[:top_k]]

        except Exception as e:
            logger.warning(f"TF-IDF reranking failed: {e}")
            return results[:top_k]

    def _rerank_with_crossencoder(
        self,
        query: str,
        results: List[ObservationIndex],
        top_k: int,
        rerank_weight: float,
    ) -> List[ObservationIndex]:
        """Rerank using cross-encoder model (most accurate).

        Cross-encoders encode query and document together, allowing for
        more nuanced similarity computation. Falls back to bi-encoder
        if cross-encoder not available.
        """
        try:
            # Try to import sentence-transformers for cross-encoder
            from sentence_transformers import CrossEncoder

            model_name = self.config.search.crossencoder_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"

            # Cache the cross-encoder model
            if not hasattr(self, "_crossencoder_model") or self._crossencoder_model_name != model_name:
                logger.info(f"Loading cross-encoder model: {model_name}")
                self._crossencoder_model = CrossEncoder(model_name)
                self._crossencoder_model_name = model_name

            # Create query-document pairs
            texts = [r.summary or "" for r in results]
            pairs = [(query, text) for text in texts]

            # Get cross-encoder scores
            scores = self._crossencoder_model.predict(pairs)

            # Combine with original scores
            rerank_scores: List[Tuple[ObservationIndex, float]] = []
            original_weight = 1.0 - rerank_weight

            # Normalize cross-encoder scores to 0-1
            min_score = min(scores) if scores.size > 0 else 0
            max_score = max(scores) if scores.size > 0 else 1
            score_range = max_score - min_score if max_score > min_score else 1

            for i, result in enumerate(results):
                normalized = (scores[i] - min_score) / score_range if score_range > 0 else 0
                combined = (result.score or 0.0) * original_weight + normalized * rerank_weight
                result.rerank_score = combined
                rerank_scores.append((result, combined))

            rerank_scores.sort(key=lambda x: x[1], reverse=True)
            return [item[0] for item in rerank_scores[:top_k]]

        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to bi-encoder")
            return self._rerank_with_biencoder(query, results, top_k, rerank_weight)
        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed: {e}")
            return self._rerank_with_biencoder(query, results, top_k, rerank_weight)

    async def search_with_rerank(
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
        stage1_limit: int = 50,
    ) -> List[ObservationIndex]:
        """Two-stage search with reranking for improved precision.

        Stage 1: BM25 + Vector search (high recall)
        Stage 2: Embedding-based reranking (high precision)

        Args:
            query: Search query
            limit: Final number of results to return
            project: Project filter
            obs_type: Observation type filter
            session_id: Session filter
            date_start: Start date filter
            date_end: End date filter
            since: Alternative to date_start
            tag_filters: Tag filters
            stage1_limit: Number of candidates to fetch in Stage 1

        Returns:
            Reranked list of ObservationIndex
        """
        logger.debug(f"Two-stage search: query='{query}', limit={limit}, stage1_limit={stage1_limit}")
        start = time.perf_counter()

        # Stage 1: Get candidates using existing hybrid search
        stage1_results = await self.search(
            query=query,
            limit=stage1_limit,
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=date_start,
            date_end=date_end,
            since=since,
            tag_filters=tag_filters,
        )

        if not stage1_results:
            return []

        # Stage 2: Rerank for precision
        final_results = self._rerank_results(query, stage1_results, top_k=limit)

        # Track access for retrieved results
        for result in final_results[:5]:  # Track top 5 accessed
            try:
                await self.db.increment_access_count(result.id)
            except Exception:
                pass  # Don't fail search if access tracking fails

        duration_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Two-stage search completed in {duration_ms:.2f}ms, returned {len(final_results)} results")

        return final_results

    # =========================================================================
    # User-Level Memory Methods
    # =========================================================================

    async def add_user_memory(
        self,
        content: str,
        obs_type: str = "preference",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, object]] = None,
        title: Optional[str] = None,
        summarize: bool = True,
    ) -> Optional[Observation]:
        """Add a user-level (global) memory.

        User memories are not tied to a specific project and persist across
        all projects. Use for user preferences, conventions, and global context.

        Args:
            content: Memory content
            obs_type: Observation type (default: "preference")
            tags: List of tags
            metadata: Additional metadata
            title: Optional title
            summarize: Whether to generate summary

        Returns:
            Created Observation or None if filtered
        """
        logger.debug("Adding user-level memory")
        return await self.add_observation(
            content=content,
            obs_type=obs_type,
            project=USER_SCOPE_PROJECT,
            session_id=None,
            tags=tags or [],
            metadata=metadata,
            title=title,
            summarize=summarize,
        )

    async def get_user_memories(
        self,
        limit: int = 50,
        obs_type: Optional[str] = None,
        tag_filters: Optional[List[str]] = None,
    ) -> List[ObservationIndex]:
        """Get user-level memories.

        Args:
            limit: Maximum memories to return
            obs_type: Filter by observation type
            tag_filters: Filter by tags

        Returns:
            List of user memory indices
        """
        logger.debug(f"Getting user memories: limit={limit}")
        return await self.db.list_observations(
            project=USER_SCOPE_PROJECT,
            limit=limit,
            obs_type=obs_type,
            tag_filters=tag_filters,
        )

    async def search_user_memories(
        self,
        query: str,
        limit: int = 10,
        obs_type: Optional[str] = None,
        tag_filters: Optional[List[str]] = None,
    ) -> List[ObservationIndex]:
        """Search user-level memories.

        Args:
            query: Search query
            limit: Maximum results
            obs_type: Filter by observation type
            tag_filters: Filter by tags

        Returns:
            List of matching user memories
        """
        logger.debug(f"Searching user memories: query='{query}'")
        return await self.search(
            query=query,
            limit=limit,
            project=USER_SCOPE_PROJECT,
            obs_type=obs_type,
            tag_filters=tag_filters,
        )

    async def export_user_memories(
        self,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export user memories to JSON file.

        Args:
            output_path: Path to output file (default: ~/.config/ai-mem/user-memory.json)

        Returns:
            Dict with export info and path
        """
        import os
        from pathlib import Path

        if output_path is None:
            config_dir = Path.home() / ".config" / "ai-mem"
            config_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(config_dir / "user-memory.json")

        logger.info(f"Exporting user memories to {output_path}")

        # Get all user memories
        memories_list = await self.db.list_observations(
            project=USER_SCOPE_PROJECT,
            limit=1000,
        )

        # Need full observation data for export
        full_memories = []
        for idx in memories_list:
            obs_dict = await self.db.get_observation(idx.id)
            if obs_dict:
                full_memories.append(obs_dict)

        export_data = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "count": len(full_memories),
            "memories": full_memories,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported {len(full_memories)} user memories to {output_path}")

        return {
            "path": output_path,
            "count": len(full_memories),
        }

    async def import_user_memories(
        self,
        input_path: Optional[str] = None,
        merge: bool = True,
    ) -> Dict[str, Any]:
        """Import user memories from JSON file.

        Args:
            input_path: Path to input file (default: ~/.config/ai-mem/user-memory.json)
            merge: If True, merge with existing; if False, replace all

        Returns:
            Dict with import info
        """
        from pathlib import Path

        if input_path is None:
            input_path = str(Path.home() / ".config" / "ai-mem" / "user-memory.json")

        if not os.path.exists(input_path):
            logger.warning(f"User memory file not found: {input_path}")
            return {"imported": 0, "skipped": 0, "errors": [], "path": input_path}

        logger.info(f"Importing user memories from {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            import_data = json.load(f)

        memories = import_data.get("memories", [])
        imported = 0
        skipped = 0
        errors: List[str] = []

        if not merge:
            # Delete existing user memories
            existing = await self.db.list_observations(project=USER_SCOPE_PROJECT, limit=10000)
            for mem in existing:
                await self.db.delete_observation(mem.id)  # ObservationIndex has .id attribute
            logger.info(f"Cleared {len(existing)} existing user memories")

        for mem in memories:
            try:
                content = mem.get("content")
                if not content:
                    skipped += 1
                    continue

                obs = await self.add_user_memory(
                    content=content,
                    obs_type=mem.get("type", "preference"),
                    tags=mem.get("tags", []),
                    metadata=mem.get("metadata"),
                    summarize=False,
                )

                if obs:
                    imported += 1
                else:
                    skipped += 1

            except Exception as e:
                errors.append(f"Error importing memory: {str(e)}")
                logger.warning(f"Failed to import memory: {e}")

        logger.info(f"Imported {imported} user memories, skipped {skipped}")

        return {
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
            "path": input_path,
        }

    async def get_user_memory_count(self) -> int:
        """Get count of user-level memories.

        Returns:
            Number of user memories
        """
        stats = await self.db.get_stats(project=USER_SCOPE_PROJECT)
        return stats.get("total", 0)

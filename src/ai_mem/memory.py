import hashlib
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from .chunking import chunk_text
from .config import AppConfig, load_config, resolve_storage_paths
from .db import DatabaseManager
from .embeddings.base import EmbeddingProvider
from .models import Observation, ObservationIndex, Session
from .providers.base import ChatProvider, NoOpChatProvider
from .vector_store import VectorStore


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
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


class MemoryManager:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        storage = resolve_storage_paths(self.config)
        os.makedirs(storage.data_dir, exist_ok=True)
        os.makedirs(storage.vector_dir, exist_ok=True)
        self.db = DatabaseManager(storage.sqlite_path)
        self.vector_store = VectorStore(storage.vector_dir)
        self.chat_provider = _build_chat_provider(self.config)
        self.embedding_provider = _build_embedding_provider(self.config)
        self.allow_llm_summaries = self.chat_provider.get_name() != "none"
        self.current_session: Optional[Session] = None

    def start_session(self, project: str, goal: str = "") -> Session:
        if self.current_session:
            self.close_session()
        self.current_session = Session(project=project, goal=goal)
        self.db.add_session(self.current_session)
        return self.current_session

    def close_session(self) -> None:
        if not self.current_session:
            return
        import time

        self.current_session.end_time = time.time()
        self.db.add_session(self.current_session)
        self.current_session = None

    def add_observation(
        self,
        content: str,
        obs_type: str,
        project: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, object]] = None,
        title: Optional[str] = None,
        summarize: bool = True,
        dedupe: bool = True,
        summary: Optional[str] = None,
        created_at: Optional[float] = None,
        importance_score: float = 0.5,
    ) -> Observation:
        project_name = project or (self.current_session.project if self.current_session else os.getcwd())
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if dedupe:
            existing = self.db.find_observation_by_hash(content_hash, project_name)
            if existing:
                return Observation.model_validate(existing)

        if not self.current_session:
            self.start_session(project=project_name, goal="Auto-started session")

        summary_text = summary if summary is not None else content
        if summary is None and len(content) > 500:
            if summarize and self.allow_llm_summaries:
                try:
                    summary_text = self.chat_provider.summarize(content)
                except Exception:
                    summary_text = content[:500] + "..."
            else:
                summary_text = content[:500] + "..."

        obs = Observation(
            session_id=self.current_session.id,
            project=project_name,
            type=obs_type,
            title=title,
            content=content,
            summary=summary_text,
            created_at=created_at or time.time(),
            importance_score=importance_score,
            tags=tags or [],
            metadata=metadata or {},
            content_hash=content_hash,
        )
        self.db.add_observation(obs)
        self._index_observation(obs)
        return obs

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

    def search(
        self,
        query: str,
        limit: int = 10,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
    ) -> List[ObservationIndex]:
        start_ts = _parse_date(date_start)
        end_ts = _parse_date(date_end)
        if not query.strip():
            return self.db.get_recent_observations(project, limit, start_ts, end_ts)

        fts_results = self.db.search_observations_fts(
            query,
            project=project,
            obs_type=obs_type,
            date_start=start_ts,
            date_end=end_ts,
            limit=self.config.search.fts_top_k,
        )
        vector_hits = self._vector_search(
            query,
            project=project,
            date_start=start_ts,
            date_end=end_ts,
            limit=self.config.search.vector_top_k,
        )

        merged: Dict[str, ObservationIndex] = {}
        for item in fts_results:
            merged[item.id] = item

        for obs_id, score in vector_hits.items():
            if obs_id in merged:
                merged[obs_id].score = max(merged[obs_id].score, score)
            else:
                obs = self.db.get_observation(obs_id)
                if obs:
                    merged[obs_id] = ObservationIndex(
                        id=obs["id"],
                        summary=obs["summary"] or "",
                        project=obs["project"],
                        type=obs["type"],
                        created_at=obs["created_at"],
                        score=score,
                    )

        results = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        if obs_type:
            results = [item for item in results if item.type == obs_type]
        return results[:limit]

    def _vector_search(
        self,
        query: str,
        project: Optional[str],
        date_start: Optional[float],
        date_end: Optional[float],
        limit: int,
    ) -> Dict[str, float]:
        embedding = self.embedding_provider.embed([query])[0]
        where = {"project": project} if project else None
        results = self.vector_store.query(
            embedding=embedding,
            n_results=limit,
            where=where,
        )
        scores: Dict[str, float] = {}
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
                distance = 0.0
                if results.get("distances"):
                    distance = float(results["distances"][0][idx])
                score = 1.0 - distance
                scores[obs_id] = max(scores.get(obs_id, 0.0), score)
        return scores

    def timeline(
        self,
        anchor_id: Optional[str] = None,
        query: Optional[str] = None,
        depth_before: int = 3,
        depth_after: int = 3,
        project: Optional[str] = None,
    ) -> List[ObservationIndex]:
        if not anchor_id and query:
            results = self.search(query, limit=1, project=project)
            if results:
                anchor_id = results[0].id

        if not anchor_id:
            return []

        anchor = self.db.get_observation(anchor_id)
        if not anchor:
            return []

        project_name = project or anchor["project"]
        before = self.db.get_observations_before(
            project=project_name,
            anchor_time=anchor["created_at"],
            limit=depth_before,
        )
        after = self.db.get_observations_after(
            project=project_name,
            anchor_time=anchor["created_at"],
            limit=depth_after,
        )
        anchor_index = ObservationIndex(
            id=anchor["id"],
            summary=anchor["summary"] or "",
            project=anchor["project"],
            type=anchor["type"],
            created_at=anchor["created_at"],
            score=0.0,
        )
        timeline = list(reversed(before)) + [anchor_index] + after
        return timeline

    def get_observations(self, ids: List[str]) -> List[Dict[str, object]]:
        return self.db.get_observations(ids)

    def list_projects(self) -> List[str]:
        return self.db.list_projects()

    def get_stats(
        self,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        tag_limit: int = 10,
        day_limit: int = 14,
        type_tag_limit: int = 3,
    ) -> Dict[str, Any]:
        start_ts = _parse_date(date_start)
        end_ts = _parse_date(date_end)
        return self.db.get_stats(
            project=project,
            obs_type=obs_type,
            date_start=start_ts,
            date_end=end_ts,
            tag_limit=tag_limit,
            day_limit=day_limit,
            type_tag_limit=type_tag_limit,
        )

    def export_observations(
        self,
        project: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        return self.db.list_observations(project=project, limit=limit)

    def delete_observation(self, obs_id: str) -> bool:
        deleted = self.db.delete_observation(obs_id)
        if deleted:
            self.vector_store.delete_where({"observation_id": obs_id})
        return bool(deleted)

    def delete_project(self, project: str) -> int:
        deleted = self.db.delete_project(project)
        if deleted:
            self.vector_store.delete_where({"project": project})
        return deleted

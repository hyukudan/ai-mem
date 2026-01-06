import hashlib
import os
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .chunking import chunk_text
from .config import AppConfig, load_config, resolve_storage_paths
from .db import DatabaseManager
from .embeddings.base import EmbeddingProvider
from .models import Observation, ObservationIndex, Session
from .providers.base import ChatMessage, ChatProvider, NoOpChatProvider
from .privacy import strip_memory_tags
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
        self.vector_store = VectorStore(storage.vector_dir)
        self.chat_provider = _build_chat_provider(self.config)
        self.embedding_provider = _build_embedding_provider(self.config)
        self.allow_llm_summaries = self.chat_provider.get_name() != "none"
        self.current_session: Optional[Session] = None
        self._listeners: List[Callable[[Observation], None]] = []
        self._listener_lock = threading.Lock()

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

    def start_session(
        self,
        project: str,
        goal: str = "",
        session_id: Optional[str] = None,
    ) -> Session:
        if self.current_session:
            self.close_session()
        if session_id:
            existing = self.db.get_session(session_id)
            if existing:
                session = Session.model_validate(existing)
                if goal and goal != session.goal:
                    session.goal = goal
                    self.db.add_session(session)
                self.current_session = session
                return session
            self.current_session = Session(id=session_id, project=project, goal=goal)
        else:
            self.current_session = Session(project=project, goal=goal)
        self.db.add_session(self.current_session)
        return self.current_session

    def _ensure_session(self, project: str) -> Session:
        if self.current_session:
            if self.current_session.project == project:
                return self.current_session
            self.close_session()
        existing = self.db.list_sessions(project=project, active_only=True, limit=1)
        if existing:
            self.current_session = Session.model_validate(existing[0])
            return self.current_session
        return self.start_session(project=project, goal="Auto-started session")

    def close_session(self) -> None:
        if not self.current_session:
            return
        import time

        self.current_session.end_time = time.time()
        self.db.add_session(self.current_session)
        self.current_session = None

    def end_session(self, session_id: Optional[str] = None) -> Optional[Session]:
        if session_id:
            data = self.db.get_session(session_id)
            if not data:
                return None
            session = Session.model_validate(data)
        else:
            session = self.current_session
            if not session:
                return None
        session.end_time = time.time()
        self.db.add_session(session)
        if self.current_session and self.current_session.id == session.id:
            self.current_session = None
        return session

    def end_latest_session(self, project: Optional[str] = None) -> Optional[Session]:
        project_name = project or (self.current_session.project if self.current_session else None)
        if not project_name:
            return None
        sessions = self.db.list_sessions(project=project_name, active_only=True, limit=1)
        if not sessions:
            return None
        session = Session.model_validate(sessions[0])
        session.end_time = time.time()
        self.db.add_session(session)
        if self.current_session and self.current_session.id == session.id:
            self.current_session = None
        return session

    def add_observation(
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
    ) -> Optional[Observation]:
        session = None
        if session_id:
            session_data = self.db.get_session(session_id)
            if session_data:
                session = Session.model_validate(session_data)
        if session:
            project_name = session.project
        else:
            project_name = project or (os.getcwd() if session_id else (self.current_session.project if self.current_session else os.getcwd()))
            if session_id:
                session = Session(id=session_id, project=project_name)
                self.db.add_session(session)
        cleaned_content, stripped = strip_memory_tags(content)
        if not cleaned_content:
            return None

        metadata = dict(metadata or {})
        if stripped:
            metadata["redacted"] = True

        content_hash = hashlib.sha256(cleaned_content.encode("utf-8")).hexdigest()
        if dedupe:
            existing = self.db.find_observation_by_hash(content_hash, project_name)
            if existing:
                return Observation.model_validate(existing)

        if not session_id:
            if not self.current_session:
                self._ensure_session(project_name)
            session = self.current_session

        summary_text = summary if summary is not None else cleaned_content
        if summary is None and len(cleaned_content) > 500:
            if summarize and self.allow_llm_summaries:
                try:
                    summary_text = self.chat_provider.summarize(cleaned_content)
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
        )
        self.db.add_observation(obs)
        self._index_observation(obs)
        self._notify_listeners(obs)
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

    def search(
        self,
        query: str,
        limit: int = 10,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        tag_filters: Optional[List[str]] = None,
    ) -> List[ObservationIndex]:
        start_ts = _parse_date(date_start)
        end_ts = _parse_date(date_end)
        tags = _normalize_tags(tag_filters)
        if not query.strip():
            return self.db.get_recent_observations(
                project,
                limit,
                obs_type=obs_type,
                session_id=session_id,
                tag_filters=tags,
                date_start=start_ts,
                date_end=end_ts,
            )

        fts_results = self.db.search_observations_fts(
            query,
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=start_ts,
            date_end=end_ts,
            tag_filters=tags,
            limit=self.config.search.fts_top_k,
        )
        vector_hits = self._vector_search(
            query,
            project=project,
            session_id=session_id,
            date_start=start_ts,
            date_end=end_ts,
            limit=self.config.search.vector_top_k,
            tag_filters=tags,
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
                    obs = self.db.get_observation(obs_id)
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
                            obs = self.db.get_observation(obs_id)
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

    def timeline(
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
        tag_filters: Optional[List[str]] = None,
    ) -> List[ObservationIndex]:
        tags = _normalize_tags(tag_filters)
        if not anchor_id and query:
            results = self.search(
                query,
                limit=1,
                project=project,
                obs_type=obs_type,
                session_id=session_id,
                date_start=date_start,
                date_end=date_end,
                tag_filters=tags,
            )
            if results:
                anchor_id = results[0].id

        if not anchor_id:
            return []

        anchor = self.db.get_observation(anchor_id)
        if not anchor:
            return []
        if tags and not _match_tags(anchor.get("tags") or [], tags):
            return []
        if session_id and anchor.get("session_id") != session_id:
            return []

        start_ts = _parse_date(date_start)
        end_ts = _parse_date(date_end)
        project_name = project or anchor["project"]
        before = self.db.get_observations_before(
            project=project_name,
            anchor_time=anchor["created_at"],
            limit=depth_before,
            obs_type=obs_type,
            session_id=session_id,
            tag_filters=tags,
            date_start=start_ts,
            date_end=end_ts,
        )
        after = self.db.get_observations_after(
            project=project_name,
            anchor_time=anchor["created_at"],
            limit=depth_after,
            obs_type=obs_type,
            session_id=session_id,
            tag_filters=tags,
            date_start=start_ts,
            date_end=end_ts,
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

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.db.get_session(session_id)

    def list_sessions(
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
        return self.db.list_sessions(
            project=project,
            active_only=active_only,
            goal_query=goal_query,
            date_start=start_ts,
            date_end=end_ts,
            limit=limit,
        )

    def get_stats(
        self,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        tag_filters: Optional[List[str]] = None,
        tag_limit: int = 10,
        day_limit: int = 14,
        type_tag_limit: int = 3,
    ) -> Dict[str, Any]:
        start_ts = _parse_date(date_start)
        end_ts = _parse_date(date_end)
        return self.db.get_stats(
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

    def export_observations(
        self,
        project: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        return self.db.list_observations(project=project, session_id=session_id, limit=limit)

    def summarize_project(
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
            session_data = self.db.get_session(session_id)
            if not session_data:
                return None
            session = Session.model_validate(session_data)
        project_name = (
            session.project
            if session
            else project or (self.current_session.project if self.current_session else os.getcwd())
        )
        observations = self.db.list_observations(
            project=project_name if not session_id else None,
            session_id=session_id,
            limit=limit,
        )
        if obs_type:
            observations = [obs for obs in observations if obs.get("type") == obs_type]
        if not observations:
            return None

        observations = list(reversed(observations))
        lines = []
        for obs in observations:
            summary = obs.get("summary") or obs.get("content") or ""
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
                summary_text = self.chat_provider.chat(
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
            "source_ids": [obs.get("id") for obs in observations if obs.get("id")],
        }
        if session_id:
            metadata["session_id"] = session_id

        if store:
            summary_tags = list(tags or [])
            if session_id and "session-summary" not in summary_tags:
                summary_tags.append("session-summary")
            obs = self.add_observation(
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
                self.db.add_session(session)
            return {"summary": summary_text, "observation": obs, "metadata": metadata}

        return {"summary": summary_text, "metadata": metadata}

    def delete_observation(self, obs_id: str) -> bool:
        deleted = self.db.delete_observation(obs_id)
        if deleted:
            self.vector_store.delete_where({"observation_id": obs_id})
        return bool(deleted)

    def update_observation_tags(self, obs_id: str, tags: Optional[List[str]]) -> bool:
        if tags is None:
            return False
        normalized = _normalize_tags(tags)
        updated = self.db.update_observation_tags(obs_id, normalized)
        return bool(updated)

    def delete_project(self, project: str) -> int:
        deleted = self.db.delete_project(project)
        if deleted:
            self.vector_store.delete_where({"project": project})
        return deleted

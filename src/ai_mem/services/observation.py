"""Observation Service - Create, store, retrieve observations.

This service handles all observation lifecycle operations
including creation, storage, retrieval, and deletion.
"""

import hashlib
import time
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..logging_config import get_logger

if TYPE_CHECKING:
    from ..config import AppConfig
    from ..db import DatabaseManager
    from ..models import Observation
    from .session import SessionService

logger = get_logger("services.observation")


class ObservationService:
    """Manages observation lifecycle.

    Handles creation, storage, retrieval, update, and deletion
    of observations in the memory system.

    Usage:
        service = ObservationService(db, chat_provider, config, session_service)
        obs = await service.add_observation(
            content="Important finding",
            obs_type="discovery",
            project="my-project"
        )
    """

    def __init__(
        self,
        db: "DatabaseManager",
        chat_provider: Any,
        config: "AppConfig",
        session_service: "SessionService",
    ):
        """Initialize observation service.

        Args:
            db: Database manager
            chat_provider: Provider for LLM operations (summarization)
            config: Application configuration
            session_service: Session management service
        """
        self.db = db
        self.chat_provider = chat_provider
        self.config = config
        self.session_service = session_service

    async def add_observation(
        self,
        content: str,
        obs_type: str = "note",
        project: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        summarize: bool = True,
        dedupe: bool = True,
        summary: Optional[str] = None,
        created_at: Optional[float] = None,
        importance_score: float = 0.5,
        assets: Optional[List[Dict[str, Any]]] = None,
        concept: Optional[str] = None,
        event_id: Optional[str] = None,
        host: Optional[str] = None,
    ) -> Optional["Observation"]:
        """Add a new observation.

        Args:
            content: Observation content
            obs_type: Type (note, bug, feature, etc.)
            project: Project path/identifier
            session_id: Session to associate with
            tags: List of tags
            metadata: Additional metadata
            title: Optional title
            summarize: Whether to generate summary
            dedupe: Whether to check for duplicates
            summary: Pre-computed summary
            created_at: Custom creation timestamp
            importance_score: Importance (0-1)
            assets: Associated assets
            concept: Concept type
            event_id: Idempotency key
            host: Source host identifier

        Returns:
            Created Observation or None
        """
        if not content or not content.strip():
            logger.warning("Attempted to add empty observation")
            return None

        content = content.strip()
        project = project or self.config.default_project

        # Check idempotency
        if event_id:
            existing = await self.db.get_observation_by_event_id(event_id)
            if existing:
                logger.debug(f"Duplicate event_id: {event_id}")
                return existing

        # Compute content hash for deduplication
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Check for duplicates
        if dedupe:
            existing = await self.db.get_observation_by_hash(content_hash, project)
            if existing:
                logger.debug(f"Duplicate content hash: {content_hash[:16]}")
                return existing

        # Ensure session
        if not session_id:
            session = await self.session_service.ensure_session(project)
            session_id = session["id"]

        # Generate summary if needed
        summary_text = summary
        if summary_text is None and len(content) > 500:
            if summarize and self.chat_provider:
                try:
                    summary_text = await self.chat_provider.summarize(content)
                except Exception as e:
                    logger.debug(f"Summarization failed: {type(e).__name__}")
                    summary_text = content[:500] + "..."
            else:
                summary_text = content[:500] + "..."
        elif summary_text is None:
            summary_text = content

        # Create observation
        obs_id = str(uuid.uuid4())
        timestamp = created_at or time.time()

        observation = await self.db.add_observation(
            obs_id=obs_id,
            session_id=session_id,
            project=project,
            obs_type=obs_type,
            concept=concept,
            title=title,
            content=content,
            summary=summary_text,
            content_hash=content_hash,
            created_at=timestamp,
            importance_score=importance_score,
            tags=tags or [],
            metadata=metadata or {},
        )

        # Store assets
        if assets:
            await self._store_assets(obs_id, assets)

        # Record event for idempotency
        if event_id:
            await self.db.record_event(event_id, obs_id, host)

        logger.info(f"Added observation: {obs_id} ({obs_type})")
        return observation

    async def add_batch(
        self,
        observations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Add multiple observations in batch.

        Args:
            observations: List of observation dictionaries

        Returns:
            Statistics about the batch operation
        """
        added = 0
        skipped = 0
        failed = 0
        ids = []

        for obs_data in observations:
            try:
                obs = await self.add_observation(**obs_data)
                if obs:
                    added += 1
                    ids.append(obs.id)
                else:
                    skipped += 1
            except Exception as e:
                logger.error(f"Failed to add observation: {e}")
                failed += 1

        return {
            "added": added,
            "skipped": skipped,
            "failed": failed,
            "ids": ids,
        }

    async def get_observation(self, obs_id: str) -> Optional["Observation"]:
        """Get observation by ID.

        Args:
            obs_id: Observation ID

        Returns:
            Observation or None
        """
        return await self.db.get_observation(obs_id)

    async def get_observations(self, ids: List[str]) -> List["Observation"]:
        """Get multiple observations by ID.

        Args:
            ids: List of observation IDs

        Returns:
            List of found observations
        """
        return await self.db.get_observations(ids)

    async def delete_observation(
        self,
        obs_id: str,
        vector_store: Optional[Any] = None,
    ) -> bool:
        """Delete an observation.

        Args:
            obs_id: Observation ID to delete
            vector_store: Optional vector store to remove from

        Returns:
            True if deleted
        """
        # Delete from vector store
        if vector_store:
            try:
                await vector_store.delete(obs_id)
            except Exception as e:
                logger.warning(f"Failed to delete from vector store: {e}")

        # Delete from database (cascades to assets)
        deleted = await self.db.delete_observation(obs_id)

        if deleted:
            logger.info(f"Deleted observation: {obs_id}")
        return deleted

    async def delete_batch(
        self,
        obs_ids: List[str],
        vector_store: Optional[Any] = None,
    ) -> int:
        """Delete multiple observations.

        Args:
            obs_ids: List of observation IDs
            vector_store: Optional vector store

        Returns:
            Number of observations deleted
        """
        deleted = 0
        for obs_id in obs_ids:
            if await self.delete_observation(obs_id, vector_store):
                deleted += 1
        return deleted

    async def export_observations(
        self,
        project: Optional[str] = None,
        session_id: Optional[str] = None,
        obs_type: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        tag_filters: Optional[List[str]] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Export observations with filters.

        Args:
            project: Filter by project
            session_id: Filter by session
            obs_type: Filter by type
            date_start: Filter by date range start
            date_end: Filter by date range end
            tag_filters: Filter by tags
            limit: Maximum observations

        Returns:
            List of observation dictionaries
        """
        return await self.db.export_observations(
            project=project,
            session_id=session_id,
            obs_type=obs_type,
            date_start=date_start,
            date_end=date_end,
            tag_filters=tag_filters,
            limit=limit,
        )

    async def update_tags(
        self,
        obs_id: str,
        tags: List[str],
    ) -> bool:
        """Update observation tags.

        Args:
            obs_id: Observation ID
            tags: New tags list

        Returns:
            True if updated
        """
        return await self.db.update_observation_tags(obs_id, tags)

    async def _store_assets(
        self,
        observation_id: str,
        assets: List[Dict[str, Any]],
    ) -> None:
        """Store assets for an observation.

        Args:
            observation_id: Parent observation ID
            assets: Asset dictionaries
        """
        normalized = []
        for asset in assets:
            normalized.append({
                "type": asset.get("type", "file"),
                "name": asset.get("name", "unknown"),
                "path": asset.get("path"),
                "content": asset.get("content"),
                "mime_type": asset.get("mime_type"),
                "size_bytes": asset.get("size_bytes"),
                "metadata": asset.get("metadata", {}),
            })

        await self.db.store_assets(observation_id, normalized)

    async def get_observation_count(
        self,
        project: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Get count of observations.

        Args:
            project: Filter by project
            session_id: Filter by session

        Returns:
            Number of observations
        """
        return await self.db.count_observations(
            project=project,
            session_id=session_id,
        )

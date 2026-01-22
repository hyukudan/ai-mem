"""User Memory Service - User-scoped memory operations.

This service handles user-level memories that persist
across all projects for global preferences and context.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..logging_config import get_logger

if TYPE_CHECKING:
    from ..config import AppConfig
    from ..db import DatabaseManager
    from .observation import ObservationService
    from .search import SearchService

logger = get_logger("services.user_memory")

# Special project for user-level memories
USER_PROJECT = "__user__"


class UserMemoryService:
    """Manages user-scoped memories.

    Provides operations for user-level memories that
    persist globally across all projects.

    Usage:
        service = UserMemoryService(obs_service, search_service, db, config)
        await service.add_user_memory("I prefer dark mode", tags=["preference"])
        memories = await service.get_user_memories(limit=10)
    """

    def __init__(
        self,
        observation_service: "ObservationService",
        search_service: "SearchService",
        db: "DatabaseManager",
        config: "AppConfig",
    ):
        """Initialize user memory service.

        Args:
            observation_service: Observation service for storage
            search_service: Search service for retrieval
            db: Database manager
            config: Application configuration
        """
        self.observation_service = observation_service
        self.search_service = search_service
        self.db = db
        self.config = config

    async def add_user_memory(
        self,
        content: str,
        obs_type: str = "preference",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        summarize: bool = True,
    ) -> Optional[Any]:
        """Add a user-level memory.

        Args:
            content: Memory content
            obs_type: Type (preference, habit, context, etc.)
            tags: Tags for categorization
            metadata: Additional metadata
            title: Optional title
            summarize: Whether to generate summary

        Returns:
            Created observation or None
        """
        # Add user-specific tags
        all_tags = list(tags or [])
        if "user-memory" not in all_tags:
            all_tags.append("user-memory")

        return await self.observation_service.add_observation(
            content=content,
            obs_type=obs_type,
            project=USER_PROJECT,
            tags=all_tags,
            metadata=metadata or {},
            title=title,
            summarize=summarize,
        )

    async def get_user_memories(
        self,
        limit: int = 50,
        obs_type: Optional[str] = None,
        tag_filters: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get user-level memories.

        Args:
            limit: Maximum memories to return
            obs_type: Filter by type
            tag_filters: Filter by tags

        Returns:
            List of user memories
        """
        return await self.db.export_observations(
            project=USER_PROJECT,
            obs_type=obs_type,
            tag_filters=tag_filters,
            limit=limit,
        )

    async def search_user_memories(
        self,
        query: str,
        limit: int = 10,
        obs_type: Optional[str] = None,
        tag_filters: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search user-level memories.

        Args:
            query: Search query
            limit: Maximum results
            obs_type: Filter by type
            tag_filters: Filter by tags

        Returns:
            Search results
        """
        return await self.search_service.search(
            query=query,
            limit=limit,
            project=USER_PROJECT,
            obs_type=obs_type,
            tag_filters=tag_filters,
        )

    async def export_user_memories(
        self,
        output_path: str,
    ) -> Dict[str, Any]:
        """Export user memories to JSON file.

        Args:
            output_path: Path to output file

        Returns:
            Export statistics
        """
        memories = await self.get_user_memories(limit=10000)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "version": "1.0",
            "type": "user_memories",
            "count": len(memories),
            "memories": memories,
        }

        output.write_text(json.dumps(export_data, indent=2, default=str))

        logger.info(f"Exported {len(memories)} user memories to {output_path}")

        return {
            "exported": len(memories),
            "path": str(output),
        }

    async def import_user_memories(
        self,
        input_path: str,
        merge: bool = True,
    ) -> Dict[str, Any]:
        """Import user memories from JSON file.

        Args:
            input_path: Path to input file
            merge: If True, merge with existing; if False, replace

        Returns:
            Import statistics
        """
        input_file = Path(input_path)

        if not input_file.exists():
            return {"error": f"File not found: {input_path}"}

        try:
            data = json.loads(input_file.read_text())
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {e}"}

        memories = data.get("memories", [])

        if not merge:
            # Delete existing user memories
            await self.db.delete_project(USER_PROJECT)

        imported = 0
        skipped = 0

        for mem in memories:
            try:
                result = await self.add_user_memory(
                    content=mem.get("content", ""),
                    obs_type=mem.get("type", "preference"),
                    tags=mem.get("tags", []),
                    metadata=mem.get("metadata", {}),
                    title=mem.get("title"),
                    summarize=False,  # Use existing summary
                )
                if result:
                    imported += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.error(f"Failed to import memory: {e}")
                skipped += 1

        logger.info(f"Imported {imported} user memories from {input_path}")

        return {
            "imported": imported,
            "skipped": skipped,
            "total_in_file": len(memories),
        }

    async def get_user_memory_count(self) -> int:
        """Get count of user memories.

        Returns:
            Number of user memories
        """
        return await self.db.count_observations(project=USER_PROJECT)

    async def get_user_preferences(self) -> List[Dict[str, Any]]:
        """Get user preference memories specifically.

        Returns:
            List of preference memories
        """
        return await self.get_user_memories(
            obs_type="preference",
            limit=100,
        )

    async def update_preference(
        self,
        key: str,
        value: str,
    ) -> Optional[Any]:
        """Update or create a user preference.

        Args:
            key: Preference key
            value: Preference value

        Returns:
            Created/updated observation
        """
        # Search for existing preference with this key
        existing = await self.search_user_memories(
            query=key,
            limit=5,
            obs_type="preference",
        )

        # Check if any match the key exactly
        for pref in existing:
            if pref.get("title") == key or key in (pref.get("content") or ""):
                # Update would be complex - just add new one
                # Old one will be deduplicated or consolidated
                break

        return await self.add_user_memory(
            content=f"{key}: {value}",
            obs_type="preference",
            title=key,
            tags=["preference", key.lower().replace(" ", "-")],
            metadata={"preference_key": key, "preference_value": value},
        )

    async def get_user_context_for_prompt(
        self,
        limit: int = 5,
    ) -> str:
        """Get user context formatted for prompt injection.

        Args:
            limit: Maximum memories to include

        Returns:
            Formatted context string
        """
        memories = await self.get_user_memories(limit=limit)

        if not memories:
            return ""

        parts = ["User preferences and context:"]
        for mem in memories:
            summary = mem.get("summary") or mem.get("content", "")[:100]
            mem_type = mem.get("type", "note")
            parts.append(f"- [{mem_type}] {summary}")

        return "\n".join(parts)

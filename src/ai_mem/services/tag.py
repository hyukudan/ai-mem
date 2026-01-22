"""Tag Service - Manage observation tags.

This service handles tag operations including listing,
adding, renaming, and deleting tags.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..logging_config import get_logger

if TYPE_CHECKING:
    from ..db import DatabaseManager

logger = get_logger("services.tag")


class TagService:
    """Manages observation tags.

    Provides operations for working with tags across
    observations in the memory system.

    Usage:
        service = TagService(db)
        tags = await service.list_tags(project="my-project")
        await service.add_tag("important", project="my-project")
        await service.rename_tag("old-name", "new-name")
    """

    def __init__(self, db: "DatabaseManager"):
        """Initialize tag service.

        Args:
            db: Database manager
        """
        self.db = db

    async def list_tags(
        self,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        tag_filters: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List tags with frequency counts.

        Args:
            project: Filter by project
            obs_type: Filter by observation type
            session_id: Filter by session
            date_start: Filter by date range start
            date_end: Filter by date range end
            tag_filters: Filter to specific tags
            limit: Maximum tags to return

        Returns:
            List of {tag, count} dictionaries
        """
        return await self.db.get_tag_counts(
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=date_start,
            date_end=date_end,
            tag_filters=tag_filters,
            limit=limit,
        )

    async def add_tag(
        self,
        tag: str,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        tag_filters: Optional[List[str]] = None,
    ) -> int:
        """Add a tag to matching observations.

        Args:
            tag: Tag to add
            project: Filter by project
            obs_type: Filter by observation type
            session_id: Filter by session
            date_start: Filter by date range start
            date_end: Filter by date range end
            tag_filters: Filter to observations with these tags

        Returns:
            Number of observations updated
        """
        if not tag or not tag.strip():
            logger.warning("Attempted to add empty tag")
            return 0

        tag = tag.strip()
        updated = await self.db.add_tag(
            tag=tag,
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=date_start,
            date_end=date_end,
            tag_filters=tag_filters,
        )

        logger.info(f"Added tag '{tag}' to {updated} observations")
        return updated

    async def rename_tag(
        self,
        old_tag: str,
        new_tag: str,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        tag_filters: Optional[List[str]] = None,
    ) -> int:
        """Rename a tag across matching observations.

        Args:
            old_tag: Current tag name
            new_tag: New tag name
            project: Filter by project
            obs_type: Filter by observation type
            session_id: Filter by session
            date_start: Filter by date range start
            date_end: Filter by date range end
            tag_filters: Filter to observations with these tags

        Returns:
            Number of observations updated
        """
        if not old_tag or not new_tag:
            logger.warning("Attempted to rename with empty tag")
            return 0

        old_tag = old_tag.strip()
        new_tag = new_tag.strip()

        if old_tag == new_tag:
            logger.debug(f"Tag rename no-op: '{old_tag}' == '{new_tag}'")
            return 0

        updated = await self.db.replace_tag(
            old_tag=old_tag,
            new_tag=new_tag,
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=date_start,
            date_end=date_end,
            tag_filters=tag_filters,
        )

        logger.info(f"Renamed tag '{old_tag}' -> '{new_tag}' in {updated} observations")
        return updated

    async def delete_tag(
        self,
        tag: str,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        tag_filters: Optional[List[str]] = None,
    ) -> int:
        """Delete a tag from matching observations.

        Args:
            tag: Tag to delete
            project: Filter by project
            obs_type: Filter by observation type
            session_id: Filter by session
            date_start: Filter by date range start
            date_end: Filter by date range end
            tag_filters: Filter to observations with these tags

        Returns:
            Number of observations updated
        """
        if not tag:
            logger.warning("Attempted to delete empty tag")
            return 0

        tag = tag.strip()
        updated = await self.db.delete_tag(
            tag=tag,
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=date_start,
            date_end=date_end,
            tag_filters=tag_filters,
        )

        logger.info(f"Deleted tag '{tag}' from {updated} observations")
        return updated

    async def get_tag_cloud(
        self,
        project: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get a tag cloud with normalized weights.

        Args:
            project: Filter by project
            limit: Maximum tags

        Returns:
            List of {tag, count, weight} dictionaries
        """
        tags = await self.list_tags(project=project, limit=limit)

        if not tags:
            return []

        max_count = max(t["count"] for t in tags)
        min_count = min(t["count"] for t in tags)
        range_count = max_count - min_count or 1

        return [
            {
                "tag": t["tag"],
                "count": t["count"],
                "weight": (t["count"] - min_count) / range_count,
            }
            for t in tags
        ]

    async def merge_tags(
        self,
        source_tags: List[str],
        target_tag: str,
        project: Optional[str] = None,
    ) -> int:
        """Merge multiple tags into one.

        Args:
            source_tags: Tags to merge from
            target_tag: Tag to merge into
            project: Filter by project

        Returns:
            Total observations updated
        """
        total_updated = 0

        for source in source_tags:
            if source != target_tag:
                updated = await self.rename_tag(
                    old_tag=source,
                    new_tag=target_tag,
                    project=project,
                )
                total_updated += updated

        logger.info(f"Merged {len(source_tags)} tags into '{target_tag}': {total_updated} updates")
        return total_updated

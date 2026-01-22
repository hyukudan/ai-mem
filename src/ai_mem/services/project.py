"""Project Service - Manage projects and statistics.

This service handles project-level operations including
listing, deletion, and statistics computation.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..logging_config import get_logger

if TYPE_CHECKING:
    from ..config import AppConfig
    from ..db import DatabaseManager
    from .observation import ObservationService

logger = get_logger("services.project")


class ProjectService:
    """Manages project operations.

    Handles project listing, deletion, and statistics
    computation for memory analytics.

    Usage:
        service = ProjectService(db, vector_store, chat_provider, obs_service, config)
        projects = await service.list_projects()
        stats = await service.get_stats(project="my-project")
    """

    def __init__(
        self,
        db: "DatabaseManager",
        vector_store: Any,
        chat_provider: Any,
        observation_service: "ObservationService",
        config: "AppConfig",
    ):
        """Initialize project service.

        Args:
            db: Database manager
            vector_store: Vector store for cleanup
            chat_provider: Provider for summarization
            observation_service: Observation service for creating summaries
            config: Application configuration
        """
        self.db = db
        self.vector_store = vector_store
        self.chat_provider = chat_provider
        self.observation_service = observation_service
        self.config = config

    async def list_projects(self) -> List[str]:
        """List all projects.

        Returns:
            List of project identifiers
        """
        return await self.db.list_projects()

    async def delete_project(self, project: str) -> Dict[str, Any]:
        """Delete a project and all its data.

        Args:
            project: Project to delete

        Returns:
            Deletion statistics
        """
        # Delete from vector store
        vector_deleted = 0
        if self.vector_store:
            try:
                vector_deleted = await self.vector_store.delete_by_metadata(
                    {"project": project}
                )
            except Exception as e:
                logger.warning(f"Failed to delete vectors for project: {e}")

        # Delete from database
        db_deleted = await self.db.delete_project(project)

        logger.info(f"Deleted project: {project} ({db_deleted} observations, {vector_deleted} vectors)")

        return {
            "project": project,
            "observations_deleted": db_deleted,
            "vectors_deleted": vector_deleted,
        }

    async def get_stats(
        self,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        since: Optional[float] = None,
        tag_filters: Optional[List[str]] = None,
        tag_limit: int = 20,
        day_limit: int = 30,
        type_tag_limit: int = 5,
    ) -> Dict[str, Any]:
        """Get observation statistics.

        Args:
            project: Filter by project
            obs_type: Filter by observation type
            session_id: Filter by session
            date_start: Filter by date range start
            date_end: Filter by date range end
            since: Filter to observations after this timestamp
            tag_filters: Filter by tags
            tag_limit: Maximum tags to return
            day_limit: Days for daily breakdown
            type_tag_limit: Tags per type

        Returns:
            Statistics dictionary
        """
        if since and not date_start:
            date_start = since

        return await self.db.get_stats(
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=date_start,
            date_end=date_end,
            tag_filters=tag_filters,
            tag_limit=tag_limit,
            day_limit=day_limit,
            type_tag_limit=type_tag_limit,
        )

    async def summarize_project(
        self,
        project: str,
        session_id: Optional[str] = None,
        limit: int = 50,
        obs_type: Optional[str] = None,
        store: bool = True,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate an AI summary of a project's observations.

        Args:
            project: Project to summarize
            session_id: Optional session filter
            limit: Maximum observations to include
            obs_type: Optional type filter
            store: Whether to store summary as observation
            tags: Tags for stored summary

        Returns:
            Summary result with text and metadata
        """
        if not self.chat_provider:
            return {"error": "No chat provider available for summarization"}

        # Get recent observations
        observations = await self.db.export_observations(
            project=project,
            session_id=session_id,
            obs_type=obs_type,
            limit=limit,
        )

        if not observations:
            return {
                "project": project,
                "summary": "No observations found for this project.",
                "observation_count": 0,
            }

        # Build context for summarization
        context_parts = []
        for obs in observations:
            summary = obs.get("summary") or obs.get("content", "")[:200]
            obs_type_str = obs.get("type", "note")
            context_parts.append(f"[{obs_type_str}] {summary}")

        context = "\n".join(context_parts)

        # Generate summary
        prompt = f"""Summarize the following project observations into a concise overview.
Focus on key findings, decisions, and patterns.

Observations:
{context}

Provide a structured summary with:
1. Main themes and topics
2. Key decisions or discoveries
3. Notable patterns or issues"""

        try:
            summary_text = await self.chat_provider.complete(prompt)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return {"error": f"Summarization failed: {e}"}

        result = {
            "project": project,
            "summary": summary_text,
            "observation_count": len(observations),
        }

        # Store summary as observation
        if store:
            summary_obs = await self.observation_service.add_observation(
                content=summary_text,
                obs_type="summary",
                project=project,
                title=f"Project Summary: {project}",
                tags=tags or ["auto-summary", "project-summary"],
                metadata={
                    "source_count": len(observations),
                    "session_id": session_id,
                },
            )
            if summary_obs:
                result["stored_as"] = summary_obs.id

        return result

    async def get_project_overview(self, project: str) -> Dict[str, Any]:
        """Get a comprehensive project overview.

        Args:
            project: Project identifier

        Returns:
            Overview with stats, recent activity, and top tags
        """
        stats = await self.get_stats(project=project, tag_limit=10, day_limit=7)

        # Get recent sessions
        sessions = await self.db.list_sessions(
            project=project,
            limit=5,
        )

        return {
            "project": project,
            "total_observations": stats.get("total", 0),
            "type_breakdown": stats.get("by_type", {}),
            "top_tags": stats.get("top_tags", []),
            "daily_activity": stats.get("daily", []),
            "recent_sessions": [
                {
                    "id": s.get("id"),
                    "goal": s.get("goal"),
                    "start_time": s.get("start_time"),
                    "end_time": s.get("end_time"),
                }
                for s in sessions
            ],
        }

    async def compare_projects(
        self,
        projects: List[str],
    ) -> Dict[str, Any]:
        """Compare statistics across multiple projects.

        Args:
            projects: List of project identifiers

        Returns:
            Comparison data
        """
        comparisons = {}

        for project in projects:
            stats = await self.get_stats(project=project, tag_limit=5)
            comparisons[project] = {
                "total": stats.get("total", 0),
                "by_type": stats.get("by_type", {}),
                "top_tags": [t["tag"] for t in stats.get("top_tags", [])[:5]],
            }

        return {
            "projects": comparisons,
            "total_observations": sum(
                p["total"] for p in comparisons.values()
            ),
        }

"""Session Service - Manage session lifecycle.

This service handles creation, tracking, and closure of
memory sessions within projects.
"""

import time
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..logging_config import get_logger

if TYPE_CHECKING:
    from ..config import AppConfig
    from ..db import DatabaseManager

logger = get_logger("services.session")


class SessionService:
    """Manages session lifecycle.

    Handles creation, update, and closure of sessions
    that group related observations together.

    Usage:
        service = SessionService(db, config)
        session = await service.start_session("my-project", "Fix bugs")
        # ... add observations ...
        await service.end_session(session["id"])
    """

    def __init__(self, db: "DatabaseManager", config: "AppConfig"):
        """Initialize session service.

        Args:
            db: Database manager
            config: Application configuration
        """
        self.db = db
        self.config = config
        self._current_session: Optional[Dict[str, Any]] = None

    @property
    def current_session(self) -> Optional[Dict[str, Any]]:
        """Get current active session."""
        return self._current_session

    async def start_session(
        self,
        project: str,
        goal: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start a new session or resume an existing one.

        Args:
            project: Project path or identifier
            goal: Optional session goal description
            session_id: Optional specific session ID to use/resume

        Returns:
            Session dictionary with id, project, goal, start_time
        """
        if session_id:
            # Try to resume existing session
            existing = await self.get_session(session_id)
            if existing and existing.get("end_time") is None:
                self._current_session = existing
                logger.info(f"Resumed session: {session_id}")
                return existing

        # Create new session
        new_id = session_id or str(uuid.uuid4())
        session = {
            "id": new_id,
            "project": project,
            "goal": goal,
            "start_time": time.time(),
            "end_time": None,
            "summary": None,
        }

        await self.db.add_session(
            session_id=new_id,
            project=project,
            goal=goal,
            start_time=session["start_time"],
        )

        self._current_session = session
        logger.info(f"Started session: {new_id} for project: {project}")
        return session

    async def ensure_session(self, project: str) -> Dict[str, Any]:
        """Get or create a session for a project.

        Args:
            project: Project path or identifier

        Returns:
            Active session for the project
        """
        if self._current_session and self._current_session.get("project") == project:
            return self._current_session

        # Check for active session in DB
        sessions = await self.list_sessions(project=project, active_only=True, limit=1)
        if sessions:
            self._current_session = sessions[0]
            return self._current_session

        # Create new session
        return await self.start_session(project)

    async def end_session(
        self,
        session_id: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """End a session.

        Args:
            session_id: Session to end (current if not specified)
            summary: Optional session summary

        Returns:
            Ended session or None if not found
        """
        target_id = session_id or (self._current_session.get("id") if self._current_session else None)

        if not target_id:
            logger.warning("No session to end")
            return None

        end_time = time.time()
        await self.db.end_session(target_id, end_time, summary)

        if self._current_session and self._current_session.get("id") == target_id:
            self._current_session["end_time"] = end_time
            self._current_session["summary"] = summary
            ended = self._current_session
            self._current_session = None
            logger.info(f"Ended session: {target_id}")
            return ended

        # Fetch from DB
        return await self.get_session(target_id)

    async def close_current(self) -> Optional[Dict[str, Any]]:
        """Close current session without ending it in DB.

        Returns:
            Previously current session or None
        """
        if self._current_session:
            session = self._current_session
            self._current_session = None
            logger.debug(f"Closed current session reference: {session.get('id')}")
            return session
        return None

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session dictionary or None
        """
        return await self.db.get_session(session_id)

    async def list_sessions(
        self,
        project: Optional[str] = None,
        active_only: bool = False,
        goal_query: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List sessions with optional filters.

        Args:
            project: Filter by project
            active_only: Only return sessions without end_time
            goal_query: Filter by goal text
            date_start: Filter by start time >= this
            date_end: Filter by start time <= this
            limit: Maximum results

        Returns:
            List of session dictionaries
        """
        return await self.db.list_sessions(
            project=project,
            active_only=active_only,
            goal_query=goal_query,
            date_start=date_start,
            date_end=date_end,
            limit=limit,
        )

    async def end_latest_session(
        self,
        project: str,
        summary: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """End the most recent active session for a project.

        Args:
            project: Project to find session for
            summary: Optional session summary

        Returns:
            Ended session or None
        """
        sessions = await self.list_sessions(project=project, active_only=True, limit=1)
        if sessions:
            return await self.end_session(sessions[0]["id"], summary)
        return None

    def get_session_info(self) -> Dict[str, Any]:
        """Get information about current session state.

        Returns:
            Dictionary with session status
        """
        if self._current_session:
            duration = time.time() - self._current_session.get("start_time", 0)
            return {
                "has_session": True,
                "session_id": self._current_session.get("id"),
                "project": self._current_session.get("project"),
                "goal": self._current_session.get("goal"),
                "duration_seconds": duration,
            }
        return {"has_session": False}

"""Resumable Sessions - Continue sessions across conversations.

This module provides the ability to resume sessions between conversations,
maintaining context and state across disconnections.

Features:
- Session state persistence
- Resume from session ID
- Cross-conversation continuity
- State recovery after failures

Configuration:
    AI_MEM_SESSIONS_PERSIST=true
    AI_MEM_SESSIONS_DIR=~/.ai-mem/sessions
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logging_config import get_logger

logger = get_logger("resumable_sessions")

# Configuration
SESSIONS_PERSIST = os.environ.get("AI_MEM_SESSIONS_PERSIST", "true").lower() in ("true", "1", "yes")
SESSIONS_DIR = Path(os.environ.get("AI_MEM_SESSIONS_DIR", "~/.ai-mem/sessions")).expanduser()


@dataclass
class SessionState:
    """Persistent state for a session."""

    session_id: str
    project: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    goal: Optional[str] = None
    status: str = "active"  # active, paused, completed, failed

    # Context state
    observation_count: int = 0
    last_observation_id: Optional[str] = None
    context_tokens_used: int = 0

    # Mode state
    mode: Optional[str] = None
    endless_enabled: bool = False

    # Custom state (for SDK extensions)
    custom_data: Dict[str, Any] = field(default_factory=dict)

    # Checkpoint data
    checkpoint_data: Optional[Dict[str, Any]] = None
    checkpoint_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SessionCheckpoint:
    """Checkpoint for session recovery."""

    session_id: str
    timestamp: float
    observation_ids: List[str]
    context_summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionPersistence:
    """Handles session state persistence.

    Usage:
        persistence = SessionPersistence()

        # Save session state
        state = SessionState(session_id="sess-123", project="/path")
        persistence.save(state)

        # Load session state
        state = persistence.load("sess-123")

        # List sessions
        sessions = persistence.list_sessions(project="/path")
    """

    def __init__(self, sessions_dir: Optional[Path] = None):
        """Initialize persistence.

        Args:
            sessions_dir: Directory for session files
        """
        self.sessions_dir = sessions_dir or SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_file(self, session_id: str) -> Path:
        """Get path to session state file."""
        return self.sessions_dir / f"{session_id}.json"

    def save(self, state: SessionState) -> None:
        """Save session state.

        Args:
            state: Session state to save
        """
        if not SESSIONS_PERSIST:
            return

        state.updated_at = time.time()
        path = self._session_file(state.session_id)

        try:
            path.write_text(json.dumps(state.to_dict(), indent=2))
            logger.debug(f"Saved session state: {state.session_id}")
        except Exception as e:
            logger.error(f"Failed to save session state: {e}")

    def load(self, session_id: str) -> Optional[SessionState]:
        """Load session state.

        Args:
            session_id: Session ID

        Returns:
            SessionState or None if not found
        """
        path = self._session_file(session_id)

        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            return SessionState.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load session state: {e}")
            return None

    def delete(self, session_id: str) -> bool:
        """Delete session state.

        Args:
            session_id: Session ID

        Returns:
            True if deleted
        """
        path = self._session_file(session_id)

        if path.exists():
            try:
                path.unlink()
                return True
            except Exception as e:
                logger.error(f"Failed to delete session state: {e}")

        return False

    def list_sessions(
        self,
        project: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[SessionState]:
        """List saved sessions.

        Args:
            project: Filter by project
            status: Filter by status
            limit: Maximum sessions to return

        Returns:
            List of session states
        """
        sessions = []

        for path in self.sessions_dir.glob("*.json"):
            try:
                state = self.load(path.stem)
                if state is None:
                    continue

                # Apply filters
                if project and state.project != project:
                    continue
                if status and state.status != status:
                    continue

                sessions.append(state)
            except Exception:
                continue

        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions[:limit]

    def cleanup_old(self, max_age_days: int = 30) -> int:
        """Clean up old session files.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of sessions cleaned up
        """
        cutoff = time.time() - (max_age_days * 86400)
        cleaned = 0

        for path in self.sessions_dir.glob("*.json"):
            try:
                state = self.load(path.stem)
                if state and state.updated_at < cutoff:
                    if self.delete(state.session_id):
                        cleaned += 1
            except Exception:
                continue

        return cleaned


class ResumableSession:
    """A resumable session that persists state.

    Usage:
        # Create new session
        session = ResumableSession.create(project="/path", goal="Fix bugs")

        # Resume existing session
        session = ResumableSession.resume("sess-123")

        # Update state
        session.add_observation("obs-456")
        session.set_custom("last_file", "/src/main.py")

        # Create checkpoint
        session.checkpoint("Completed first phase")

        # End session
        session.complete()
    """

    def __init__(self, state: SessionState, persistence: SessionPersistence):
        """Initialize session.

        Args:
            state: Session state
            persistence: Persistence handler
        """
        self.state = state
        self.persistence = persistence

    @classmethod
    def create(
        cls,
        project: str,
        session_id: Optional[str] = None,
        goal: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> "ResumableSession":
        """Create a new resumable session.

        Args:
            project: Project path
            session_id: Optional session ID (generated if not provided)
            goal: Optional session goal
            mode: Optional work mode

        Returns:
            New ResumableSession
        """
        import uuid

        persistence = SessionPersistence()

        state = SessionState(
            session_id=session_id or str(uuid.uuid4()),
            project=project,
            goal=goal,
            mode=mode,
        )

        session = cls(state, persistence)
        session.save()

        logger.info(f"Created resumable session: {state.session_id}")
        return session

    @classmethod
    def resume(cls, session_id: str) -> Optional["ResumableSession"]:
        """Resume an existing session.

        Args:
            session_id: Session ID to resume

        Returns:
            ResumableSession or None if not found
        """
        persistence = SessionPersistence()
        state = persistence.load(session_id)

        if state is None:
            logger.warning(f"Session not found: {session_id}")
            return None

        if state.status == "completed":
            logger.warning(f"Session already completed: {session_id}")
            return None

        # Update status
        state.status = "active"
        state.updated_at = time.time()

        session = cls(state, persistence)
        session.save()

        logger.info(f"Resumed session: {session_id}")
        return session

    @property
    def session_id(self) -> str:
        """Get session ID."""
        return self.state.session_id

    @property
    def project(self) -> str:
        """Get project path."""
        return self.state.project

    def save(self) -> None:
        """Save current state."""
        self.persistence.save(self.state)

    def add_observation(self, observation_id: str) -> None:
        """Record an observation added.

        Args:
            observation_id: Observation ID
        """
        self.state.observation_count += 1
        self.state.last_observation_id = observation_id
        self.save()

    def update_context_tokens(self, tokens: int) -> None:
        """Update context token count.

        Args:
            tokens: New token count
        """
        self.state.context_tokens_used = tokens
        self.save()

    def set_mode(self, mode: str) -> None:
        """Set work mode.

        Args:
            mode: Mode name
        """
        self.state.mode = mode
        self.save()

    def set_endless(self, enabled: bool) -> None:
        """Set Endless Mode status.

        Args:
            enabled: Whether enabled
        """
        self.state.endless_enabled = enabled
        self.save()

    def set_custom(self, key: str, value: Any) -> None:
        """Set custom data.

        Args:
            key: Data key
            value: Data value
        """
        self.state.custom_data[key] = value
        self.save()

    def get_custom(self, key: str, default: Any = None) -> Any:
        """Get custom data.

        Args:
            key: Data key
            default: Default value

        Returns:
            Stored value or default
        """
        return self.state.custom_data.get(key, default)

    def checkpoint(self, summary: Optional[str] = None) -> SessionCheckpoint:
        """Create a checkpoint.

        Args:
            summary: Optional summary of current state

        Returns:
            Checkpoint object
        """
        checkpoint = SessionCheckpoint(
            session_id=self.state.session_id,
            timestamp=time.time(),
            observation_ids=[self.state.last_observation_id] if self.state.last_observation_id else [],
            context_summary=summary,
            metadata={
                "observation_count": self.state.observation_count,
                "mode": self.state.mode,
            },
        )

        self.state.checkpoint_data = asdict(checkpoint)
        self.state.checkpoint_at = checkpoint.timestamp
        self.save()

        logger.info(f"Created checkpoint for session: {self.state.session_id}")
        return checkpoint

    def get_last_checkpoint(self) -> Optional[SessionCheckpoint]:
        """Get the last checkpoint.

        Returns:
            Last checkpoint or None
        """
        if not self.state.checkpoint_data:
            return None

        return SessionCheckpoint(**self.state.checkpoint_data)

    def pause(self) -> None:
        """Pause the session."""
        self.state.status = "paused"
        self.save()
        logger.info(f"Paused session: {self.state.session_id}")

    def complete(self) -> None:
        """Mark session as completed."""
        self.state.status = "completed"
        self.save()
        logger.info(f"Completed session: {self.state.session_id}")

    def fail(self, reason: Optional[str] = None) -> None:
        """Mark session as failed.

        Args:
            reason: Optional failure reason
        """
        self.state.status = "failed"
        if reason:
            self.state.custom_data["failure_reason"] = reason
        self.save()
        logger.info(f"Failed session: {self.state.session_id} - {reason}")

    def get_summary(self) -> Dict[str, Any]:
        """Get session summary.

        Returns:
            Summary dictionary
        """
        return {
            "session_id": self.state.session_id,
            "project": self.state.project,
            "goal": self.state.goal,
            "status": self.state.status,
            "observation_count": self.state.observation_count,
            "mode": self.state.mode,
            "endless_enabled": self.state.endless_enabled,
            "created_at": datetime.fromtimestamp(self.state.created_at).isoformat(),
            "updated_at": datetime.fromtimestamp(self.state.updated_at).isoformat(),
            "has_checkpoint": self.state.checkpoint_data is not None,
        }


# Convenience functions

def create_session(project: str, **kwargs) -> ResumableSession:
    """Create a new resumable session.

    Args:
        project: Project path
        **kwargs: Additional arguments

    Returns:
        New session
    """
    return ResumableSession.create(project, **kwargs)


def resume_session(session_id: str) -> Optional[ResumableSession]:
    """Resume a session by ID.

    Args:
        session_id: Session ID

    Returns:
        Resumed session or None
    """
    return ResumableSession.resume(session_id)


def list_resumable_sessions(
    project: Optional[str] = None,
    active_only: bool = True,
) -> List[SessionState]:
    """List resumable sessions.

    Args:
        project: Filter by project
        active_only: Only show active/paused sessions

    Returns:
        List of session states
    """
    persistence = SessionPersistence()
    sessions = persistence.list_sessions(project=project)

    if active_only:
        sessions = [s for s in sessions if s.status in ("active", "paused")]

    return sessions

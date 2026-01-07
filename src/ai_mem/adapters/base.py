"""Base adapter interface for LLM host event translation.

All host-specific adapters inherit from EventAdapter and implement
the translation from host-native formats to Event Schema v1.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..events import (
    EventContext,
    EventSource,
    PrivacyFlags,
    ToolCategory,
    ToolEvent,
    ToolExecution,
    UserPromptEvent,
    SessionEvent,
    EventType,
)


class EventAdapter(ABC):
    """Base class for host-specific event adapters.

    Adapters translate events from host-native formats (Claude hook JSON,
    Gemini callback data, etc.) into the canonical Event Schema v1.

    The adapter is responsible for:
    - Parsing host-specific field names
    - Mapping tool categories
    - Extracting context information
    - Setting appropriate privacy flags
    """

    @property
    @abstractmethod
    def host_name(self) -> str:
        """Return the canonical host name for this adapter."""
        pass

    @abstractmethod
    def parse_tool_event(self, payload: Dict[str, Any]) -> Optional[ToolEvent]:
        """Parse a raw payload into a ToolEvent.

        Args:
            payload: Raw event data from the host (e.g., hook stdin JSON)

        Returns:
            A ToolEvent if the payload contains tool data, None otherwise.
        """
        pass

    def parse_user_prompt(self, payload: Dict[str, Any]) -> Optional[UserPromptEvent]:
        """Parse a raw payload into a UserPromptEvent.

        Args:
            payload: Raw event data from the host

        Returns:
            A UserPromptEvent if the payload contains prompt data, None otherwise.
        """
        # Default implementation - override in subclasses
        content = self._extract_first(payload, ["content", "prompt", "message", "text"])
        if not content:
            return None

        return UserPromptEvent(
            content=str(content),
            session_id=self._extract_session_id(payload),
            source=EventSource(host=self.host_name),
            context=self._extract_context(payload),
            privacy=self._extract_privacy(payload),
        )

    def parse_session_event(
        self, payload: Dict[str, Any], event_type: EventType
    ) -> Optional[SessionEvent]:
        """Parse a raw payload into a SessionEvent.

        Args:
            payload: Raw event data from the host
            event_type: SESSION_START or SESSION_END

        Returns:
            A SessionEvent if valid, None otherwise.
        """
        session_id = self._extract_session_id(payload)
        if not session_id:
            return None

        return SessionEvent(
            event_type=event_type,
            session_id=session_id,
            goal=self._extract_first(payload, ["goal", "description", "objective"]),
            source=EventSource(host=self.host_name),
            context=self._extract_context(payload),
        )

    # === Helper methods for subclasses ===

    def _extract_first(
        self, payload: Dict[str, Any], keys: List[str]
    ) -> Optional[Any]:
        """Extract the first present value from a list of possible keys."""
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
        return None

    def _extract_session_id(self, payload: Dict[str, Any]) -> Optional[str]:
        """Extract session ID from payload."""
        value = self._extract_first(
            payload, ["session_id", "sessionId", "session", "conversation_id"]
        )
        return str(value) if value else None

    def _extract_project(self, payload: Dict[str, Any]) -> Optional[str]:
        """Extract project path from payload."""
        value = self._extract_first(
            payload, ["project", "cwd", "working_dir", "workspace", "root"]
        )
        return str(value) if value else None

    def _extract_context(self, payload: Dict[str, Any]) -> EventContext:
        """Extract context information from payload."""
        files = payload.get("files_touched", [])
        if isinstance(files, str):
            files = [f.strip() for f in files.split(",") if f.strip()]

        return EventContext(
            cwd=self._extract_first(payload, ["cwd", "working_dir", "pwd"]),
            project=self._extract_project(payload),
            files_touched=files if isinstance(files, list) else [],
        )

    def _extract_privacy(self, payload: Dict[str, Any]) -> PrivacyFlags:
        """Extract privacy flags from payload."""
        privacy_data = payload.get("privacy", {})
        if isinstance(privacy_data, dict):
            return PrivacyFlags(
                redact=privacy_data.get("redact", True),
                fully_private=privacy_data.get("fully_private", False),
                strip_tags=privacy_data.get("strip_tags", []),
            )
        return PrivacyFlags()

    def _map_tool_category(self, tool_name: str) -> Optional[ToolCategory]:
        """Map a tool name to a category (best effort).

        Override in subclasses for host-specific mappings.
        """
        name_lower = tool_name.lower()

        # Filesystem tools
        if any(x in name_lower for x in ["read", "write", "file", "glob", "ls"]):
            return ToolCategory.FILESYSTEM

        # Shell tools
        if any(x in name_lower for x in ["bash", "shell", "exec", "run", "command"]):
            return ToolCategory.SHELL

        # Network tools
        if any(x in name_lower for x in ["http", "fetch", "curl", "request", "api"]):
            return ToolCategory.NETWORK

        # Search tools
        if any(x in name_lower for x in ["search", "grep", "find", "query"]):
            return ToolCategory.SEARCH

        # Meta tools (about the agent itself)
        if any(x in name_lower for x in ["todo", "plan", "skill", "slash"]):
            return ToolCategory.META

        return ToolCategory.CUSTOM

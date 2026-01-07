"""Cursor IDE adapter.

Translates Cursor AI tool call events to Event Schema v1.

Cursor AI typically uses a format similar to OpenAI function calls:
- name/function_name for tool name
- arguments/parameters for input
- output/result for response
"""

from typing import Any, Dict, Optional

from ..events import (
    EventContext,
    EventSource,
    ToolCategory,
    ToolEvent,
    ToolExecution,
)
from ..logging_config import get_logger
from .base import EventAdapter

logger = get_logger("adapters.cursor")


class CursorAdapter(EventAdapter):
    """Adapter for Cursor IDE AI events."""

    @property
    def host_name(self) -> str:
        return "cursor"

    def parse_tool_event(self, payload: Dict[str, Any]) -> Optional[ToolEvent]:
        """Parse a Cursor tool call payload into a ToolEvent.

        Cursor uses OpenAI-compatible format for function/tool calls.
        """
        logger.debug("Parsing Cursor tool call")

        # Try various field names for tool name
        tool_name = self._extract_first(
            payload,
            [
                "name",
                "function_name",
                "tool_name",
                "function",
                "tool",
                "tool_call.function.name",
            ],
        )

        # Check nested tool_call structure
        tool_call = payload.get("tool_call", {})
        if isinstance(tool_call, dict):
            function = tool_call.get("function", {})
            if isinstance(function, dict):
                tool_name = tool_name or function.get("name")

        if not tool_name:
            return None

        # Extract input/arguments
        tool_input = self._extract_first(
            payload,
            ["arguments", "parameters", "args", "input", "tool_input"],
        )
        if tool_input is None and isinstance(tool_call, dict):
            function = tool_call.get("function", {})
            if isinstance(function, dict):
                tool_input = function.get("arguments")

        # Extract output/result
        tool_output = self._extract_first(
            payload,
            ["output", "result", "response", "tool_output", "content"],
        )

        # Determine success/failure
        error = self._extract_first(payload, ["error", "error_message"])
        success = error is None and not payload.get("failed", False)

        # Extract latency if available
        latency_ms = None
        latency_value = self._extract_first(payload, ["latency_ms", "duration_ms", "elapsed_ms"])
        if latency_value is not None:
            try:
                latency_ms = float(latency_value)
            except (ValueError, TypeError):
                pass

        return ToolEvent(
            session_id=self._extract_session_id(payload),
            source=EventSource(
                host=self.host_name,
                agent_id=payload.get("model") or payload.get("agent_id"),
            ),
            tool=ToolExecution(
                name=str(tool_name),
                category=self._map_tool_category(str(tool_name)),
                input=tool_input,
                output=tool_output,
                success=success,
                error=str(error) if error else None,
                latency_ms=latency_ms,
            ),
            context=self._extract_context(payload),
            privacy=self._extract_privacy(payload),
            metadata={
                "model": payload.get("model"),
                "tool_call_id": payload.get("tool_call_id") or (tool_call.get("id") if isinstance(tool_call, dict) else None),
            },
        )

    def _extract_session_id(self, payload: Dict[str, Any]) -> Optional[str]:
        """Extract session ID from Cursor payload."""
        value = self._extract_first(
            payload,
            [
                "session_id",
                "sessionId",
                "conversation_id",
                "thread_id",
                "chat_id",
                "workspace_id",
            ],
        )
        return str(value) if value else None

    def _map_tool_category(self, tool_name: str) -> Optional[ToolCategory]:
        """Map Cursor tool names to categories.

        Cursor tools often mirror IDE operations.
        """
        name = tool_name.lower()

        # Cursor-specific tools
        if any(x in name for x in ["edit", "apply", "insert", "replace", "delete"]):
            return ToolCategory.FILESYSTEM
        if any(x in name for x in ["terminal", "shell", "run", "exec"]):
            return ToolCategory.SHELL
        if any(x in name for x in ["search", "find", "grep", "ripgrep"]):
            return ToolCategory.SEARCH
        if any(x in name for x in ["read", "open", "view", "show"]):
            return ToolCategory.FILESYSTEM
        if any(x in name for x in ["create", "new", "write"]):
            return ToolCategory.FILESYSTEM
        if any(x in name for x in ["lint", "format", "diagnostic"]):
            return ToolCategory.META

        return super()._map_tool_category(tool_name)

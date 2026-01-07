"""Gemini CLI / Gemini API adapter.

Translates Gemini function call responses to Event Schema v1.

Gemini function calls typically include:
- function_call.name
- function_call.args
- function_response.response
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

logger = get_logger("adapters.gemini")


class GeminiAdapter(EventAdapter):
    """Adapter for Gemini CLI and Gemini API events."""

    @property
    def host_name(self) -> str:
        return "gemini"

    def parse_tool_event(self, payload: Dict[str, Any]) -> Optional[ToolEvent]:
        """Parse a Gemini function call payload into a ToolEvent.

        Gemini can send function calls in various formats:
        - Direct: name, args, response
        - Nested: function_call.name, function_call.args
        - Response: function_response.name, function_response.response
        """
        logger.debug("Parsing Gemini function call")
        # Try nested function_call format first
        function_call = payload.get("function_call", {})
        function_response = payload.get("function_response", {})

        tool_name = (
            function_call.get("name")
            or function_response.get("name")
            or self._extract_first(payload, ["name", "tool_name", "function_name"])
        )
        if not tool_name:
            return None

        tool_input = (
            function_call.get("args")
            or self._extract_first(payload, ["args", "arguments", "input", "parameters"])
        )
        tool_output = (
            function_response.get("response")
            or self._extract_first(payload, ["response", "result", "output", "content"])
        )

        # Gemini typically doesn't provide success/failure directly
        # Check for error fields
        error = self._extract_first(payload, ["error", "error_message"])
        success = error is None

        return ToolEvent(
            session_id=self._extract_session_id(payload),
            source=EventSource(
                host=self.host_name,
                agent_id=payload.get("model") or payload.get("model_id"),
            ),
            tool=ToolExecution(
                name=str(tool_name),
                category=self._map_tool_category(str(tool_name)),
                input=tool_input,
                output=tool_output,
                success=success,
                error=str(error) if error else None,
            ),
            context=self._extract_context(payload),
            privacy=self._extract_privacy(payload),
            metadata={
                "model": payload.get("model"),
                "generation_id": payload.get("generation_id"),
            },
        )

    def _extract_session_id(self, payload: Dict[str, Any]) -> Optional[str]:
        """Extract session ID from Gemini payload.

        Gemini may use different field names for conversation tracking.
        """
        value = self._extract_first(
            payload, [
                "session_id", "sessionId", "conversation_id",
                "chat_id", "thread_id"
            ]
        )
        return str(value) if value else None

    def _map_tool_category(self, tool_name: str) -> Optional[ToolCategory]:
        """Map Gemini function names to categories.

        Gemini functions are typically custom-defined, so we rely more
        on generic patterns.
        """
        name = tool_name.lower()

        # Google-specific tools
        if "search" in name or "google" in name:
            return ToolCategory.SEARCH
        if "code" in name or "execute" in name:
            return ToolCategory.SHELL
        if "file" in name or "storage" in name:
            return ToolCategory.FILESYSTEM

        return super()._map_tool_category(tool_name)

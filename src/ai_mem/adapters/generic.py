"""Generic adapter for any LLM host.

This adapter provides a flexible parser that works with various payload
formats. Use it as a fallback when no specific adapter is available.

Supports:
- OpenAI function calling format
- Anthropic tool use format
- Custom webhook formats
- Direct Event Schema v1 payloads
"""

from typing import Any, Dict, Optional

from ..events import (
    EventContext,
    EventSource,
    PrivacyFlags,
    ToolCategory,
    ToolEvent,
    ToolExecution,
)
from ..logging_config import get_logger
from .base import EventAdapter

logger = get_logger("adapters.generic")


class GenericAdapter(EventAdapter):
    """Generic adapter that works with any payload format.

    This adapter tries multiple field name patterns to extract tool
    information, making it compatible with various LLM hosts and
    custom integrations.
    """

    def __init__(self, host_name: str = "generic"):
        self._host_name = host_name

    @property
    def host_name(self) -> str:
        return self._host_name

    def parse_tool_event(self, payload: Dict[str, Any]) -> Optional[ToolEvent]:
        """Parse any payload format into a ToolEvent.

        Tries multiple field name patterns:
        - Standard: name, input, output
        - Claude: tool_name, tool_input, tool_response
        - OpenAI: function.name, function.arguments
        - Anthropic: tool_use.name, tool_use.input
        - Direct schema: tool.name, tool.input, tool.output
        """
        logger.debug(f"Parsing generic event for host: {self._host_name}")
        # Check if this is already an Event Schema v1 payload
        if "schema_version" in payload and "tool" in payload:
            logger.debug("Detected Event Schema v1 format")
            return self._parse_schema_v1(payload)

        # Check for nested tool objects
        tool_data = (
            payload.get("tool")
            or payload.get("tool_use")
            or payload.get("function")
            or payload.get("function_call")
        )
        if isinstance(tool_data, dict):
            return self._parse_nested_tool(payload, tool_data)

        # Try flat field extraction
        return self._parse_flat_fields(payload)

    def _parse_schema_v1(self, payload: Dict[str, Any]) -> Optional[ToolEvent]:
        """Parse a payload that's already in Event Schema v1 format."""
        try:
            return ToolEvent.from_dict(payload)
        except Exception:
            return None

    def _parse_nested_tool(
        self, payload: Dict[str, Any], tool_data: Dict[str, Any]
    ) -> Optional[ToolEvent]:
        """Parse a payload with nested tool object."""
        tool_name = tool_data.get("name")
        if not tool_name:
            return None

        tool_input = (
            tool_data.get("input")
            or tool_data.get("arguments")
            or tool_data.get("args")
            or tool_data.get("parameters")
        )
        tool_output = (
            tool_data.get("output")
            or tool_data.get("response")
            or tool_data.get("result")
            or payload.get("tool_response")
            or payload.get("function_response")
        )

        success = tool_data.get("success", True)
        error = tool_data.get("error")
        if error and success is True:
            success = False

        latency = tool_data.get("latency_ms") or tool_data.get("duration")
        if latency is not None:
            try:
                latency = int(latency)
            except (ValueError, TypeError):
                latency = None

        category = tool_data.get("category")
        if category and isinstance(category, str):
            try:
                category = ToolCategory(category)
            except ValueError:
                category = self._map_tool_category(str(tool_name))
        else:
            category = self._map_tool_category(str(tool_name))

        # Build kwargs, only including event_id and timestamp if present
        kwargs: Dict[str, Any] = {
            "session_id": self._extract_session_id(payload),
            "source": self._extract_source(payload),
            "tool": ToolExecution(
                name=str(tool_name),
                category=category,
                input=tool_input,
                output=tool_output,
                success=bool(success),
                latency_ms=latency,
                error=str(error) if error else None,
            ),
            "context": self._extract_context(payload),
            "privacy": self._extract_privacy(payload),
            "metadata": payload.get("metadata", {}),
        }
        if payload.get("event_id"):
            kwargs["event_id"] = payload["event_id"]
        if payload.get("timestamp"):
            kwargs["timestamp"] = payload["timestamp"]

        return ToolEvent(**kwargs)

    def _parse_flat_fields(self, payload: Dict[str, Any]) -> Optional[ToolEvent]:
        """Parse a payload with flat field structure."""
        tool_name = self._extract_first(
            payload, [
                "tool_name", "tool", "name", "function_name",
                "function", "action", "command"
            ]
        )
        if not tool_name:
            return None

        tool_input = self._extract_first(
            payload, [
                "tool_input", "input", "arguments", "args",
                "parameters", "params", "tool_args"
            ]
        )
        tool_output = self._extract_first(
            payload, [
                "tool_output", "tool_response", "output", "response",
                "result", "content", "data"
            ]
        )

        success = self._extract_first(payload, ["success", "succeeded", "ok", "status"])
        if success is None:
            success = True
        elif isinstance(success, str):
            success = success.lower() in ("true", "ok", "success", "1")

        error = self._extract_first(payload, ["error", "error_message", "failure"])

        latency = self._extract_first(
            payload, ["latency_ms", "duration_ms", "elapsed", "duration"]
        )
        if latency is not None:
            try:
                latency = int(latency)
            except (ValueError, TypeError):
                latency = None

        # Build kwargs, only including event_id and timestamp if present
        kwargs: Dict[str, Any] = {
            "session_id": self._extract_session_id(payload),
            "source": self._extract_source(payload),
            "tool": ToolExecution(
                name=str(tool_name),
                category=self._map_tool_category(str(tool_name)),
                input=tool_input,
                output=tool_output,
                success=bool(success),
                latency_ms=latency,
                error=str(error) if error else None,
            ),
            "context": self._extract_context(payload),
            "privacy": self._extract_privacy(payload),
            "metadata": payload.get("metadata", {}),
        }
        # Only set if present (let defaults apply otherwise)
        if payload.get("event_id"):
            kwargs["event_id"] = payload["event_id"]
        if payload.get("timestamp"):
            kwargs["timestamp"] = payload["timestamp"]

        return ToolEvent(**kwargs)

    def _extract_source(self, payload: Dict[str, Any]) -> EventSource:
        """Extract source information from payload."""
        source_data = payload.get("source", {})
        if isinstance(source_data, dict):
            return EventSource(
                host=source_data.get("host", self.host_name),
                agent_id=source_data.get("agent_id"),
                host_version=source_data.get("host_version"),
            )

        return EventSource(
            host=payload.get("host", self.host_name),
            agent_id=payload.get("agent_id") or payload.get("model"),
        )

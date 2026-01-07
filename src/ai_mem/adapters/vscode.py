"""VS Code / GitHub Copilot adapter.

Translates VS Code AI assistant events (like GitHub Copilot Chat) to Event Schema v1.

VS Code extensions may send events in various formats depending on the extension.
This adapter handles common patterns from Copilot and similar extensions.
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

logger = get_logger("adapters.vscode")


class VSCodeAdapter(EventAdapter):
    """Adapter for VS Code AI extension events."""

    @property
    def host_name(self) -> str:
        return "vscode"

    def parse_tool_event(self, payload: Dict[str, Any]) -> Optional[ToolEvent]:
        """Parse a VS Code tool/action event into a ToolEvent.

        VS Code extensions may use different formats:
        - command: for VS Code commands
        - action: for AI actions
        - tool_call: for tool-based interactions
        """
        logger.debug("Parsing VS Code tool event")

        # Try various field names for tool/action name
        tool_name = self._extract_first(
            payload,
            [
                "command",
                "action",
                "name",
                "tool_name",
                "function",
                "operation",
            ],
        )

        if not tool_name:
            return None

        # Extract input/arguments
        tool_input = self._extract_first(
            payload,
            ["args", "arguments", "parameters", "input", "options"],
        )

        # Extract output/result
        tool_output = self._extract_first(
            payload,
            ["result", "output", "response", "content", "data"],
        )

        # Determine success/failure
        error = self._extract_first(payload, ["error", "errorMessage", "error_message"])
        success_field = payload.get("success")
        if success_field is not None:
            success = bool(success_field)
        else:
            success = error is None

        # Extract latency if available
        latency_ms = None
        latency_value = self._extract_first(
            payload, ["duration", "durationMs", "latency_ms", "elapsed"]
        )
        if latency_value is not None:
            try:
                latency_ms = float(latency_value)
            except (ValueError, TypeError):
                pass

        return ToolEvent(
            session_id=self._extract_session_id(payload),
            source=EventSource(
                host=self.host_name,
                agent_id=payload.get("extensionId") or payload.get("extension_id"),
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
                "extension_id": payload.get("extensionId") or payload.get("extension_id"),
                "editor": payload.get("editor"),
                "language": payload.get("languageId") or payload.get("language"),
            },
        )

    def _extract_session_id(self, payload: Dict[str, Any]) -> Optional[str]:
        """Extract session ID from VS Code payload."""
        value = self._extract_first(
            payload,
            [
                "sessionId",
                "session_id",
                "conversationId",
                "conversation_id",
                "threadId",
                "thread_id",
            ],
        )
        return str(value) if value else None

    def _extract_context(self, payload: Dict[str, Any]) -> EventContext:
        """Extract context with VS Code-specific fields."""
        base_context = super()._extract_context(payload)

        # VS Code often includes file/editor context
        active_file = payload.get("activeFile") or payload.get("file") or payload.get("uri")
        if active_file and isinstance(active_file, str):
            if active_file not in base_context.files_touched:
                base_context.files_touched.append(active_file)

        return base_context

    def _map_tool_category(self, tool_name: str) -> Optional[ToolCategory]:
        """Map VS Code tool/command names to categories."""
        name = tool_name.lower()

        # VS Code command patterns
        if name.startswith("workbench.action."):
            return ToolCategory.META
        if name.startswith("editor.action."):
            return ToolCategory.FILESYSTEM
        if name.startswith("git."):
            return ToolCategory.SHELL
        if name.startswith("terminal."):
            return ToolCategory.SHELL

        # Copilot-specific patterns
        if "copilot" in name:
            if any(x in name for x in ["edit", "insert", "apply"]):
                return ToolCategory.FILESYSTEM
            if "explain" in name:
                return ToolCategory.META
            return ToolCategory.CUSTOM

        # Generic patterns
        if any(x in name for x in ["edit", "format", "refactor", "rename"]):
            return ToolCategory.FILESYSTEM
        if any(x in name for x in ["run", "execute", "debug", "test"]):
            return ToolCategory.SHELL
        if any(x in name for x in ["search", "find", "goto"]):
            return ToolCategory.SEARCH
        if any(x in name for x in ["read", "open", "show", "preview"]):
            return ToolCategory.FILESYSTEM

        return super()._map_tool_category(tool_name)

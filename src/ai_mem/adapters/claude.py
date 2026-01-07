"""Claude Code / Claude Desktop adapter.

Translates Claude hook payloads to Event Schema v1.

Claude Code hooks send JSON via stdin with fields like:
- tool_name, tool_input, tool_response
- session_id, cwd, prompt_number
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

logger = get_logger("adapters.claude")


class ClaudeAdapter(EventAdapter):
    """Adapter for Claude Code and Claude Desktop events."""

    @property
    def host_name(self) -> str:
        return "claude-code"

    def parse_tool_event(self, payload: Dict[str, Any]) -> Optional[ToolEvent]:
        """Parse a Claude hook payload into a ToolEvent.

        Claude hooks provide:
        - tool_name / tool / name
        - tool_input / input / tool_args / arguments
        - tool_response / response / result / output
        - session_id / sessionId
        - cwd / working_dir
        - prompt_number (for ordering)
        """
        logger.debug("Parsing Claude tool event")
        tool_name = self._extract_first(
            payload, ["tool_name", "tool", "name"]
        )
        if not tool_name:
            logger.debug("No tool_name found in payload, skipping")
            return None

        tool_input = self._extract_first(
            payload, ["tool_input", "input", "tool_args", "arguments"]
        )
        tool_output = self._extract_first(
            payload, ["tool_response", "response", "result", "output"]
        )
        success = self._extract_first(
            payload, ["success", "tool_success", "succeeded"]
        )
        if success is None:
            success = True

        # Extract latency if available
        latency = self._extract_first(payload, ["latency_ms", "duration_ms", "elapsed"])
        if latency is not None:
            try:
                latency = int(latency)
            except (ValueError, TypeError):
                latency = None

        # Build the event
        return ToolEvent(
            session_id=self._extract_session_id(payload),
            source=EventSource(
                host=self.host_name,
                agent_id=payload.get("agent_id"),
            ),
            tool=ToolExecution(
                name=str(tool_name),
                category=self._map_tool_category(str(tool_name)),
                input=tool_input,
                output=tool_output,
                success=bool(success),
                latency_ms=latency,
            ),
            context=self._extract_context(payload),
            privacy=self._extract_privacy(payload),
            metadata={
                "prompt_number": payload.get("prompt_number"),
                "tool_call_id": payload.get("tool_call_id"),
            },
        )

    def _map_tool_category(self, tool_name: str) -> Optional[ToolCategory]:
        """Map Claude Code tool names to categories."""
        name = tool_name.lower()

        # Claude Code specific tools
        claude_fs_tools = {"read", "write", "glob", "grep", "edit", "multiedit"}
        claude_shell_tools = {"bash", "bashoutput", "killshell"}
        claude_meta_tools = {
            "todowrite", "todoread", "slashcommand", "skill",
            "askfollowupquestion", "attemptcompletion", "enterplanmode",
            "exitplanmode", "task"
        }
        claude_search_tools = {"websearch", "webfetch"}

        if name in claude_fs_tools:
            return ToolCategory.FILESYSTEM
        if name in claude_shell_tools:
            return ToolCategory.SHELL
        if name in claude_meta_tools:
            return ToolCategory.META
        if name in claude_search_tools:
            return ToolCategory.SEARCH
        if name.startswith("mcp__"):
            return ToolCategory.CUSTOM
        if name.startswith("notebook"):
            return ToolCategory.CUSTOM

        return super()._map_tool_category(tool_name)

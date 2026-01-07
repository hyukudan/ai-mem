"""LLM-agnostic adapters for translating host-specific events to Event Schema v1.

This module provides adapters that translate events from different LLM hosts
(Claude, Gemini, Cursor, VS Code, etc.) into the canonical ToolEvent schema.

Usage:
    from ai_mem.adapters import get_adapter, ClaudeAdapter, GeminiAdapter

    # Auto-detect adapter from environment
    adapter = get_adapter()
    event = adapter.parse_tool_event(raw_payload)

    # Or use specific adapter
    adapter = ClaudeAdapter()
    event = adapter.parse_tool_event(claude_hook_payload)
"""

from ..logging_config import get_logger
from .base import EventAdapter
from .claude import ClaudeAdapter
from .cursor import CursorAdapter
from .gemini import GeminiAdapter
from .generic import GenericAdapter
from .vscode import VSCodeAdapter

logger = get_logger("adapters")

__all__ = [
    "EventAdapter",
    "ClaudeAdapter",
    "CursorAdapter",
    "GeminiAdapter",
    "GenericAdapter",
    "VSCodeAdapter",
    "get_adapter",
]


def get_adapter(host: str = None) -> EventAdapter:
    """Get the appropriate adapter for a host.

    Args:
        host: Host identifier. If None, reads from AI_MEM_HOST env var.

    Returns:
        An EventAdapter instance for the specified host.

    Supported hosts:
        - claude, claude-code, claude-desktop, anthropic -> ClaudeAdapter
        - gemini, gemini-cli, google, vertex -> GeminiAdapter
        - cursor, cursor-ai -> CursorAdapter
        - vscode, vs-code, copilot, github-copilot -> VSCodeAdapter
        - (other) -> GenericAdapter
    """
    import os

    if host is None:
        host = os.environ.get("AI_MEM_HOST", "generic")

    host_lower = host.lower().strip()

    if host_lower in ("claude", "claude-code", "claude-desktop", "anthropic"):
        logger.debug(f"Using ClaudeAdapter for host: {host}")
        return ClaudeAdapter()
    elif host_lower in ("gemini", "gemini-cli", "google", "vertex"):
        logger.debug(f"Using GeminiAdapter for host: {host}")
        return GeminiAdapter()
    elif host_lower in ("cursor", "cursor-ai"):
        logger.debug(f"Using CursorAdapter for host: {host}")
        return CursorAdapter()
    elif host_lower in ("vscode", "vs-code", "copilot", "github-copilot"):
        logger.debug(f"Using VSCodeAdapter for host: {host}")
        return VSCodeAdapter()
    else:
        logger.debug(f"Using GenericAdapter for host: {host}")
        return GenericAdapter(host_name=host)

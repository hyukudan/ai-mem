"""LLM-agnostic adapters for translating host-specific events to Event Schema v1.

This module provides adapters that translate events from different LLM hosts
(Claude, Gemini, OpenAI, Cursor, etc.) into the canonical ToolEvent schema.

Usage:
    from ai_mem.adapters import get_adapter, ClaudeAdapter, GeminiAdapter

    # Auto-detect adapter from environment
    adapter = get_adapter()
    event = adapter.parse_tool_event(raw_payload)

    # Or use specific adapter
    adapter = ClaudeAdapter()
    event = adapter.parse_tool_event(claude_hook_payload)
"""

from .base import EventAdapter
from .claude import ClaudeAdapter
from .gemini import GeminiAdapter
from .generic import GenericAdapter

__all__ = [
    "EventAdapter",
    "ClaudeAdapter",
    "GeminiAdapter",
    "GenericAdapter",
    "get_adapter",
]


def get_adapter(host: str = None) -> EventAdapter:
    """Get the appropriate adapter for a host.

    Args:
        host: Host identifier. If None, reads from AI_MEM_HOST env var.

    Returns:
        An EventAdapter instance for the specified host.
    """
    import os

    if host is None:
        host = os.environ.get("AI_MEM_HOST", "generic")

    host_lower = host.lower().strip()

    if host_lower in ("claude", "claude-code", "anthropic"):
        return ClaudeAdapter()
    elif host_lower in ("gemini", "gemini-cli", "google", "vertex"):
        return GeminiAdapter()
    else:
        return GenericAdapter(host_name=host)

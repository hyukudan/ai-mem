"""Tests for LLM host adapters."""

import pytest

from ai_mem.adapters import get_adapter
from ai_mem.adapters.base import EventAdapter
from ai_mem.adapters.claude import ClaudeAdapter
from ai_mem.adapters.gemini import GeminiAdapter
from ai_mem.adapters.generic import GenericAdapter
from ai_mem.events import EventType, ToolCategory


class TestGetAdapter:
    def test_claude_adapter(self):
        adapter = get_adapter("claude-code")
        assert isinstance(adapter, ClaudeAdapter)

    def test_claude_desktop_adapter(self):
        adapter = get_adapter("claude-desktop")
        assert isinstance(adapter, ClaudeAdapter)

    def test_gemini_adapter(self):
        adapter = get_adapter("gemini")
        assert isinstance(adapter, GeminiAdapter)

    def test_gemini_cli_adapter(self):
        adapter = get_adapter("gemini-cli")
        assert isinstance(adapter, GeminiAdapter)

    def test_generic_adapter_fallback(self):
        adapter = get_adapter("unknown-host")
        assert isinstance(adapter, GenericAdapter)

    def test_cursor_uses_generic(self):
        adapter = get_adapter("cursor")
        assert isinstance(adapter, GenericAdapter)


class TestClaudeAdapter:
    def test_parse_tool_event_basic(self):
        adapter = ClaudeAdapter()
        payload = {
            "tool_name": "Read",
            "tool_input": {"path": "/src/main.py"},
            "tool_response": "file contents...",
        }
        event = adapter.parse_tool_event(payload)
        assert event is not None
        assert event.tool.name == "Read"
        assert event.tool.input == {"path": "/src/main.py"}
        assert event.tool.output == "file contents..."
        assert event.source.host == "claude-code"

    def test_parse_tool_event_with_success(self):
        adapter = ClaudeAdapter()
        payload = {
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "tool_response": "files...",
            "success": True,
        }
        event = adapter.parse_tool_event(payload)
        assert event.tool.success is True

    def test_parse_tool_event_failed(self):
        adapter = ClaudeAdapter()
        payload = {
            "tool_name": "Read",
            "tool_input": {"path": "/missing"},
            "tool_response": "Error: file not found",
            "success": False,
        }
        event = adapter.parse_tool_event(payload)
        assert event.tool.success is False

    def test_parse_tool_event_with_latency(self):
        adapter = ClaudeAdapter()
        payload = {
            "tool_name": "Read",
            "latency_ms": 150,
        }
        event = adapter.parse_tool_event(payload)
        assert event.tool.latency_ms == 150

    def test_parse_tool_event_alternative_fields(self):
        adapter = ClaudeAdapter()
        payload = {
            "tool": "Grep",
            "input": {"pattern": "foo"},
            "response": "matches...",
        }
        event = adapter.parse_tool_event(payload)
        assert event.tool.name == "Grep"
        assert event.tool.input == {"pattern": "foo"}
        assert event.tool.output == "matches..."

    def test_parse_tool_event_returns_none_for_empty(self):
        adapter = ClaudeAdapter()
        event = adapter.parse_tool_event({})
        assert event is None

    def test_parse_user_prompt(self):
        adapter = ClaudeAdapter()
        payload = {
            "prompt": "Help me debug this",
            "session_id": "session-1",
        }
        event = adapter.parse_user_prompt(payload)
        assert event is not None
        assert event.content == "Help me debug this"
        assert event.session_id == "session-1"

    def test_parse_session_event_start(self):
        adapter = ClaudeAdapter()
        payload = {
            "session_id": "session-abc",
            "goal": "Implement feature X",
        }
        event = adapter.parse_session_event(payload, EventType.SESSION_START)
        assert event is not None
        assert event.event_type == EventType.SESSION_START
        assert event.session_id == "session-abc"
        assert event.goal == "Implement feature X"


class TestGeminiAdapter:
    def test_parse_tool_event_basic(self):
        adapter = GeminiAdapter()
        payload = {
            "tool_name": "Read",
            "tool_input": {"path": "/src/main.py"},
            "tool_response": "file contents...",
        }
        event = adapter.parse_tool_event(payload)
        assert event is not None
        assert event.tool.name == "Read"
        assert event.source.host == "gemini"

    def test_parse_tool_event_gemini_specific_fields(self):
        adapter = GeminiAdapter()
        # Gemini uses nested function_call object
        payload = {
            "function_call": {"name": "Read", "args": {"path": "/foo"}},
            "function_response": {"response": "content..."},
        }
        event = adapter.parse_tool_event(payload)
        assert event.tool.name == "Read"
        assert event.tool.input == {"path": "/foo"}
        assert event.tool.output == "content..."

    def test_parse_user_prompt(self):
        adapter = GeminiAdapter()
        payload = {
            "content": "What is this code doing?",
        }
        event = adapter.parse_user_prompt(payload)
        assert event is not None
        assert event.content == "What is this code doing?"


class TestGenericAdapter:
    def test_parse_tool_event_basic(self):
        adapter = GenericAdapter()
        payload = {
            "tool_name": "Read",
            "tool_input": {"path": "/src/main.py"},
            "tool_response": "file contents...",
        }
        event = adapter.parse_tool_event(payload)
        assert event is not None
        assert event.tool.name == "Read"

    def test_parse_tool_event_many_field_variations(self):
        adapter = GenericAdapter()

        # Test various field name combinations that GenericAdapter supports
        variations = [
            {"tool_name": "A", "tool_input": {}, "tool_output": "out"},
            {"tool": "B", "input": {}, "output": "out"},
            {"name": "C", "arguments": {}, "result": "out"},
            {"function_name": "D", "parameters": {}, "response": "out"},
        ]

        for payload in variations:
            event = adapter.parse_tool_event(payload)
            assert event is not None, f"Failed for payload: {payload}"
            assert event.tool.name in ["A", "B", "C", "D"]

    def test_parse_schema_v1_passthrough(self):
        adapter = GenericAdapter()
        # If payload is already in Schema v1 format, pass through
        payload = {
            "schema_version": "1.0",
            "event_type": "tool_use",
            "event_id": "existing-id",
            "tool": {
                "name": "Read",
                "input": {"path": "/foo"},
            },
        }
        event = adapter.parse_tool_event(payload)
        assert event is not None
        assert event.event_id == "existing-id"
        assert event.tool.name == "Read"

    def test_tool_category_mapping(self):
        """Test that tool names are mapped to correct categories."""
        adapter = GenericAdapter()

        # Test via actual parsing (category is set internally)
        fs_event = adapter.parse_tool_event({"tool_name": "Read"})
        assert fs_event.tool.category == ToolCategory.FILESYSTEM

        shell_event = adapter.parse_tool_event({"tool_name": "Bash"})
        assert shell_event.tool.category == ToolCategory.SHELL

        search_event = adapter.parse_tool_event({"tool_name": "Grep"})
        assert search_event.tool.category == ToolCategory.SEARCH

        # Unknown tools get CUSTOM category
        custom_event = adapter.parse_tool_event({"tool_name": "UnknownTool"})
        assert custom_event.tool.category == ToolCategory.CUSTOM

    def test_parse_returns_none_for_empty(self):
        adapter = GenericAdapter()
        event = adapter.parse_tool_event({})
        assert event is None

    def test_parse_returns_none_for_no_tool_name(self):
        adapter = GenericAdapter()
        event = adapter.parse_tool_event({"input": "something"})
        assert event is None


class TestAdapterHostIdentifier:
    def test_claude_adapter_host(self):
        adapter = ClaudeAdapter()
        event = adapter.parse_tool_event({"tool_name": "Read"})
        assert event.source.host == "claude-code"

    def test_gemini_adapter_host(self):
        adapter = GeminiAdapter()
        event = adapter.parse_tool_event({"tool_name": "Read"})
        assert event.source.host == "gemini"

    def test_generic_adapter_default_host(self):
        adapter = GenericAdapter()
        event = adapter.parse_tool_event({"tool_name": "Read"})
        assert event.source.host == "generic"

    def test_generic_adapter_custom_host(self):
        adapter = GenericAdapter(host_name="cursor")
        event = adapter.parse_tool_event({"tool_name": "Read"})
        assert event.source.host == "cursor"


class TestAdapterSessionId:
    def test_extracts_session_id(self):
        adapter = ClaudeAdapter()
        payload = {
            "tool_name": "Read",
            "session_id": "my-session",
        }
        event = adapter.parse_tool_event(payload)
        assert event.session_id == "my-session"

    def test_no_session_id(self):
        adapter = ClaudeAdapter()
        payload = {"tool_name": "Read"}
        event = adapter.parse_tool_event(payload)
        assert event.session_id is None


class TestAdapterProjectContext:
    def test_extracts_project(self):
        adapter = GenericAdapter()
        payload = {
            "tool_name": "Read",
            "project": "/my/project",
        }
        event = adapter.parse_tool_event(payload)
        assert event.context.project == "/my/project"

    def test_extracts_cwd(self):
        adapter = GenericAdapter()
        payload = {
            "tool_name": "Read",
            "cwd": "/home/user",
        }
        event = adapter.parse_tool_event(payload)
        assert event.context.cwd == "/home/user"

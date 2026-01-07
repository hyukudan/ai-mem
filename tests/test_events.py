"""Tests for Event Schema v1."""

import pytest
import time
import uuid

from ai_mem.events import (
    EventType,
    ToolCategory,
    EventSource,
    ToolExecution,
    EventContext,
    PrivacyFlags,
    ToolEvent,
    UserPromptEvent,
    SessionEvent,
)


class TestEventSource:
    def test_default_values(self):
        source = EventSource()
        assert source.host == "unknown"
        assert source.agent_id is None
        assert source.host_version is None

    def test_custom_values(self):
        source = EventSource(host="gemini", agent_id="agent-1", host_version="1.0")
        assert source.host == "gemini"
        assert source.agent_id == "agent-1"
        assert source.host_version == "1.0"


class TestToolExecution:
    def test_required_name(self):
        tool = ToolExecution(name="Read")
        assert tool.name == "Read"
        assert tool.success is True
        assert tool.input is None
        assert tool.output is None

    def test_full_execution(self):
        tool = ToolExecution(
            name="Bash",
            category=ToolCategory.SHELL,
            input={"command": "ls -la"},
            output="file list...",
            success=True,
            latency_ms=150,
        )
        assert tool.name == "Bash"
        assert tool.category == ToolCategory.SHELL
        assert tool.input == {"command": "ls -la"}
        assert tool.latency_ms == 150

    def test_failed_execution(self):
        tool = ToolExecution(
            name="Read",
            success=False,
            error="File not found",
        )
        assert tool.success is False
        assert tool.error == "File not found"


class TestEventContext:
    def test_default_values(self):
        ctx = EventContext()
        assert ctx.cwd is None
        assert ctx.project is None
        assert ctx.files_touched == []
        assert ctx.related_ids == []

    def test_with_values(self):
        ctx = EventContext(
            cwd="/home/user",
            project="/my-project",
            files_touched=["a.py", "b.py"],
        )
        assert ctx.cwd == "/home/user"
        assert ctx.files_touched == ["a.py", "b.py"]


class TestPrivacyFlags:
    def test_defaults(self):
        flags = PrivacyFlags()
        assert flags.redact is True
        assert flags.fully_private is False
        assert flags.strip_tags == []

    def test_fully_private(self):
        flags = PrivacyFlags(fully_private=True)
        assert flags.fully_private is True


class TestToolEvent:
    def test_minimal_event(self):
        event = ToolEvent(
            tool=ToolExecution(name="Read"),
        )
        assert event.schema_version == "1.0"
        assert event.event_type == EventType.TOOL_USE
        assert event.event_id  # auto-generated UUID
        assert event.tool.name == "Read"
        assert event.timestamp > 0

    def test_full_event(self):
        event = ToolEvent(
            event_id="custom-id",
            session_id="session-123",
            source=EventSource(host="claude-code"),
            tool=ToolExecution(
                name="Bash",
                input={"command": "ls"},
                output="files...",
            ),
            context=EventContext(project="/my-project"),
            metadata={"custom": "data"},
        )
        assert event.event_id == "custom-id"
        assert event.session_id == "session-123"
        assert event.source.host == "claude-code"
        assert event.context.project == "/my-project"
        assert event.metadata["custom"] == "data"

    def test_idempotency_key_with_event_id(self):
        event = ToolEvent(
            event_id="my-unique-id",
            tool=ToolExecution(name="Read"),
        )
        assert event.idempotency_key() == "my-unique-id"

    def test_idempotency_key_without_event_id(self):
        # When event_id is empty string, it should construct from parts
        event = ToolEvent(
            event_id="",
            session_id="session-1",
            tool=ToolExecution(name="Read"),
            timestamp=1234567890.123,
        )
        key = event.idempotency_key()
        # Empty string is falsy, so it constructs from parts
        assert "session-1" in key
        assert "Read" in key

    def test_from_dict(self):
        data = {
            "event_id": "dict-id",
            "tool": {
                "name": "Read",
                "input": {"path": "/foo"},
            },
            "source": {"host": "gemini"},
        }
        event = ToolEvent.from_dict(data)
        assert event.event_id == "dict-id"
        assert event.tool.name == "Read"
        assert event.source.host == "gemini"


class TestUserPromptEvent:
    def test_creation(self):
        event = UserPromptEvent(
            content="Help me fix this bug",
            session_id="session-1",
        )
        assert event.event_type == EventType.USER_PROMPT
        assert event.content == "Help me fix this bug"
        assert event.session_id == "session-1"

    def test_idempotency_key(self):
        event = UserPromptEvent(
            event_id="prompt-123",
            content="test",
        )
        assert event.idempotency_key() == "prompt-123"


class TestSessionEvent:
    def test_session_start(self):
        event = SessionEvent(
            event_type=EventType.SESSION_START,
            session_id="session-abc",
            goal="Implement new feature",
        )
        assert event.event_type == EventType.SESSION_START
        assert event.session_id == "session-abc"
        assert event.goal == "Implement new feature"

    def test_session_end(self):
        event = SessionEvent(
            event_type=EventType.SESSION_END,
            session_id="session-abc",
        )
        assert event.event_type == EventType.SESSION_END


class TestEventTypes:
    def test_all_event_types(self):
        assert EventType.TOOL_USE == "tool_use"
        assert EventType.USER_PROMPT == "user_prompt"
        assert EventType.SESSION_START == "session_start"
        assert EventType.SESSION_END == "session_end"


class TestToolCategories:
    def test_all_categories(self):
        assert ToolCategory.FILESYSTEM == "fs"
        assert ToolCategory.SHELL == "shell"
        assert ToolCategory.NETWORK == "network"
        assert ToolCategory.SEARCH == "search"
        assert ToolCategory.META == "meta"

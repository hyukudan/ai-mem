"""Event Schema v1 for LLM-agnostic tool event ingestion.

This module defines a neutral event schema that works across all LLM hosts
(Claude, Gemini, OpenAI, Cursor, etc.). Events from any host should be
translated to this schema before processing.

The schema is designed to:
- Be LLM-agnostic (no Claude/Gemini specific fields)
- Support idempotency via event_id
- Capture tool execution context
- Enable filtering and redaction

Usage:
    from ai_mem.events import ToolEvent, EventSource

    event = ToolEvent(
        event_id="uuid",
        session_id="session-123",
        tool=ToolExecution(name="Read", input={"path": "/foo"}, output="..."),
        source=EventSource(host="gemini"),
    )
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid
import time


class EventType(str, Enum):
    """Types of events that can be ingested."""
    TOOL_USE = "tool_use"
    USER_PROMPT = "user_prompt"
    ASSISTANT_RESPONSE = "assistant_response"
    SESSION_START = "session_start"
    SESSION_END = "session_end"


class ToolCategory(str, Enum):
    """Standard categories for tools (optional, for filtering)."""
    FILESYSTEM = "fs"
    SHELL = "shell"
    NETWORK = "network"
    SEARCH = "search"
    LLM = "llm"
    DATABASE = "db"
    META = "meta"  # Tools about the agent itself (TodoWrite, etc.)
    CUSTOM = "custom"


class EventSource(BaseModel):
    """Information about the source/host that generated the event."""

    # Host identifier: claude-code, gemini, openai, cursor, vscode, custom
    host: str = "unknown"
    # Optional agent/model identifier
    agent_id: Optional[str] = None
    # Host-specific version info
    host_version: Optional[str] = None


class ToolExecution(BaseModel):
    """Details of a tool execution."""

    # Tool name (required)
    name: str
    # Tool category (optional, for filtering)
    category: Optional[ToolCategory] = None
    # Tool input (can be string, dict, or any JSON-serializable value)
    input: Optional[Any] = None
    # Tool output (can be string, dict, or any JSON-serializable value)
    output: Optional[Any] = None
    # Whether the tool succeeded
    success: bool = True
    # Execution time in milliseconds
    latency_ms: Optional[int] = None
    # Error message if failed
    error: Optional[str] = None


class EventContext(BaseModel):
    """Contextual information about the event."""

    # Current working directory
    cwd: Optional[str] = None
    # Project path or identifier
    project: Optional[str] = None
    # Files touched by this operation
    files_touched: List[str] = Field(default_factory=list)
    # Related observation IDs (for linking)
    related_ids: List[str] = Field(default_factory=list)


class PrivacyFlags(BaseModel):
    """Privacy and redaction settings for the event."""

    # Apply configured redaction patterns
    redact: bool = True
    # Do not persist this event at all
    fully_private: bool = False
    # Additional tags to strip
    strip_tags: List[str] = Field(default_factory=list)


class ToolEvent(BaseModel):
    """Schema v1 for tool execution events.

    This is the canonical format for all tool events, regardless of
    which LLM host generated them. Adapters translate host-specific
    formats to this schema.
    """

    # Schema version for future compatibility
    schema_version: str = "1.0"

    # Event type
    event_type: EventType = EventType.TOOL_USE

    # Unique event identifier (for idempotency)
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Session identifier (for grouping related events)
    session_id: Optional[str] = None

    # Timestamp (ISO-8601 or Unix epoch)
    timestamp: float = Field(default_factory=time.time)

    # Source information
    source: EventSource = Field(default_factory=EventSource)

    # Tool execution details
    tool: ToolExecution

    # Context information
    context: EventContext = Field(default_factory=EventContext)

    # Privacy settings
    privacy: PrivacyFlags = Field(default_factory=PrivacyFlags)

    # Additional metadata (extensible)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def idempotency_key(self) -> str:
        """Generate a key for idempotency checking.

        Uses event_id if available, otherwise constructs from
        session_id + tool name + timestamp.
        """
        if self.event_id:
            return self.event_id
        parts = [
            self.session_id or "no-session",
            self.tool.name,
            str(int(self.timestamp * 1000)),  # ms precision
        ]
        return ":".join(parts)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolEvent":
        """Create a ToolEvent from a dictionary (flexible parsing)."""
        # Handle nested objects
        if "tool" in data and isinstance(data["tool"], dict):
            data["tool"] = ToolExecution(**data["tool"])
        if "source" in data and isinstance(data["source"], dict):
            data["source"] = EventSource(**data["source"])
        if "context" in data and isinstance(data["context"], dict):
            data["context"] = EventContext(**data["context"])
        if "privacy" in data and isinstance(data["privacy"], dict):
            data["privacy"] = PrivacyFlags(**data["privacy"])
        return cls(**data)


class UserPromptEvent(BaseModel):
    """Schema for user prompt events."""

    schema_version: str = "1.0"
    event_type: EventType = EventType.USER_PROMPT
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    source: EventSource = Field(default_factory=EventSource)

    # The user's prompt content
    content: str

    # Context
    context: EventContext = Field(default_factory=EventContext)
    privacy: PrivacyFlags = Field(default_factory=PrivacyFlags)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def idempotency_key(self) -> str:
        if self.event_id:
            return self.event_id
        parts = [
            self.session_id or "no-session",
            "user_prompt",
            str(int(self.timestamp * 1000)),
        ]
        return ":".join(parts)


class SessionEvent(BaseModel):
    """Schema for session lifecycle events."""

    schema_version: str = "1.0"
    event_type: EventType  # SESSION_START or SESSION_END
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    timestamp: float = Field(default_factory=time.time)
    source: EventSource = Field(default_factory=EventSource)

    # Session goal/description
    goal: Optional[str] = None

    # Context
    context: EventContext = Field(default_factory=EventContext)
    metadata: Dict[str, Any] = Field(default_factory=dict)

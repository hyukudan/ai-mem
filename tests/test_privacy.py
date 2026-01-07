import tempfile
import unittest
import pytest
from unittest import mock

from ai_mem.config import AppConfig, EmbeddingConfig, LLMConfig, StorageConfig
from ai_mem.memory import MemoryManager
from ai_mem.privacy import (
    strip_memory_tags,
    apply_redaction_patterns,
    truncate_content,
    sanitize_for_storage,
    should_skip_tool,
)


class DummyEmbeddingProvider:
    def embed(self, chunks):
        return [[0.0] * 3 for _ in chunks]


class DummyVectorStore:
    def __init__(self, *args, **kwargs):
        pass

    def add(self, embeddings=None, documents=None, metadatas=None, ids=None):
        return None

    def query(self, embedding=None, n_results=0, where=None):
        return {"ids": [[]], "metadatas": [[]], "distances": [[]]}

    def delete_where(self, where):
        return None

    def delete_ids(self, ids):
        return None


@pytest.mark.asyncio
async def test_strip_private_tag():
    cleaned, stripped = strip_memory_tags("Hello<private>secret</private>world")
    assert cleaned == "Helloworld"
    assert stripped

@pytest.mark.asyncio
async def test_strip_context_tag():
    cleaned, stripped = strip_memory_tags("<ai-mem-context>ctx</ai-mem-context>after")
    assert cleaned == "after"
    assert stripped

@pytest.mark.asyncio
async def test_skip_private_only_observation():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AppConfig(
            llm=LLMConfig(provider="none"),
            embeddings=EmbeddingConfig(provider="fastembed"),
            storage=StorageConfig(data_dir=tmpdir),
        )
        with mock.patch("ai_mem.memory._build_embedding_provider", return_value=DummyEmbeddingProvider()), \
             mock.patch("ai_mem.memory.build_vector_store", return_value=DummyVectorStore()):
            manager = MemoryManager(config)
            await manager.initialize()
            obs = await manager.add_observation(
                content="<private>secret</private>",
                obs_type="note",
                project="proj",
            )
            assert obs is None


# === Tests for apply_redaction_patterns ===

def test_apply_redaction_patterns_api_key():
    """Test redaction of API keys in various formats."""
    patterns = [
        r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"]?[\w\-\.]+['\"]?",
    ]

    # Test key=value format
    text = "Config: api_key=sk-abc123xyz"
    result, redacted, count = apply_redaction_patterns(text, patterns)
    assert redacted
    assert count == 1
    assert "sk-abc123xyz" not in result
    assert "[REDACTED]" in result

    # Test apikey: format
    text = "apikey: my-secret-key"
    result, redacted, count = apply_redaction_patterns(text, patterns)
    assert redacted
    assert "my-secret-key" not in result


def test_apply_redaction_patterns_openai_key():
    """Test redaction of OpenAI API keys."""
    patterns = [r"sk-[a-zA-Z0-9]{20,}"]

    text = "Using key: sk-abcdefghijklmnopqrstuvwxyz123456"
    result, redacted, count = apply_redaction_patterns(text, patterns)
    assert redacted
    assert count == 1
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in result


def test_apply_redaction_patterns_no_match():
    """Test that non-sensitive content is not redacted."""
    patterns = [r"sk-[a-zA-Z0-9]{20,}"]

    text = "This is normal text without secrets"
    result, redacted, count = apply_redaction_patterns(text, patterns)
    assert not redacted
    assert count == 0
    assert result == text


def test_apply_redaction_patterns_multiple_matches():
    """Test redaction of multiple occurrences."""
    patterns = [r"secret\d+"]

    text = "Found secret1 and secret2 in config"
    result, redacted, count = apply_redaction_patterns(text, patterns)
    assert redacted
    assert count == 2
    assert "secret1" not in result
    assert "secret2" not in result


def test_apply_redaction_patterns_invalid_regex():
    """Test that invalid regex patterns are skipped."""
    patterns = [r"[invalid(regex", r"valid\d+"]

    text = "Has valid123 in it"
    result, redacted, count = apply_redaction_patterns(text, patterns)
    assert redacted
    assert count == 1
    assert "valid123" not in result


def test_apply_redaction_patterns_empty():
    """Test with empty content or patterns."""
    result, redacted, count = apply_redaction_patterns("", ["pattern"])
    assert result == ""
    assert not redacted

    result, redacted, count = apply_redaction_patterns("text", [])
    assert result == "text"
    assert not redacted

    result, redacted, count = apply_redaction_patterns("text", None)
    assert result == "text"
    assert not redacted


# === Tests for truncate_content ===

def test_truncate_content_short():
    """Test that short content is not truncated."""
    text = "Short text"
    result, truncated = truncate_content(text, 100)
    assert result == text
    assert not truncated


def test_truncate_content_long():
    """Test truncation of long content."""
    text = "A" * 1000
    result, truncated = truncate_content(text, 100)
    assert truncated
    assert len(result) <= 100
    assert result.endswith("[TRUNCATED]")


def test_truncate_content_zero_max():
    """Test that zero max_chars means no truncation."""
    text = "A" * 1000
    result, truncated = truncate_content(text, 0)
    assert result == text
    assert not truncated


def test_truncate_content_custom_suffix():
    """Test truncation with custom suffix."""
    text = "A" * 100
    result, truncated = truncate_content(text, 50, suffix="...")
    assert truncated
    assert result.endswith("...")


# === Tests for sanitize_for_storage ===

def test_sanitize_for_storage_full_pipeline():
    """Test the full sanitization pipeline."""
    content = "<private>hidden</private> api_key=secret123 " + "A" * 1000
    patterns = [r"api_key=\w+"]

    result, metadata = sanitize_for_storage(content, patterns, max_chars=500)

    assert metadata["stripped"]  # private tag was removed
    assert metadata["redacted"]  # api_key was redacted
    assert metadata["redaction_count"] == 1
    assert metadata["truncated"]  # content was truncated
    assert len(result) <= 500
    assert "hidden" not in result
    assert "secret123" not in result


def test_sanitize_for_storage_no_changes():
    """Test sanitization when no changes needed."""
    content = "Normal text"
    result, metadata = sanitize_for_storage(content, None, max_chars=0)

    assert result == content
    assert not metadata["stripped"]
    assert not metadata["redacted"]
    assert not metadata["truncated"]


# === Tests for should_skip_tool ===

def test_should_skip_tool_exact_match():
    """Test skipping by exact tool name."""
    assert should_skip_tool("TodoWrite", skip_names=["TodoWrite", "SlashCommand"])
    assert should_skip_tool("SlashCommand", skip_names=["TodoWrite", "SlashCommand"])
    assert not should_skip_tool("Read", skip_names=["TodoWrite", "SlashCommand"])


def test_should_skip_tool_prefix_match():
    """Test skipping by tool name prefix."""
    assert should_skip_tool("mcp__filesystem__read", skip_prefixes=["mcp__"])
    assert should_skip_tool("_internal_helper", skip_prefixes=["_internal"])
    assert not should_skip_tool("Read", skip_prefixes=["mcp__", "_internal"])


def test_should_skip_tool_category_match():
    """Test skipping by tool category."""
    assert should_skip_tool("SomeTool", skip_categories=["meta", "admin"], tool_category="admin")
    assert not should_skip_tool("SomeTool", skip_categories=["meta", "admin"], tool_category="code")
    assert not should_skip_tool("SomeTool", skip_categories=["meta"], tool_category=None)


def test_should_skip_tool_empty_name():
    """Test that empty/None tool names are skipped."""
    assert should_skip_tool(None)
    assert should_skip_tool("")
    assert should_skip_tool("  ")


def test_should_skip_tool_combined():
    """Test combined filtering rules."""
    # Should skip: matches prefix
    assert should_skip_tool(
        "mcp__test",
        skip_names=["TodoWrite"],
        skip_prefixes=["mcp__"],
        skip_categories=["admin"],
    )

    # Should skip: matches exact name
    assert should_skip_tool(
        "TodoWrite",
        skip_names=["TodoWrite"],
        skip_prefixes=["mcp__"],
    )

    # Should NOT skip: no matches
    assert not should_skip_tool(
        "Read",
        skip_names=["TodoWrite"],
        skip_prefixes=["mcp__"],
        skip_categories=["admin"],
        tool_category="code",
    )

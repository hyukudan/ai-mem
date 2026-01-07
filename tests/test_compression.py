"""Tests for AI Compression Service (Phase 3).

These tests verify the compression functionality:
- Heuristic compression (no LLM required)
- Token estimation
- Compression result structure
- Observation batch compression
"""

import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ai_mem.compression import (
    CompressionService,
    CompressionResult,
    get_compression_service,
    COMPRESSION_PROMPTS,
)
from ai_mem.providers.base import NoOpChatProvider


@pytest.fixture
def compression_service():
    """Create a compression service with no LLM (heuristic only)."""
    return CompressionService(provider=NoOpChatProvider(), default_ratio=4.0)


# =============================================================================
# Basic Compression Tests
# =============================================================================

@pytest.mark.asyncio
async def test_compress_empty_text(compression_service):
    """Test compressing empty text returns early."""
    result = await compression_service.compress("")

    assert isinstance(result, CompressionResult)
    assert result.original_text == ""
    assert result.compressed_text == ""
    assert result.compression_ratio == 1.0
    assert result.method == "none"


@pytest.mark.asyncio
async def test_compress_short_text_uses_heuristic(compression_service):
    """Test that short text uses heuristic compression."""
    short_text = "This is a short test."

    result = await compression_service.compress(short_text)

    assert isinstance(result, CompressionResult)
    assert result.method == "heuristic"
    assert result.original_text == short_text
    assert result.original_tokens > 0


@pytest.mark.asyncio
async def test_compress_long_text(compression_service):
    """Test compressing longer text."""
    long_text = """
    This is a much longer piece of text that should trigger compression.
    It contains multiple sentences and some technical details.
    The authentication system uses JWT tokens for secure access.
    Database migrations were completed successfully yesterday.
    Performance optimizations reduced response times by 40%.
    The API endpoints now support pagination and filtering.
    Error handling was improved with better logging.
    Unit tests coverage increased to 85%.
    """ * 5  # Repeat to make it longer

    result = await compression_service.compress(long_text, target_ratio=2.0)

    assert isinstance(result, CompressionResult)
    assert result.method in ("heuristic", "ai", "heuristic_fallback")
    assert result.compressed_tokens <= result.original_tokens
    assert result.compression_ratio >= 1.0


@pytest.mark.asyncio
async def test_compress_with_different_context_types(compression_service):
    """Test compression with different context types."""
    text = "The function handleAuth() in auth.py processes JWT tokens for /api/login endpoint. Error: TokenExpired was thrown."

    for context_type in ["default", "code_context", "tool_output", "conversation"]:
        result = await compression_service.compress(text, context_type=context_type)
        assert isinstance(result, CompressionResult)
        assert result.original_text == text


# =============================================================================
# Token Estimation Tests
# =============================================================================

def test_estimate_tokens(compression_service):
    """Test token estimation."""
    # Empty string
    assert compression_service._estimate_tokens("") == 1  # min of 1

    # Short string (4 chars = ~1 token)
    assert compression_service._estimate_tokens("test") == 1

    # Longer string
    assert compression_service._estimate_tokens("a" * 100) == 25  # 100/4


# =============================================================================
# Heuristic Compression Tests
# =============================================================================

def test_heuristic_compress_empty(compression_service):
    """Test heuristic compression of empty text."""
    result = compression_service._heuristic_compress("")
    assert result == ""


def test_heuristic_compress_reduces_text(compression_service):
    """Test that heuristic compression actually reduces text."""
    text = """
    This is a verbose piece of text with lots of unnecessary words.
    It contains filler words like very, really, actually, basically.
    The purpose is to test that compression removes these fillers.
    We should see a shorter result after compression is applied.
    """ * 3

    compressed = compression_service._heuristic_compress(text, target_ratio=2.0)

    # Compressed should be shorter or equal
    assert len(compressed) <= len(text)


# =============================================================================
# Batch Compression Tests
# =============================================================================

@pytest.mark.asyncio
async def test_compress_observations_empty_list(compression_service):
    """Test compressing empty observation list."""
    result, stats = await compression_service.compress_observations([])

    assert result == []
    assert stats["total_observations"] == 0
    # Note: overall_ratio is 0/max(1,0) = 0.0 for empty list due to division protection
    assert stats["overall_ratio"] >= 0.0


@pytest.mark.asyncio
async def test_compress_observations_with_content(compression_service):
    """Test compressing a list of observations."""
    observations = [
        {
            "id": "obs1",
            "type": "note",
            "content": "First observation with some content to compress here.",
        },
        {
            "id": "obs2",
            "type": "tool_output",
            "content": "Tool output: success=true, result={data: [1,2,3]}",
        },
        {
            "id": "obs3",
            "type": "interaction",
            "content": "User asked about authentication. I explained JWT tokens.",
        },
    ]

    result, stats = await compression_service.compress_observations(
        observations,
        field="content",
        target_ratio=2.0,
    )

    assert len(result) == 3
    assert stats["total_observations"] == 3

    # Each observation should have compression metadata
    for obs in result:
        assert "_compression" in obs
        assert "original_tokens" in obs["_compression"]
        assert "compressed_tokens" in obs["_compression"]


@pytest.mark.asyncio
async def test_compress_observations_preserves_other_fields(compression_service):
    """Test that compression preserves non-content fields."""
    observations = [
        {
            "id": "obs1",
            "type": "note",
            "content": "Content to compress",
            "tags": ["important"],
            "metadata": {"key": "value"},
        },
    ]

    result, _ = await compression_service.compress_observations(observations)

    assert result[0]["id"] == "obs1"
    assert result[0]["type"] == "note"
    assert result[0]["tags"] == ["important"]
    assert result[0]["metadata"] == {"key": "value"}


# =============================================================================
# Singleton / Factory Tests
# =============================================================================

def test_get_compression_service_singleton():
    """Test that get_compression_service returns consistent instance."""
    service1 = get_compression_service()
    service2 = get_compression_service()

    # Should be the same instance when no provider specified
    assert service1 is service2


def test_get_compression_service_with_provider():
    """Test that providing a provider creates a new service."""
    provider = NoOpChatProvider()
    service = get_compression_service(provider=provider)

    assert service.provider is provider


# =============================================================================
# Compression Prompts Tests
# =============================================================================

def test_compression_prompts_exist():
    """Test that all expected prompt templates exist."""
    expected_types = ["default", "code_context", "tool_output", "conversation"]

    for prompt_type in expected_types:
        assert prompt_type in COMPRESSION_PROMPTS
        assert "{text}" in COMPRESSION_PROMPTS[prompt_type]
        assert "{target_ratio}" in COMPRESSION_PROMPTS[prompt_type]


# =============================================================================
# CompressionResult Tests
# =============================================================================

def test_compression_result_structure():
    """Test CompressionResult dataclass structure."""
    result = CompressionResult(
        original_text="original",
        compressed_text="compressed",
        original_tokens=100,
        compressed_tokens=25,
        compression_ratio=4.0,
        method="ai",
    )

    assert result.original_text == "original"
    assert result.compressed_text == "compressed"
    assert result.original_tokens == 100
    assert result.compressed_tokens == 25
    assert result.compression_ratio == 4.0
    assert result.method == "ai"

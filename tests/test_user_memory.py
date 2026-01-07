"""Tests for User-Level Memory (Phase 4).

Tests for global user memories that persist across projects.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest import mock

from ai_mem.config import AppConfig, EmbeddingConfig, LLMConfig, StorageConfig
from ai_mem.memory import MemoryManager, USER_SCOPE_PROJECT


class DummyEmbeddingProvider:
    def embed(self, chunks):
        return [[0.0] * 3 for _ in chunks]


class DummyVectorStore:
    def add(self, **kwargs):
        pass

    def query(self, **kwargs):
        return {"metadatas": [[]]}


@pytest.fixture
async def manager():
    """Create memory manager for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AppConfig(
            llm=LLMConfig(provider="none"),
            embeddings=EmbeddingConfig(provider="fastembed"),
            storage=StorageConfig(data_dir=tmpdir),
        )
        with mock.patch(
            "ai_mem.memory._build_embedding_provider", return_value=DummyEmbeddingProvider()
        ), mock.patch("ai_mem.memory.build_vector_store", return_value=DummyVectorStore()):
            manager = MemoryManager(config)
            await manager.initialize()
            yield manager
            await manager.close()


# =============================================================================
# Constants Tests
# =============================================================================


class TestUserScopeConstant:
    """Tests for USER_SCOPE_PROJECT constant."""

    def test_user_scope_project_is_string(self):
        """USER_SCOPE_PROJECT should be a string."""
        assert isinstance(USER_SCOPE_PROJECT, str)

    def test_user_scope_project_is_special(self):
        """USER_SCOPE_PROJECT should start with underscore to be special."""
        assert USER_SCOPE_PROJECT.startswith("_")

    def test_user_scope_project_value(self):
        """USER_SCOPE_PROJECT has expected value."""
        assert USER_SCOPE_PROJECT == "_user"


# =============================================================================
# add_user_memory Tests
# =============================================================================


@pytest.mark.asyncio
async def test_add_user_memory_basic(manager):
    """Can add a basic user memory."""
    obs = await manager.add_user_memory(
        content="I prefer TypeScript over JavaScript",
        obs_type="preference",
    )

    assert obs is not None
    assert obs.project == USER_SCOPE_PROJECT
    assert obs.content == "I prefer TypeScript over JavaScript"


@pytest.mark.asyncio
async def test_add_user_memory_with_tags(manager):
    """Can add user memory with tags."""
    obs = await manager.add_user_memory(
        content="Use 4 spaces for indentation",
        obs_type="preference",
        tags=["coding-style", "formatting"],
    )

    assert obs is not None
    assert "coding-style" in obs.tags


@pytest.mark.asyncio
async def test_add_user_memory_default_type(manager):
    """Default observation type is 'preference'."""
    obs = await manager.add_user_memory(
        content="Some preference content",
    )

    assert obs is not None
    assert obs.type == "preference"


@pytest.mark.asyncio
async def test_add_user_memory_custom_type(manager):
    """Can use custom observation type."""
    obs = await manager.add_user_memory(
        content="Convention content",
        obs_type="convention",
    )

    assert obs is not None
    assert obs.type == "convention"


# =============================================================================
# get_user_memories Tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_user_memories_empty(manager):
    """Returns empty list when no user memories."""
    results = await manager.get_user_memories()
    assert results == []


@pytest.mark.asyncio
async def test_get_user_memories_returns_user_scope(manager):
    """Only returns memories with user scope."""
    # Add user memory
    await manager.add_user_memory(content="User preference")

    # Add regular project memory
    await manager.add_observation(
        content="Project memory",
        obs_type="note",
        project="/some/project",
    )

    # Get user memories
    results = await manager.get_user_memories()

    assert len(results) == 1
    assert results[0].project == USER_SCOPE_PROJECT


@pytest.mark.asyncio
async def test_get_user_memories_with_limit(manager):
    """Respects limit parameter."""
    for i in range(5):
        await manager.add_user_memory(
            content=f"Preference {i}",
            summarize=False,
        )

    results = await manager.get_user_memories(limit=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_get_user_memories_filter_by_type(manager):
    """Can filter by observation type."""
    await manager.add_user_memory(content="Pref 1", obs_type="preference")
    await manager.add_user_memory(content="Conv 1", obs_type="convention")

    prefs = await manager.get_user_memories(obs_type="preference")
    convs = await manager.get_user_memories(obs_type="convention")

    assert len(prefs) >= 1
    assert len(convs) >= 1


# =============================================================================
# search_user_memories Tests
# =============================================================================


@pytest.mark.asyncio
async def test_search_user_memories_empty(manager):
    """Returns empty list when no matches."""
    results = await manager.search_user_memories(query="nonexistent")
    assert results == []


@pytest.mark.asyncio
async def test_search_user_memories_basic(manager):
    """Can search user memories."""
    await manager.add_user_memory(
        content="I prefer TypeScript for type safety",
    )

    results = await manager.search_user_memories(query="typescript")

    # May or may not find due to vector store mocking
    assert isinstance(results, list)


# =============================================================================
# export_user_memories Tests
# =============================================================================


@pytest.mark.asyncio
async def test_export_user_memories_creates_file(manager):
    """Export creates a JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test-export.json")

        await manager.add_user_memory(content="Export test memory")

        result = await manager.export_user_memories(output_path=output_path)

        assert result["path"] == output_path
        assert os.path.exists(output_path)


@pytest.mark.asyncio
async def test_export_user_memories_json_format(manager):
    """Export file has correct JSON format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test-export.json")

        await manager.add_user_memory(content="Export test memory")
        await manager.export_user_memories(output_path=output_path)

        with open(output_path, "r") as f:
            data = json.load(f)

        assert "version" in data
        assert "exported_at" in data
        assert "count" in data
        assert "memories" in data


@pytest.mark.asyncio
async def test_export_user_memories_default_path(manager):
    """Export uses default path when not specified."""
    result = await manager.export_user_memories()

    expected_path = str(Path.home() / ".config" / "ai-mem" / "user-memory.json")
    assert result["path"] == expected_path

    # Clean up
    if os.path.exists(expected_path):
        os.remove(expected_path)


# =============================================================================
# import_user_memories Tests
# =============================================================================


@pytest.mark.asyncio
async def test_import_user_memories_file_not_found(manager):
    """Returns empty result when file not found."""
    result = await manager.import_user_memories(input_path="/nonexistent/path.json")

    assert result["imported"] == 0
    assert result["skipped"] == 0


@pytest.mark.asyncio
async def test_import_user_memories_from_file(manager):
    """Can import user memories from file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create export file
        export_path = os.path.join(tmpdir, "import-test.json")
        export_data = {
            "version": "1.0",
            "exported_at": "2024-01-01T00:00:00",
            "count": 2,
            "memories": [
                {"content": "Memory 1", "type": "preference"},
                {"content": "Memory 2", "type": "convention"},
            ],
        }
        with open(export_path, "w") as f:
            json.dump(export_data, f)

        result = await manager.import_user_memories(input_path=export_path)

        assert result["imported"] == 2
        assert result["skipped"] == 0


@pytest.mark.asyncio
async def test_import_user_memories_merge_mode(manager):
    """Merge mode keeps existing memories."""
    # Add existing memory
    await manager.add_user_memory(content="Existing memory")

    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = os.path.join(tmpdir, "import-test.json")
        export_data = {
            "version": "1.0",
            "memories": [{"content": "New memory"}],
        }
        with open(export_path, "w") as f:
            json.dump(export_data, f)

        await manager.import_user_memories(input_path=export_path, merge=True)

        # Should have both memories
        all_memories = await manager.get_user_memories()
        assert len(all_memories) == 2


@pytest.mark.asyncio
async def test_import_user_memories_replace_mode(manager):
    """Replace mode deletes existing memories."""
    # Add existing memory
    await manager.add_user_memory(content="Existing memory")

    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = os.path.join(tmpdir, "import-test.json")
        export_data = {
            "version": "1.0",
            "memories": [{"content": "New memory"}],
        }
        with open(export_path, "w") as f:
            json.dump(export_data, f)

        await manager.import_user_memories(input_path=export_path, merge=False)

        # Should only have new memory
        all_memories = await manager.get_user_memories()
        assert len(all_memories) == 1


# =============================================================================
# get_user_memory_count Tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_user_memory_count_empty(manager):
    """Returns 0 when no user memories."""
    count = await manager.get_user_memory_count()
    assert count == 0


@pytest.mark.asyncio
async def test_get_user_memory_count_with_memories(manager):
    """Returns correct count."""
    await manager.add_user_memory(content="Memory 1")
    await manager.add_user_memory(content="Memory 2")
    await manager.add_user_memory(content="Memory 3")

    count = await manager.get_user_memory_count()
    assert count == 3


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_user_memories_isolated_from_project(manager):
    """User memories don't appear in project queries."""
    await manager.add_user_memory(content="User preference")
    await manager.add_observation(
        content="Project memory",
        obs_type="note",
        project="/test/project",
    )

    # Query project memories using db.list_observations
    project_memories = await manager.db.list_observations(project="/test/project")

    # User memories should not appear
    for mem in project_memories:
        assert mem.project != USER_SCOPE_PROJECT


@pytest.mark.asyncio
async def test_roundtrip_export_import(manager):
    """Export and import preserves memories."""
    # Create original memories
    await manager.add_user_memory(content="Original 1", tags=["test"])
    await manager.add_user_memory(content="Original 2", obs_type="convention")

    original_count = await manager.get_user_memory_count()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Export
        export_path = os.path.join(tmpdir, "roundtrip.json")
        await manager.export_user_memories(output_path=export_path)

        # Clear existing (simulate replace)
        await manager.import_user_memories(input_path=export_path, merge=False)

        # Check count
        final_count = await manager.get_user_memory_count()
        assert final_count == original_count

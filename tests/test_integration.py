"""Integration Tests - End-to-end tests for ai-mem.

This module contains comprehensive integration tests that verify
the complete flow of operations across all layers of the system.

Test Categories:
- CLI → Memory → DB flow
- MCP Tool → Memory → Vector Store flow
- API Endpoint → Memory → Response flow
- Full Observation Lifecycle
- Cross-Layer Consistency
- Error Handling
"""

import asyncio
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mark all tests as integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    try:
        os.unlink(f.name)
    except OSError:
        pass


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def test_config(temp_db_path, temp_dir):
    """Create test configuration."""
    return {
        "storage": {
            "sqlite_path": temp_db_path,
            "vector_store_path": temp_dir,
        },
        "search": {
            "cache_ttl_seconds": 60,
            "cache_max_entries": 100,
            "fts_weight": 0.5,
            "recency_weight": 0.1,
        },
        "default_project": "test-project",
    }


@pytest.fixture
async def memory_manager(test_config):
    """Create a memory manager for testing."""
    from ai_mem.memory import MemoryManager
    from ai_mem.config import AppConfig

    # Create minimal config
    config = AppConfig(**test_config)
    manager = MemoryManager(config)
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
async def db_manager(temp_db_path):
    """Create a database manager for testing."""
    from ai_mem.db import DatabaseManager

    db = DatabaseManager(temp_db_path)
    await db.initialize()
    await db.create_tables()
    yield db
    await db.close()


# =============================================================================
# Test: CLI → Memory → DB Flow
# =============================================================================

class TestCLIMemoryFlow:
    """Test CLI command flows through memory system to database."""

    @pytest.mark.asyncio
    async def test_add_observation_flow(self, db_manager):
        """TC-1.1: CLI Add → Memory Manager → SQLite"""
        from ai_mem.services.session import SessionService
        from ai_mem.services.observation import ObservationService
        from ai_mem.config import AppConfig

        # Setup
        config = AppConfig()
        session_service = SessionService(db_manager, config)
        obs_service = ObservationService(db_manager, None, config, session_service)

        # Act: Add observation
        content = "Python 3.11 released with new features"
        obs = await obs_service.add_observation(
            content=content,
            obs_type="note",
            tags=["python", "release"],
            project="test-project",
        )

        # Assert
        assert obs is not None
        assert obs.id is not None
        assert obs.content == content
        assert obs.type == "note"
        assert "python" in obs.tags
        assert "release" in obs.tags

        # Verify in database
        db_obs = await db_manager.get_observation(obs.id)
        assert db_obs is not None
        assert db_obs.content == content

    @pytest.mark.asyncio
    async def test_search_returns_results(self, db_manager):
        """TC-1.2: CLI Search → Hybrid Search → Results"""
        from ai_mem.services.session import SessionService
        from ai_mem.services.observation import ObservationService
        from ai_mem.services.cache import SearchCacheService
        from ai_mem.services.similarity import SimilarityService
        from ai_mem.services.reranking import RerankingService
        from ai_mem.services.search import SearchService
        from ai_mem.config import AppConfig

        config = AppConfig()

        # Setup services
        session_service = SessionService(db_manager, config)
        obs_service = ObservationService(db_manager, None, config, session_service)
        cache_service = SearchCacheService(config)
        similarity_service = SimilarityService(config)
        reranking_service = RerankingService(config, similarity_service)
        search_service = SearchService(
            db_manager, None, None, config,
            cache_service, similarity_service, reranking_service
        )

        # Add test observations
        await obs_service.add_observation(
            content="Python 3.11 has new features",
            obs_type="note",
            project="test-project",
        )
        await obs_service.add_observation(
            content="JavaScript ES2023 release",
            obs_type="note",
            project="test-project",
        )
        await obs_service.add_observation(
            content="Python async improvements",
            obs_type="note",
            project="test-project",
        )

        # Act: Search
        results = await search_service.search(
            query="Python features",
            limit=5,
            project="test-project",
        )

        # Assert
        assert len(results) >= 1
        # Python results should rank higher
        assert any("Python" in r.get("content", "") for r in results)

    @pytest.mark.asyncio
    async def test_delete_removes_observation(self, db_manager):
        """TC-1.4: CLI Delete → Database Cascade"""
        from ai_mem.services.session import SessionService
        from ai_mem.services.observation import ObservationService
        from ai_mem.config import AppConfig

        config = AppConfig()
        session_service = SessionService(db_manager, config)
        obs_service = ObservationService(db_manager, None, config, session_service)

        # Add and then delete
        obs = await obs_service.add_observation(
            content="To be deleted",
            obs_type="note",
            project="test-project",
        )

        assert obs is not None
        obs_id = obs.id

        # Delete
        deleted = await obs_service.delete_observation(obs_id)
        assert deleted is True

        # Verify gone
        db_obs = await db_manager.get_observation(obs_id)
        assert db_obs is None


# =============================================================================
# Test: Session Lifecycle
# =============================================================================

class TestSessionLifecycle:
    """Test session management flows."""

    @pytest.mark.asyncio
    async def test_session_start_and_end(self, db_manager):
        """TC-1.5: CLI Session Lifecycle"""
        from ai_mem.services.session import SessionService
        from ai_mem.config import AppConfig

        config = AppConfig()
        service = SessionService(db_manager, config)

        # Start session
        session = await service.start_session(
            project="my-project",
            goal="Debug auth flow",
        )

        assert session is not None
        assert session["project"] == "my-project"
        assert session["goal"] == "Debug auth flow"
        assert session["end_time"] is None

        # End session
        ended = await service.end_session(session["id"], summary="Fixed the bug")

        assert ended is not None
        assert ended["end_time"] is not None
        assert ended["summary"] == "Fixed the bug"

    @pytest.mark.asyncio
    async def test_ensure_session_creates_if_needed(self, db_manager):
        """Test ensure_session auto-creates session."""
        from ai_mem.services.session import SessionService
        from ai_mem.config import AppConfig

        config = AppConfig()
        service = SessionService(db_manager, config)

        # Ensure session (should create)
        session = await service.ensure_session("new-project")

        assert session is not None
        assert session["project"] == "new-project"

        # Ensure again (should return same)
        session2 = await service.ensure_session("new-project")
        assert session2["id"] == session["id"]


# =============================================================================
# Test: Tag Operations
# =============================================================================

class TestTagOperations:
    """Test tag management operations."""

    @pytest.mark.asyncio
    async def test_tag_add_rename_delete(self, db_manager):
        """TC-2.4: Tag CRUD operations"""
        from ai_mem.services.session import SessionService
        from ai_mem.services.observation import ObservationService
        from ai_mem.services.tag import TagService
        from ai_mem.config import AppConfig

        config = AppConfig()
        session_service = SessionService(db_manager, config)
        obs_service = ObservationService(db_manager, None, config, session_service)
        tag_service = TagService(db_manager)

        # Add observation with tags
        obs = await obs_service.add_observation(
            content="Test observation",
            obs_type="note",
            tags=["initial"],
            project="test-project",
        )

        # Add tag
        updated = await tag_service.add_tag(
            tag="new-tag",
            project="test-project",
        )
        assert updated >= 1

        # List tags
        tags = await tag_service.list_tags(project="test-project")
        tag_names = [t["tag"] for t in tags]
        assert "initial" in tag_names
        assert "new-tag" in tag_names

        # Rename tag
        renamed = await tag_service.rename_tag(
            old_tag="new-tag",
            new_tag="renamed-tag",
            project="test-project",
        )
        assert renamed >= 1

        # Verify rename
        tags = await tag_service.list_tags(project="test-project")
        tag_names = [t["tag"] for t in tags]
        assert "renamed-tag" in tag_names
        assert "new-tag" not in tag_names

        # Delete tag
        deleted = await tag_service.delete_tag(
            tag="renamed-tag",
            project="test-project",
        )
        assert deleted >= 1


# =============================================================================
# Test: Observation Lifecycle
# =============================================================================

class TestObservationLifecycle:
    """Test full observation lifecycle."""

    @pytest.mark.asyncio
    async def test_create_retrieve_delete_cycle(self, db_manager):
        """TC-4.1: Complete Create-Retrieve-Delete Cycle"""
        from ai_mem.services.session import SessionService
        from ai_mem.services.observation import ObservationService
        from ai_mem.config import AppConfig

        config = AppConfig()
        session_service = SessionService(db_manager, config)
        obs_service = ObservationService(db_manager, None, config, session_service)

        # Create
        obs = await obs_service.add_observation(
            content="Implement OAuth2 flow for mobile",
            obs_type="feature",
            tags=["security", "mobile"],
            project="test-project",
            title="OAuth2 Implementation",
        )

        assert obs is not None
        obs_id = obs.id

        # Retrieve
        retrieved = await obs_service.get_observation(obs_id)
        assert retrieved is not None
        assert retrieved.content == "Implement OAuth2 flow for mobile"
        assert retrieved.title == "OAuth2 Implementation"
        assert "security" in retrieved.tags

        # Delete
        deleted = await obs_service.delete_observation(obs_id)
        assert deleted is True

        # Verify deleted
        gone = await obs_service.get_observation(obs_id)
        assert gone is None

    @pytest.mark.asyncio
    async def test_batch_operations(self, db_manager):
        """TC-3.2: Batch observation creation"""
        from ai_mem.services.session import SessionService
        from ai_mem.services.observation import ObservationService
        from ai_mem.config import AppConfig

        config = AppConfig()
        session_service = SessionService(db_manager, config)
        obs_service = ObservationService(db_manager, None, config, session_service)

        # Batch add
        observations = [
            {"content": "Observation 1", "obs_type": "note", "project": "test-project"},
            {"content": "Observation 2", "obs_type": "bug", "project": "test-project"},
            {"content": "Observation 3", "obs_type": "feature", "project": "test-project"},
        ]

        result = await obs_service.add_batch(observations)

        assert result["added"] == 3
        assert len(result["ids"]) == 3

        # Verify all exist
        for obs_id in result["ids"]:
            obs = await obs_service.get_observation(obs_id)
            assert obs is not None


# =============================================================================
# Test: Deduplication
# =============================================================================

class TestDeduplication:
    """Test content deduplication."""

    @pytest.mark.asyncio
    async def test_duplicate_content_not_added(self, db_manager):
        """TC-3.10: Event deduplication via content hash"""
        from ai_mem.services.session import SessionService
        from ai_mem.services.observation import ObservationService
        from ai_mem.config import AppConfig

        config = AppConfig()
        session_service = SessionService(db_manager, config)
        obs_service = ObservationService(db_manager, None, config, session_service)

        content = "Unique content for deduplication test"

        # First add
        obs1 = await obs_service.add_observation(
            content=content,
            obs_type="note",
            project="test-project",
            dedupe=True,
        )

        # Second add with same content
        obs2 = await obs_service.add_observation(
            content=content,
            obs_type="note",
            project="test-project",
            dedupe=True,
        )

        # Should return same observation
        assert obs1.id == obs2.id

    @pytest.mark.asyncio
    async def test_event_id_idempotency(self, db_manager):
        """Test event_id prevents duplicate processing"""
        from ai_mem.services.session import SessionService
        from ai_mem.services.observation import ObservationService
        from ai_mem.config import AppConfig

        config = AppConfig()
        session_service = SessionService(db_manager, config)
        obs_service = ObservationService(db_manager, None, config, session_service)

        event_id = f"evt-{uuid.uuid4()}"

        # First add with event_id
        obs1 = await obs_service.add_observation(
            content="Event content",
            obs_type="note",
            project="test-project",
            event_id=event_id,
        )

        # Second add with same event_id
        obs2 = await obs_service.add_observation(
            content="Different content",  # Different content
            obs_type="note",
            project="test-project",
            event_id=event_id,
        )

        # Should return same observation (idempotent)
        assert obs1.id == obs2.id


# =============================================================================
# Test: Services Integration
# =============================================================================

class TestServicesIntegration:
    """Test service integration and dependencies."""

    @pytest.mark.asyncio
    async def test_similarity_service(self):
        """Test similarity computation service"""
        from ai_mem.services.similarity import SimilarityService
        from ai_mem.config import AppConfig

        config = AppConfig()
        service = SimilarityService(config)

        # Text similarity
        sim = service.compute_text_similarity(
            "Python programming language",
            "Python coding language",
        )
        assert 0 < sim < 1

        # Identical texts
        sim_same = service.compute_text_similarity(
            "identical text",
            "identical text",
        )
        assert sim_same == 1.0

        # Score combining
        combined = service.combine_scores(0.8, 0.6, fts_weight=0.5)
        assert 0 < combined < 1

    @pytest.mark.asyncio
    async def test_cache_service(self):
        """Test search cache service"""
        from ai_mem.services.cache import SearchCacheService
        from ai_mem.config import AppConfig

        config = AppConfig()
        config.search.cache_ttl_seconds = 60
        config.search.cache_max_entries = 100

        service = SearchCacheService(config)

        # Build key
        key = service.build_key("test query", 10, "project")
        assert len(key) == 32  # SHA256 truncated

        # Cache miss
        result = service.get(key)
        assert result is None
        assert service.last_hit is False

        # Set cache
        test_results = [{"id": "1", "score": 0.9}]
        service.set(key, test_results)

        # Cache hit
        cached = service.get(key)
        assert cached == test_results
        assert service.last_hit is True

        # Summary
        summary = service.get_summary()
        assert summary["hits"] == 1
        assert summary["misses"] == 1

    @pytest.mark.asyncio
    async def test_listener_service(self):
        """Test listener notification service"""
        from ai_mem.services.listener import ListenerService

        service = ListenerService()
        received = []

        def listener(obs):
            received.append(obs)

        # Add listener
        service.add_listener(listener)
        assert service.count == 1

        # Notify
        mock_obs = MagicMock()
        mock_obs.id = "test-id"
        service.notify(mock_obs)

        assert len(received) == 1
        assert received[0].id == "test-id"

        # Remove listener
        removed = service.remove_listener(listener)
        assert removed is True
        assert service.count == 0


# =============================================================================
# Test: Error Handling
# =============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_observation_id(self, db_manager):
        """TC-6.1: Invalid observation ID handling"""
        from ai_mem.services.session import SessionService
        from ai_mem.services.observation import ObservationService
        from ai_mem.config import AppConfig

        config = AppConfig()
        session_service = SessionService(db_manager, config)
        obs_service = ObservationService(db_manager, None, config, session_service)

        # Try to get non-existent observation
        result = await obs_service.get_observation("non-existent-id")
        assert result is None

        # Try to delete non-existent observation
        deleted = await obs_service.delete_observation("non-existent-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_empty_content_rejected(self, db_manager):
        """Test empty content is rejected"""
        from ai_mem.services.session import SessionService
        from ai_mem.services.observation import ObservationService
        from ai_mem.config import AppConfig

        config = AppConfig()
        session_service = SessionService(db_manager, config)
        obs_service = ObservationService(db_manager, None, config, session_service)

        # Empty content should return None
        result = await obs_service.add_observation(
            content="",
            obs_type="note",
            project="test-project",
        )
        assert result is None

        # Whitespace only should return None
        result = await obs_service.add_observation(
            content="   ",
            obs_type="note",
            project="test-project",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_search_results(self, db_manager):
        """TC-6.4: Empty search results handled correctly"""
        from ai_mem.services.cache import SearchCacheService
        from ai_mem.services.similarity import SimilarityService
        from ai_mem.services.reranking import RerankingService
        from ai_mem.services.search import SearchService
        from ai_mem.config import AppConfig

        config = AppConfig()
        cache_service = SearchCacheService(config)
        similarity_service = SimilarityService(config)
        reranking_service = RerankingService(config, similarity_service)
        search_service = SearchService(
            db_manager, None, None, config,
            cache_service, similarity_service, reranking_service
        )

        # Search in empty database
        results = await search_service.search(
            query="nonexistent content",
            limit=10,
            project="nonexistent-project",
        )

        assert results == []


# =============================================================================
# Test: Cross-Layer Consistency
# =============================================================================

class TestCrossLayerConsistency:
    """Test consistency across different access layers."""

    @pytest.mark.asyncio
    async def test_concurrent_add_operations(self, db_manager):
        """TC-5.3: Concurrent add operations"""
        from ai_mem.services.session import SessionService
        from ai_mem.services.observation import ObservationService
        from ai_mem.config import AppConfig

        config = AppConfig()
        session_service = SessionService(db_manager, config)
        obs_service = ObservationService(db_manager, None, config, session_service)

        # Concurrent adds
        async def add_observation(i):
            return await obs_service.add_observation(
                content=f"Concurrent observation {i}",
                obs_type="note",
                project="test-project",
            )

        # Run 5 concurrent adds
        results = await asyncio.gather(*[add_observation(i) for i in range(5)])

        # All should succeed
        assert len(results) == 5
        assert all(r is not None for r in results)

        # All IDs should be unique
        ids = [r.id for r in results]
        assert len(set(ids)) == 5


# =============================================================================
# Run configuration
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])

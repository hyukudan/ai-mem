"""Pytest Configuration - Shared fixtures and configuration.

This module provides shared fixtures and pytest configuration
for all test modules.
"""

import os
import tempfile
from typing import Generator

import pytest


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks (run with '-m benchmark')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# =============================================================================
# Shared Fixtures
# =============================================================================

@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Create a temporary database file path.

    Yields:
        Path to temporary database file
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    try:
        os.unlink(f.name)
    except OSError:
        pass


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory.

    Yields:
        Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def test_config(temp_db_path: str, temp_dir: str) -> dict:
    """Create test configuration.

    Args:
        temp_db_path: Temporary database path
        temp_dir: Temporary directory

    Returns:
        Configuration dictionary
    """
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
def sample_observations() -> list:
    """Create sample observation data.

    Returns:
        List of sample observation dictionaries
    """
    return [
        {
            "content": "Python 3.11 introduces new features",
            "obs_type": "note",
            "tags": ["python", "release"],
            "project": "test-project",
        },
        {
            "content": "Fixed memory leak in cache module",
            "obs_type": "bug",
            "tags": ["memory", "fix"],
            "project": "test-project",
        },
        {
            "content": "Implemented async batch processing",
            "obs_type": "feature",
            "tags": ["async", "performance"],
            "project": "test-project",
        },
        {
            "content": "Database query optimization patterns",
            "obs_type": "discovery",
            "tags": ["database", "optimization"],
            "project": "test-project",
        },
        {
            "content": "Decision to use SQLite for local storage",
            "obs_type": "decision",
            "tags": ["architecture", "database"],
            "project": "test-project",
        },
    ]


# =============================================================================
# Async Fixtures
# =============================================================================

@pytest.fixture
async def db_manager(temp_db_path: str):
    """Create an initialized database manager.

    Args:
        temp_db_path: Temporary database path

    Yields:
        Initialized DatabaseManager
    """
    from ai_mem.db import DatabaseManager

    db = DatabaseManager(temp_db_path)
    await db.initialize()
    await db.create_tables()
    yield db
    await db.close()


@pytest.fixture
async def populated_db(db_manager, sample_observations):
    """Create a database populated with sample observations.

    Args:
        db_manager: Database manager
        sample_observations: Sample observation data

    Yields:
        Database manager with populated data
    """
    import time
    import uuid

    # Add a session
    await db_manager.add_session(
        session_id="test-session",
        project="test-project",
        start_time=time.time(),
        goal="Test session",
    )

    # Add observations
    for obs in sample_observations:
        await db_manager.add_observation(
            obs_id=str(uuid.uuid4()),
            session_id="test-session",
            project=obs["project"],
            obs_type=obs["obs_type"],
            content=obs["content"],
            summary=obs["content"][:100],
            created_at=time.time(),
            tags=obs.get("tags", []),
        )

    yield db_manager


# =============================================================================
# Service Fixtures
# =============================================================================

@pytest.fixture
def app_config():
    """Create application config."""
    from ai_mem.config import AppConfig
    return AppConfig()


@pytest.fixture
def similarity_service(app_config):
    """Create similarity service."""
    from ai_mem.services.similarity import SimilarityService
    return SimilarityService(app_config)


@pytest.fixture
def cache_service(app_config):
    """Create cache service."""
    from ai_mem.services.cache import SearchCacheService
    return SearchCacheService(app_config)


@pytest.fixture
def reranking_service(app_config, similarity_service):
    """Create reranking service."""
    from ai_mem.services.reranking import RerankingService
    return RerankingService(app_config, similarity_service)


@pytest.fixture
async def session_service(db_manager, app_config):
    """Create session service."""
    from ai_mem.services.session import SessionService
    return SessionService(db_manager, app_config)


@pytest.fixture
async def observation_service(db_manager, app_config, session_service):
    """Create observation service."""
    from ai_mem.services.observation import ObservationService
    return ObservationService(db_manager, None, app_config, session_service)


@pytest.fixture
async def search_service(db_manager, app_config, cache_service, similarity_service, reranking_service):
    """Create search service."""
    from ai_mem.services.search import SearchService
    return SearchService(
        db_manager, None, None, app_config,
        cache_service, similarity_service, reranking_service
    )


@pytest.fixture
async def tag_service(db_manager):
    """Create tag service."""
    from ai_mem.services.tag import TagService
    return TagService(db_manager)

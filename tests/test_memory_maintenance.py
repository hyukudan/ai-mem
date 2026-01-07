"""Tests for Memory Consolidation, Cleanup, and Two-Stage Retrieval (Phase 3).

These tests verify the new memory maintenance features:
- Memory consolidation (finding and merging similar observations)
- Memory cleanup (removing stale observations)
- Two-stage retrieval with reranking
"""

import pytest
import tempfile
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ai_mem.db import DatabaseManager
from ai_mem.memory import MemoryManager
from ai_mem.models import Observation, Session


@pytest.fixture
async def db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = DatabaseManager(f"{tmpdir}/ai-mem.sqlite")
        await db.connect()
        yield db
        await db.close()


@pytest.fixture
async def manager():
    """Create a MemoryManager with temporary storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import os
        os.environ["AI_MEM_DB_PATH"] = f"{tmpdir}/ai-mem.sqlite"
        os.environ["AI_MEM_CHROMA_PATH"] = f"{tmpdir}/chroma"

        mgr = MemoryManager()
        await mgr.initialize()
        yield mgr
        await mgr.close()


# =============================================================================
# Database Layer Tests
# =============================================================================

@pytest.mark.asyncio
async def test_increment_access_count(db):
    """Test that access count is incremented correctly."""
    session = Session(project="test-project")
    await db.add_session(session)

    obs = Observation(
        session_id=session.id,
        project="test-project",
        type="note",
        content="Test observation",
        summary="Test observation",
    )
    await db.add_observation(obs)

    # Initially access_count should be 0 or None
    results = await db.get_observations([obs.id])
    assert len(results) == 1

    # Increment access count
    await db.increment_access_count(obs.id)

    # Check last_accessed_at was set (via SQL query)
    async with db.conn.execute(
        "SELECT access_count, last_accessed_at FROM observations WHERE id = ?",
        (obs.id,)
    ) as cursor:
        row = await cursor.fetchone()
        assert row["access_count"] == 1
        assert row["last_accessed_at"] is not None


@pytest.mark.asyncio
async def test_mark_superseded(db):
    """Test marking an observation as superseded by another."""
    session = Session(project="test-project")
    await db.add_session(session)

    obs1 = Observation(
        session_id=session.id,
        project="test-project",
        type="note",
        content="Old observation",
        summary="Old observation",
    )
    obs2 = Observation(
        session_id=session.id,
        project="test-project",
        type="note",
        content="New observation",
        summary="New observation",
    )
    await db.add_observation(obs1)
    await db.add_observation(obs2)

    # Mark obs1 as superseded by obs2
    result = await db.mark_superseded(obs1.id, obs2.id)
    assert result == 1

    # Verify the superseded_by field
    async with db.conn.execute(
        "SELECT superseded_by FROM observations WHERE id = ?",
        (obs1.id,)
    ) as cursor:
        row = await cursor.fetchone()
        assert row["superseded_by"] == obs2.id


@pytest.mark.asyncio
async def test_get_similar_observations(db):
    """Test retrieving observations for similarity analysis."""
    session = Session(project="test-project")
    await db.add_session(session)

    # Add several observations
    for i in range(5):
        obs = Observation(
            session_id=session.id,
            project="test-project",
            type="note",
            content=f"Test observation {i}",
            summary=f"Test observation {i}",
        )
        await db.add_observation(obs)

    results = await db.get_similar_observations(
        project="test-project",
        exclude_superseded=True,
        limit=10,
    )

    assert len(results) == 5
    assert all("id" in r for r in results)
    assert all("content" in r for r in results)


@pytest.mark.asyncio
async def test_get_stale_observations(db):
    """Test retrieving stale observations for cleanup."""
    session = Session(project="test-project")
    await db.add_session(session)

    # Create an old observation (91 days ago)
    old_time = time.time() - (91 * 86400)
    obs_old = Observation(
        session_id=session.id,
        project="test-project",
        type="note",
        content="Old observation",
        summary="Old observation",
        created_at=old_time,
    )
    await db.add_observation(obs_old)

    # Create a recent observation
    obs_new = Observation(
        session_id=session.id,
        project="test-project",
        type="note",
        content="New observation",
        summary="New observation",
    )
    await db.add_observation(obs_new)

    # Get stale observations (older than 90 days)
    stale = await db.get_stale_observations(
        project="test-project",
        max_age_days=90,
        min_access_count=0,
        limit=10,
    )

    assert len(stale) == 1
    assert stale[0]["id"] == obs_old.id


# =============================================================================
# Memory Manager Tests
# =============================================================================

@pytest.mark.asyncio
async def test_compute_text_similarity(manager):
    """Test Jaccard text similarity computation."""
    # Identical texts
    sim1 = manager._compute_text_similarity("hello world", "hello world")
    assert sim1 == 1.0

    # Completely different texts
    sim2 = manager._compute_text_similarity("hello world", "foo bar baz")
    assert sim2 == 0.0

    # Partial overlap
    sim3 = manager._compute_text_similarity("hello world foo", "hello world bar")
    assert 0.0 < sim3 < 1.0

    # Empty texts
    sim4 = manager._compute_text_similarity("", "hello")
    assert sim4 == 0.0


@pytest.mark.asyncio
async def test_compute_embedding_similarity(manager):
    """Test cosine similarity computation."""
    # Identical vectors
    vec1 = [1.0, 0.0, 0.0]
    sim1 = manager._compute_embedding_similarity(vec1, vec1)
    assert abs(sim1 - 1.0) < 0.001

    # Orthogonal vectors
    vec2 = [0.0, 1.0, 0.0]
    sim2 = manager._compute_embedding_similarity(vec1, vec2)
    assert abs(sim2) < 0.001

    # Empty vectors
    sim3 = manager._compute_embedding_similarity([], [1.0])
    assert sim3 == 0.0


@pytest.mark.asyncio
async def test_find_similar_observations(manager):
    """Test finding similar observations."""
    # Add some observations
    session = await manager.start_session(project="test-project", goal="test")

    # Add similar observations
    await manager.add_observation(
        content="Authentication using JWT tokens for user login",
        obs_type="note",
        project="test-project",
        session_id=session.id,
        summarize=False,
        summary="Authentication using JWT tokens for user login",
    )
    await manager.add_observation(
        content="Authentication using JWT tokens for secure login",
        obs_type="note",
        project="test-project",
        session_id=session.id,
        summarize=False,
        summary="Authentication using JWT tokens for secure login",
    )
    # Add a different observation
    await manager.add_observation(
        content="Database migration completed successfully",
        obs_type="note",
        project="test-project",
        session_id=session.id,
        summarize=False,
        summary="Database migration completed successfully",
    )

    # Find similar pairs with a lower threshold for text similarity
    pairs = await manager.find_similar_observations(
        project="test-project",
        similarity_threshold=0.5,  # Lower threshold for text-based similarity
        use_embeddings=False,  # Use text similarity for predictable results
        limit=10,
    )

    # Should find at least the two similar auth-related observations
    assert len(pairs) >= 1


@pytest.mark.asyncio
async def test_consolidate_memories_dry_run(manager):
    """Test consolidation in dry run mode."""
    session = await manager.start_session(project="test-project", goal="test")

    # Add similar (but not identical) observations to avoid deduplication
    obs1 = await manager.add_observation(
        content="Authentication using JWT tokens for secure user login flow",
        obs_type="note",
        project="test-project",
        session_id=session.id,
        summarize=False,
        summary="Authentication using JWT tokens for secure user login flow",
        dedupe=False,  # Disable deduplication
    )
    obs2 = await manager.add_observation(
        content="Authentication using JWT tokens for secure user login process",
        obs_type="note",
        project="test-project",
        session_id=session.id,
        summarize=False,
        summary="Authentication using JWT tokens for secure user login process",
        dedupe=False,  # Disable deduplication
    )

    # Run consolidation in dry run mode
    result = await manager.consolidate_memories(
        project="test-project",
        similarity_threshold=0.8,  # Lower threshold to catch similar content
        keep_strategy="newest",
        dry_run=True,
        limit=10,
    )

    # Should be a dry run
    assert result["dry_run"] is True

    # Both observations should still exist after dry run
    obs_data = await manager.get_observations([obs1.id, obs2.id])
    assert len(obs_data) == 2  # Both should still exist in dry run


@pytest.mark.asyncio
async def test_cleanup_stale_memories_dry_run(manager):
    """Test cleanup in dry run mode."""
    session = await manager.start_session(project="test-project", goal="test")

    # Add an observation
    await manager.add_observation(
        content="Test observation",
        obs_type="note",
        project="test-project",
        session_id=session.id,
        summarize=False,
    )

    # Cleanup with dry run (nothing should be old enough)
    result = await manager.cleanup_stale_memories(
        project="test-project",
        max_age_days=1,  # 1 day - nothing should match
        dry_run=True,
        limit=10,
    )

    assert result["dry_run"] is True
    assert result["candidates_found"] == 0


# =============================================================================
# Two-Stage Retrieval Tests
# =============================================================================

@pytest.mark.asyncio
async def test_rerank_results(manager):
    """Test reranking of search results."""
    from ai_mem.models import ObservationIndex

    # Create mock results
    results = [
        ObservationIndex(
            id="obs1",
            summary="Authentication JWT tokens",
            project="test",
            created_at=time.time(),
            score=0.7,
        ),
        ObservationIndex(
            id="obs2",
            summary="Database migration script",
            project="test",
            created_at=time.time(),
            score=0.8,
        ),
        ObservationIndex(
            id="obs3",
            summary="User authentication flow",
            project="test",
            created_at=time.time(),
            score=0.75,
        ),
    ]

    # Rerank with a query about authentication
    reranked = manager._rerank_results("authentication", results, top_k=2)

    # Should return top_k results
    assert len(reranked) == 2
    # Each result should have a rerank_score
    assert all(r.rerank_score is not None for r in reranked)


@pytest.mark.asyncio
async def test_search_with_rerank(manager):
    """Test two-stage search with reranking."""
    session = await manager.start_session(project="test-project", goal="test")

    # Add some observations
    await manager.add_observation(
        content="Implemented JWT authentication for API endpoints",
        obs_type="note",
        project="test-project",
        session_id=session.id,
        summarize=False,
        summary="Implemented JWT authentication for API endpoints",
    )
    await manager.add_observation(
        content="Fixed database connection pooling issue",
        obs_type="bugfix",
        project="test-project",
        session_id=session.id,
        summarize=False,
        summary="Fixed database connection pooling issue",
    )
    await manager.add_observation(
        content="Added user login endpoint with OAuth support",
        obs_type="feature",
        project="test-project",
        session_id=session.id,
        summarize=False,
        summary="Added user login endpoint with OAuth support",
    )

    # Search with reranking
    results = await manager.search_with_rerank(
        query="authentication",
        limit=2,
        project="test-project",
        stage1_limit=10,
    )

    assert len(results) <= 2
    # Results should have rerank scores
    for r in results:
        assert r.rerank_score is not None or r.score is not None

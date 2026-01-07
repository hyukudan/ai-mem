import pytest
from unittest import mock
from ai_mem.db import DatabaseManager
import os
import shutil
import time

@pytest.fixture
async def db_manager():
    test_dir = "/tmp/ai_mem_perf_test_" + str(time.time())
    db_path = os.path.join(test_dir, "test.sqlite")
    manager = DatabaseManager(db_path)
    await manager.connect()
    await manager.create_tables()
    yield manager
    await manager.close()
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

@pytest.mark.asyncio
async def test_batch_asset_fetching(db_manager):
    # Create N observations with assets
    N = 20
    
    # Manually insert to have control
    session_id = "sess1"
    await db_manager.conn.execute("INSERT INTO sessions (id, project, start_time) VALUES (?, ?, ?)", (session_id, "proj", time.time()))
    
    for i in range(N):
        obs_id = f"obs_{i}"
        await db_manager.conn.execute(
            "INSERT INTO observations (id, session_id, project, type, content, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (obs_id, session_id, "proj", "note", "content", time.time())
        )
        # Add asset
        await db_manager.add_asset(obs_id, "file", name=f"file_{i}", content="cnt")
    
    await db_manager.conn.commit()
    
    # Spy on get_assets_for_observations and get_assets_for_observation
    with mock.patch.object(db_manager, 'get_assets_for_observations', wraps=db_manager.get_assets_for_observations) as mock_batch:
        with mock.patch.object(db_manager, 'get_assets_for_observation', wraps=db_manager.get_assets_for_observation) as mock_single:
            
            # Fetch all observations
            ids = [f"obs_{i}" for i in range(N)]
            results = await db_manager.get_observations(ids)
            
            assert len(results) == N
            assert len(results[0]["assets"]) == 1
            
            # Should have called batch ONCE
            assert mock_batch.call_count == 1
            assert mock_single.call_count == 0

@pytest.mark.asyncio
async def test_indexes_exist(db_manager):
    async with db_manager.conn.execute("PRAGMA index_list(observations)") as cursor:
        rows = await cursor.fetchall()
        indexes = {row["name"] for row in rows}
        # Primary indexes
        assert "idx_observations_session" in indexes
        assert "idx_observations_project" in indexes
        assert "idx_observations_type" in indexes
        assert "idx_observations_created" in indexes
        assert "idx_observations_hash" in indexes
        # Composite indexes for common query patterns
        assert "idx_observations_project_created" in indexes
        assert "idx_observations_project_type" in indexes
        assert "idx_observations_session_created" in indexes
        assert "idx_observations_type_created" in indexes

    async with db_manager.conn.execute("PRAGMA index_list(sessions)") as cursor:
        rows = await cursor.fetchall()
        indexes = {row["name"] for row in rows}
        assert "idx_sessions_project" in indexes
        assert "idx_sessions_start" in indexes

    # Check event_idempotency indexes
    async with db_manager.conn.execute("PRAGMA index_list(event_idempotency)") as cursor:
        rows = await cursor.fetchall()
        indexes = {row["name"] for row in rows}
        assert "idx_event_idempotency_processed" in indexes
        assert "idx_event_idempotency_host" in indexes

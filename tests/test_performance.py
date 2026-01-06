import pytest
from unittest import mock
from ai_mem.db import DatabaseManager
import os
import shutil
import time

@pytest.fixture
def db_manager():
    test_dir = "/tmp/ai_mem_perf_test_" + str(time.time())
    db_path = os.path.join(test_dir, "test.sqlite")
    manager = DatabaseManager(db_path)
    yield manager
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def test_batch_asset_fetching(db_manager):
    # Create N observations with assets
    N = 20
    payloads = []
    
    # Manually insert to have control
    session_id = "sess1"
    db_manager.conn.execute("INSERT INTO sessions (id, project, start_time) VALUES (?, ?, ?)", (session_id, "proj", time.time()))
    
    for i in range(N):
        obs_id = f"obs_{i}"
        db_manager.conn.execute(
            "INSERT INTO observations (id, session_id, project, type, content, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (obs_id, session_id, "proj", "note", "content", time.time())
        )
        # Add asset
        db_manager.add_asset(obs_id, "file", name=f"file_{i}", content="cnt")
    
    db_manager.conn.commit()
    
    # Spy on get_assets_for_observations and get_assets_for_observation
    with mock.patch.object(db_manager, 'get_assets_for_observations', wraps=db_manager.get_assets_for_observations) as mock_batch:
        with mock.patch.object(db_manager, 'get_assets_for_observation', wraps=db_manager.get_assets_for_observation) as mock_single:
            
            # Fetch all observations
            ids = [f"obs_{i}" for i in range(N)]
            results = db_manager.get_observations(ids)
            
            assert len(results) == N
            assert len(results[0]["assets"]) == 1
            
            # Should have called batch ONCE
            assert mock_batch.call_count == 1
            # Should NOT have called single N times (it might call it 0 times if fully optimized)
            # Actually, _row_to_observation calls get_assets_for_observation ONLY if assets is None.
            # get_observations passes assets, so single should be 0.
            assert mock_single.call_count == 0

def test_indexes_exist(db_manager):
    cursor = db_manager.conn.cursor()
    cursor.execute("PRAGMA index_list(observations)")
    indexes = {row["name"] for row in cursor.fetchall()}
    assert "idx_observations_session" in indexes
    assert "idx_observations_project_created" in indexes
    
    cursor.execute("PRAGMA index_list(sessions)")
    indexes = {row["name"] for row in cursor.fetchall()}
    assert "idx_sessions_project" in indexes

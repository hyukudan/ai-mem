import pytest
import time
import os
import shutil
import json
from ai_mem.db import DatabaseManager

@pytest.fixture
async def db_manager():
    test_dir = "/tmp/ai_mem_tags_test_" + str(time.time())
    db_path = os.path.join(test_dir, "test.sqlite")
    manager = DatabaseManager(db_path)
    await manager.connect()
    await manager.create_tables()
    yield manager
    await manager.close()
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

@pytest.mark.asyncio
async def test_tag_filtering_fts(db_manager):
    # Insert observations
    # Obs 1: tag1, tag2
    await db_manager.conn.execute(
        "INSERT INTO sessions (id, project, start_time) VALUES (?, ?, ?)", 
        ("sess1", "proj", time.time())
    )
    
    await db_manager.conn.execute(
        """
        INSERT INTO observations (id, session_id, project, type, content, created_at, tags) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("obs1", "sess1", "proj", "note", "content1", time.time(), json.dumps(["tag1", "tag2"]))
    )
    
    # Obs 2: tag2
    await db_manager.conn.execute(
        """
        INSERT INTO observations (id, session_id, project, type, content, created_at, tags) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("obs2", "sess1", "proj", "note", "content2", time.time(), json.dumps(["tag2"]))
    )
    
    # Obs 3: tag3
    await db_manager.conn.execute(
        """
        INSERT INTO observations (id, session_id, project, type, content, created_at, tags) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("obs3", "sess1", "proj", "note", "content3", time.time(), json.dumps(["tag3"]))
    )
    
    await db_manager.conn.commit()
    
    # Filter by tag1 -> Expect obs1
    # list_observations calls _tag_clause
    results = await db_manager.list_observations(tag_filters=["tag1"])
    assert len(results) == 1
    assert results[0].id == "obs1"
    
    # Filter by tag2 -> Expect obs1, obs2
    results = await db_manager.list_observations(tag_filters=["tag2"])
    assert len(results) == 2
    ids = {r.id for r in results}
    assert "obs1" in ids
    assert "obs2" in ids
    
    # Filter by tag1 OR tag3 -> Expect obs1, obs3
    results = await db_manager.list_observations(tag_filters=["tag1", "tag3"])
    assert len(results) == 2
    ids = {r.id for r in results}
    assert "obs1" in ids
    assert "obs3" in ids
    
    # Filter by unknown tag -> Expect 0
    results = await db_manager.list_observations(tag_filters=["unknown"])
    assert len(results) == 0

@pytest.mark.asyncio
async def test_tag_quoting_robustness(db_manager):
    await db_manager.conn.execute(
        "INSERT INTO sessions (id, project, start_time) VALUES (?, ?, ?)", 
        ("sess1", "proj", time.time())
    )
    # Tag with special chars "c++"
    # Actually FTS standard tokenizer splits on non-alphanum. "c++" might be indexed as "c".
    # But let's check basic quoting like spaces if supported.
    # We rely on json structure in DB so "tag with space".
    
    await db_manager.conn.execute(
        """
        INSERT INTO observations (id, session_id, project, type, content, created_at, tags) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("obs_cpp", "sess1", "proj", "note", "content", time.time(), json.dumps(["tag with space"]))
    )
    await db_manager.conn.commit()
    
    results = await db_manager.list_observations(tag_filters=["tag with space"])
    # FTS5 phrase match requires quotes. Our code does quoted_tags = [f'"{tag}"' ...]
    # So "tag with space" becomes '"tag with space"'.
    # This should match the phrase in the JSON string `["tag with space"]`
    assert len(results) == 1
    assert results[0].id == "obs_cpp"

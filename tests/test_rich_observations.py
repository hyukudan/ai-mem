import pytest
import os
import shutil
import time
from ai_mem.memory import MemoryManager
from ai_mem.config import AppConfig

@pytest.fixture
async def test_manager():
    # Use a temporary directory for testing
    test_dir = "/tmp/ai_mem_rich_test_" + str(time.time())
    config = AppConfig()
    config.storage.data_dir = test_dir
    config.storage.sqlite_path = os.path.join(test_dir, "test_mem.sqlite")
    config.storage.vector_dir = os.path.join(test_dir, "test_vector")
    
    manager = MemoryManager(config)
    await manager.initialize()
    yield manager
    
    # Cleanup
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

@pytest.mark.asyncio
async def test_observation_with_diff(test_manager):
    obs = await test_manager.add_observation(
        content="Refactored the login function",
        obs_type="refactor",
        project="test-project",
        diff="--- a/login.py\n+++ b/login.py\n@@ -1,2 +1,3 @@\n def login():\n-    pass\n+    print('logging in')"
    )
    
    assert obs is not None
    assert obs.diff is not None
    assert "print('logging in')" in obs.diff
    
    # Verify retrieval from DB
    retrieved = await test_manager.db.get_observation(obs.id)
    assert retrieved is not None
    assert retrieved["diff"] == obs.diff

@pytest.mark.asyncio
async def test_observation_with_assets(test_manager):
    assets = [
        {
            "type": "file",
            "name": "logs.txt",
            "path": "/var/log/app.log",
            "content": "Error: Connection refused",
            "metadata": {"size": 1024}
        },
        {
            "type": "image",
            "name": "screenshot.png",
            "path": "/tmp/screenshot.png",
            # Binary content usually not stored in string field, but path is key
            "metadata": {"width": 1920, "height": 1080}
        }
    ]
    
    obs = await test_manager.add_observation(
        content="Analyzed crash logs",
        obs_type="bugfix",
        project="test-project",
        assets=assets
    )
    
    assert obs is not None
    assert len(obs.assets) == 2
    assert obs.assets[0].name == "logs.txt"
    assert obs.assets[0].content == "Error: Connection refused"
    
    # Verify retrieval
    retrieved = await test_manager.db.get_observation(obs.id)
    assert retrieved is not None
    retrieved_assets = retrieved["assets"]
    assert len(retrieved_assets) == 2
    
    # Check asset content persistence
    log_asset = next(a for a in retrieved_assets if a["name"] == "logs.txt")
    assert log_asset["content"] == "Error: Connection refused"
    assert log_asset["metadata"]["size"] == 1024

@pytest.mark.asyncio
async def test_observation_with_diff_and_assets(test_manager):
    obs = await test_manager.add_observation(
        content="Complex change",
        obs_type="feature",
        project="test-project",
        diff="diff --git a/file b/file",
        assets=[{"type": "file", "name": "spec.md"}]
    )
    
    retrieved = await test_manager.db.get_observation(obs.id)
    assert retrieved["diff"] == "diff --git a/file b/file"
    assert len(retrieved["assets"]) == 1

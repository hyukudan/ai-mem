import builtins
import json
import pytest
import tempfile
from unittest import mock
from typer.testing import CliRunner

from ai_mem.cli import app
from ai_mem.config import AppConfig, EmbeddingConfig, LLMConfig, StorageConfig
from ai_mem.mcp_server import MCPServer
from ai_mem.memory import MemoryManager


class DummyEmbeddingProvider:
    def embed(self, chunks):
        return [[0.0] * 3 for _ in chunks]


class DummyVectorStore:
    def __init__(self, *args, **kwargs):
        pass

    def add(self, **kwargs):
        pass

    def query(self, **kwargs):
        return {"metadatas": [[]]}
    
    def delete_where(self, where): return None
    def delete_ids(self, ids): return None


def _plain_console_print(*args, **kwargs):
    builtins.print(*args, **kwargs)


import asyncio

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmputils:
        yield tmputils


@pytest.fixture
def app_config(temp_dir):
    return AppConfig(
        llm=LLMConfig(provider="none"),
        embeddings=EmbeddingConfig(provider="fastembed"),
        storage=StorageConfig(data_dir=temp_dir),
    )


async def _populate_data(app_config, content, obs_type, project, summary, tags=None):
    with mock.patch("ai_mem.memory._build_embedding_provider", return_value=DummyEmbeddingProvider()), \
            mock.patch("ai_mem.memory.build_vector_store", return_value=DummyVectorStore()):
        manager = MemoryManager(app_config)
        await manager.initialize()
        obs = await manager.add_observation(
            content=content,
            obs_type=obs_type,
            project=project,
            summary=summary,
            tags=tags,
        )
        await manager.close()
        return obs.id


def test_context_command_exposes_scoreboard(app_config, temp_dir):
    # Retrieve ID via async setup run in new loop
    obs_id = asyncio.run(_populate_data(
        app_config,
        content="Shared memory for Claude and Gemini",
        obs_type="note",
        project="proj",
        summary="Cross-model memory",
        tags=["cross-model"]
    ))

    # Now run CLI
    runner = CliRunner()
    
    with mock.patch("ai_mem.cli.get_memory_manager", side_effect=lambda: MemoryManager(app_config)), \
         mock.patch("ai_mem.memory._build_embedding_provider", return_value=DummyEmbeddingProvider()), \
         mock.patch("ai_mem.memory.build_vector_store", return_value=DummyVectorStore()):
        
        result = runner.invoke(app, [
            "context", "--project", "proj", "--query", "memory", 
            "--format", "json", "--show-tokens"
        ])
    
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    metadata = payload.get("metadata", {})
    scoreboard = metadata.get("scoreboard") or {}
    assert obs_id in scoreboard


def test_endless_command_json_includes_scoreboard(app_config, temp_dir):
    obs_id = asyncio.run(_populate_data(
        app_config,
        content="Live stream memory",
        obs_type="note",
        project="proj",
        summary="Streamed memory",
    ))

    runner = CliRunner()
    with mock.patch("ai_mem.cli.get_memory_manager", side_effect=lambda: MemoryManager(app_config)), \
         mock.patch("ai_mem.memory._build_embedding_provider", return_value=DummyEmbeddingProvider()), \
         mock.patch("ai_mem.memory.build_vector_store", return_value=DummyVectorStore()), \
         mock.patch("ai_mem.cli.console.print", new=_plain_console_print), \
         mock.patch("ai_mem.cli.asyncio.sleep", side_effect=KeyboardInterrupt):
        
        result = runner.invoke(app, [
            "endless", "--project", "proj", "--query", "stream", 
            "--json", "--interval", "0", "--token-limit", "500"
        ])
        
    assert result.exit_code == 0, result.stdout
    stdout = result.stdout
    start = stdout.find("{")
    end = stdout.rfind("}")
    assert start != -1
    assert end != -1
    payload = json.loads(stdout[start : end + 1])
    metadata = payload.get("metadata") or {}
    scoreboard = metadata.get("scoreboard") or {}
    assert obs_id in scoreboard
    cache = payload.get("cache") or {}
    assert "hits" in cache


@pytest.mark.asyncio
async def test_mcp_search_tool_returns_scoreboard_and_cache(app_config):
    # Async test for MCP Server direct usage
    with mock.patch("ai_mem.memory._build_embedding_provider", return_value=DummyEmbeddingProvider()), \
            mock.patch("ai_mem.memory.build_vector_store", return_value=DummyVectorStore()):
        manager = MemoryManager(app_config)
        await manager.initialize()
        
        obs = await manager.add_observation(
            content="Shared cross-model knowledge",
            obs_type="note",
            project="proj",
            summary="Shared knowledge",
        )
        
        server = MCPServer()
        server.manager = manager
        
        result = await server.call_tool(
            "search",
            {
                "query": "shared",
                "project": "proj",
                "limit": 5,
                "show_tokens": True,
            },
        )
        
        await manager.close()
    
    assert "content" in result
    payload_text = result["content"][0]["text"]
    payload = json.loads(payload_text)
    scoreboard = payload.get("scoreboard") or {}
    assert obs.id in scoreboard
    cache = payload.get("cache", {})
    assert "hits" in cache
    assert "misses" in cache

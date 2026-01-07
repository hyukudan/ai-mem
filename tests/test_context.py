import pytest
from unittest import mock
import tempfile
from ai_mem.config import AppConfig, EmbeddingConfig, LLMConfig, StorageConfig
from ai_mem.context import build_context
from ai_mem.memory import MemoryManager


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


@pytest.mark.asyncio
async def test_context_includes_scoreboard_metadata(manager):
    obs = await manager.add_observation(
        content="Shared memory for Claude and Gemini",
        obs_type="note",
        project="proj",
        summary="Shared memory",
        tags=["cross-model"],
    )
    assert obs is not None

    context_text, meta = await build_context(
        manager,
        project="proj",
        query="shared memory",
        total_count=3,
        full_count=1,
        show_tokens=True,
    )
    assert "<ai-mem-context" in context_text  # May have mode attribute
    assert "tokens" in meta
    assert meta["tokens"]["index"] > 0
    assert meta["tokens"]["full"] > 0
    scoreboard = meta.get("scoreboard") or {}
    assert obs.id in scoreboard
    entry = scoreboard[obs.id]
    assert "fts_score" in entry
    assert "vector_score" in entry
    assert "recency_factor" in entry


@pytest.mark.asyncio
async def test_context_without_query_omits_scoreboard(manager):
    await manager.add_observation(
        content="Silent memory.",
        obs_type="note",
        project="proj",
        summary="Silent memory",
    )
    context_text, meta = await build_context(manager, project="proj")
    assert "<ai-mem-context" in context_text  # May have mode attribute
    assert not meta.get("scoreboard")

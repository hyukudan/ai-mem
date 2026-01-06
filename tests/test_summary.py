import pytest
import tempfile
from unittest import mock
from ai_mem.config import AppConfig, EmbeddingConfig, LLMConfig, StorageConfig
from ai_mem.memory import MemoryManager


class DummyEmbeddingProvider:
    def embed(self, chunks):
        return [[0.0] * 3 for _ in chunks]


class DummyVectorStore:
    def __init__(self, *args, **kwargs):
        pass

    def add(self, embeddings=None, documents=None, metadatas=None, ids=None):
        return None

    def query(self, embedding=None, n_results=0, where=None):
        return {"ids": [[]], "metadatas": [[]], "distances": [[]]}

    def delete_where(self, where):
        return None

    def delete_ids(self, ids):
        return None


@pytest.fixture
async def manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AppConfig(
            llm=LLMConfig(provider="none"),
            embeddings=EmbeddingConfig(provider="fastembed"),
            storage=StorageConfig(data_dir=tmpdir),
        )
        with mock.patch("ai_mem.memory._build_embedding_provider", return_value=DummyEmbeddingProvider()), \
            mock.patch("ai_mem.memory.build_vector_store", return_value=DummyVectorStore()):
            manager = MemoryManager(config)
            await manager.initialize()
            yield manager
            await manager.close()


@pytest.mark.asyncio
async def test_summarize_project_stores_summary(manager):
    await manager.add_observation(content="First note", obs_type="note", project="proj", summarize=False)
    await manager.add_observation(content="Second note", obs_type="note", project="proj", summarize=False)
    result = await manager.summarize_project(project="proj", limit=5, store=True)

    assert result is not None
    assert "First note" in result["summary"]
    obs = result["observation"]
    assert obs is not None
    assert obs.type == "summary"


@pytest.mark.asyncio
async def test_summarize_session(manager):
    session = await manager.start_session(project="proj")
    await manager.add_observation(content="Session note", obs_type="note", project="proj", summarize=False)
    await manager.add_observation(content="Other note", obs_type="note", project="proj", summarize=False)
    result = await manager.summarize_project(session_id=session.id, limit=5, store=True)

    assert result is not None
    assert "Session note" in result["summary"]
    obs = result["observation"]
    assert obs is not None
    assert obs.session_id == session.id
    stored_session = await manager.get_session(session.id)
    assert stored_session is not None
    assert "Session note" in (stored_session.get("summary") or "")

import tempfile
import unittest
import pytest
from unittest import mock

from ai_mem.config import AppConfig, EmbeddingConfig, LLMConfig, StorageConfig
from ai_mem.memory import MemoryManager
from ai_mem.privacy import strip_memory_tags


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


@pytest.mark.asyncio
async def test_strip_private_tag():
    cleaned, stripped = strip_memory_tags("Hello<private>secret</private>world")
    assert cleaned == "Helloworld"
    assert stripped

@pytest.mark.asyncio
async def test_strip_context_tag():
    cleaned, stripped = strip_memory_tags("<ai-mem-context>ctx</ai-mem-context>after")
    assert cleaned == "after"
    assert stripped

@pytest.mark.asyncio
async def test_skip_private_only_observation():
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
            obs = await manager.add_observation(
                content="<private>secret</private>",
                obs_type="note",
                project="proj",
            )
            assert obs is None

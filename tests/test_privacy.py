import tempfile
import unittest
from unittest import mock

from ai_mem.config import AppConfig, EmbeddingConfig, LLMConfig, StorageConfig
from ai_mem.memory import MemoryManager
from ai_mem.privacy import strip_memory_tags


class DummyEmbeddingProvider:
    def embed(self, chunks):
        return [[0.0] * 3 for _ in chunks]


class DummyVectorStore:
    def __init__(self, path: str, collection_name: str = "observations"):
        self.path = path
        self.collection_name = collection_name

    def add(self, embeddings, documents, metadatas, ids):
        return None

    def delete_where(self, where):
        return None


class PrivacyTagTests(unittest.TestCase):
    def test_strip_private_tag(self):
        cleaned, stripped = strip_memory_tags("Hello<private>secret</private>world")
        self.assertEqual(cleaned, "Helloworld")
        self.assertTrue(stripped)

    def test_strip_context_tag(self):
        cleaned, stripped = strip_memory_tags("<ai-mem-context>ctx</ai-mem-context>after")
        self.assertEqual(cleaned, "after")
        self.assertTrue(stripped)

    def test_skip_private_only_observation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(
                llm=LLMConfig(provider="none"),
                embeddings=EmbeddingConfig(provider="fastembed"),
                storage=StorageConfig(data_dir=tmpdir),
            )
            with mock.patch("ai_mem.memory._build_embedding_provider", return_value=DummyEmbeddingProvider()), \
                mock.patch("ai_mem.memory.VectorStore", DummyVectorStore):
                manager = MemoryManager(config)
                obs = manager.add_observation(
                    content="<private>secret</private>",
                    obs_type="note",
                    project="proj",
                )
                self.assertIsNone(obs)


if __name__ == "__main__":
    unittest.main()

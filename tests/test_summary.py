import tempfile
import unittest
from unittest import mock

from ai_mem.config import AppConfig, EmbeddingConfig, LLMConfig, StorageConfig
from ai_mem.memory import MemoryManager


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


class SummaryTests(unittest.TestCase):
    def test_summarize_project_stores_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(
                llm=LLMConfig(provider="none"),
                embeddings=EmbeddingConfig(provider="fastembed"),
                storage=StorageConfig(data_dir=tmpdir),
            )
            with mock.patch("ai_mem.memory._build_embedding_provider", return_value=DummyEmbeddingProvider()), \
                mock.patch("ai_mem.memory.VectorStore", DummyVectorStore):
                manager = MemoryManager(config)
                manager.add_observation(content="First note", obs_type="note", project="proj", summarize=False)
                manager.add_observation(content="Second note", obs_type="note", project="proj", summarize=False)
                result = manager.summarize_project(project="proj", limit=5, store=True)
                self.assertIsNotNone(result)
                self.assertIn("First note", result["summary"])
                obs = result["observation"]
                self.assertIsNotNone(obs)
                self.assertEqual(obs.type, "summary")

    def test_summarize_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(
                llm=LLMConfig(provider="none"),
                embeddings=EmbeddingConfig(provider="fastembed"),
                storage=StorageConfig(data_dir=tmpdir),
            )
            with mock.patch("ai_mem.memory._build_embedding_provider", return_value=DummyEmbeddingProvider()), \
                mock.patch("ai_mem.memory.VectorStore", DummyVectorStore):
                manager = MemoryManager(config)
                session = manager.start_session(project="proj")
                manager.add_observation(content="Session note", obs_type="note", project="proj", summarize=False)
                manager.add_observation(content="Other note", obs_type="note", project="proj", summarize=False)
                result = manager.summarize_project(session_id=session.id, limit=5, store=True)
                self.assertIsNotNone(result)
                self.assertIn("Session note", result["summary"])
                obs = result["observation"]
                self.assertIsNotNone(obs)
                self.assertEqual(obs.session_id, session.id)
                stored_session = manager.get_session(session.id)
                self.assertIsNotNone(stored_session)
                self.assertIn("Session note", stored_session.get("summary") or "")


if __name__ == "__main__":
    unittest.main()

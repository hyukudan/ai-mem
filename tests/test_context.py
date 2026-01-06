import tempfile
import unittest
from unittest import mock

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


class ContextMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        config = AppConfig(
            llm=LLMConfig(provider="none"),
            embeddings=EmbeddingConfig(provider="fastembed"),
            storage=StorageConfig(data_dir=self.tmpdir.name),
        )
        self.embed_patcher = mock.patch(
            "ai_mem.memory._build_embedding_provider", return_value=DummyEmbeddingProvider()
        )
        self.vector_patcher = mock.patch(
            "ai_mem.memory.build_vector_store", return_value=DummyVectorStore()
        )
        self.embed_patcher.start()
        self.vector_patcher.start()
        self.manager = MemoryManager(config)

    def tearDown(self):
        self.embed_patcher.stop()
        self.vector_patcher.stop()
        self.tmpdir.cleanup()

    def test_context_includes_scoreboard_metadata(self):
        obs = self.manager.add_observation(
            content="Shared memory for Claude and Gemini",
            obs_type="note",
            project="proj",
            summary="Shared memory",
            tags=["cross-model"],
        )
        self.assertIsNotNone(obs)

        context_text, meta = build_context(
            self.manager,
            project="proj",
            query="shared memory",
            total_count=3,
            full_count=1,
            show_tokens=True,
        )
        self.assertIn("<ai-mem-context>", context_text)
        self.assertIn("tokens", meta)
        self.assertGreater(meta["tokens"]["index"], 0)
        self.assertGreater(meta["tokens"]["full"], 0)
        scoreboard = meta.get("scoreboard") or {}
        self.assertIn(obs.id, scoreboard)
        entry = scoreboard[obs.id]
        self.assertIn("fts_score", entry)
        self.assertIn("vector_score", entry)
        self.assertIn("recency_factor", entry)

    def test_context_without_query_omits_scoreboard(self):
        self.manager.add_observation(
            content="Silent memory.",
            obs_type="note",
            project="proj",
            summary="Silent memory",
        )
        context_text, meta = build_context(self.manager, project="proj")
        self.assertIn("<ai-mem-context>", context_text)
        self.assertFalse(meta.get("scoreboard"))


if __name__ == "__main__":
    unittest.main()

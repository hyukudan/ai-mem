import builtins
import json
import tempfile
import unittest
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
    def add(self, **kwargs):
        pass

    def query(self, **kwargs):
        return {"metadatas": [[]]}


def _plain_console_print(*args, **kwargs):
    builtins.print(*args, **kwargs)


class BaseMemoryTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        config = AppConfig(
            llm=LLMConfig(provider="none"),
            embeddings=EmbeddingConfig(provider="fastembed"),
            storage=StorageConfig(data_dir=self.tmpdir.name),
        )
        self.embed_patch = mock.patch(
            "ai_mem.memory._build_embedding_provider", return_value=DummyEmbeddingProvider()
        )
        self.vector_patch = mock.patch(
            "ai_mem.memory.build_vector_store", return_value=DummyVectorStore()
        )
        self.embed_patch.start()
        self.vector_patch.start()
        self.manager = MemoryManager(config)

    def tearDown(self):
        self.embed_patch.stop()
        self.vector_patch.stop()
        self.tmpdir.cleanup()


class CLIMetadataTests(BaseMemoryTest):
    def setUp(self):
        super().setUp()
        self.runner = CliRunner()
        self.get_manager_patch = mock.patch(
            "ai_mem.cli.get_memory_manager", return_value=self.manager
        )
        self.get_manager_patch.start()

    def tearDown(self):
        self.get_manager_patch.stop()
        super().tearDown()

    def test_context_command_exposes_scoreboard(self):
        observation = self.manager.add_observation(
            content="Shared memory for Claude and Gemini",
            obs_type="note",
            project=self.tmpdir.name,
            summary="Cross-model memory",
            tags=["cross-model"],
        )
        result = self.runner.invoke(
            app,
            [
                "context",
                "--project",
                self.tmpdir.name,
                "--query",
                "memory",
                "--format",
                "json",
                "--show-tokens",
            ],
        )
        self.assertEqual(result.exit_code, 0, result.stdout)
        payload = json.loads(result.stdout)
        metadata = payload.get("metadata", {})
        scoreboard = metadata.get("scoreboard") or {}
        self.assertIn(observation.id, scoreboard)
        tokens = metadata.get("tokens", {})
        self.assertGreater(tokens.get("index", 0), 0)
        self.assertGreater(tokens.get("full", 0), 0)

    def test_endless_command_json_includes_scoreboard(self):
        observation = self.manager.add_observation(
            content="Live stream memory",
            obs_type="note",
            project=self.tmpdir.name,
            summary="Streamed memory",
        )
        with mock.patch("ai_mem.cli.console.print", new=_plain_console_print), \
            mock.patch("ai_mem.cli.time.sleep", side_effect=KeyboardInterrupt):
            result = self.runner.invoke(
                app,
                [
                    "endless",
                    "--project",
                    self.tmpdir.name,
                    "--query",
                    "stream",
                    "--json",
                    "--interval",
                    "0",
                    "--token-limit",
                    "500",
                ],
            )
        self.assertEqual(result.exit_code, 0, result.stdout)
        stdout = result.stdout
        start = stdout.find("{")
        end = stdout.rfind("}")
        self.assertNotEqual(start, -1, "JSON block missing from endless output")
        self.assertNotEqual(end, -1, "JSON block missing from endless output")
        payload = json.loads(stdout[start : end + 1])
        metadata = payload.get("metadata") or {}
        scoreboard = metadata.get("scoreboard") or {}
        self.assertIn(observation.id, scoreboard)
        cache = payload.get("cache") or {}
        self.assertIn("hits", cache)


class MCPServerMetadataTests(BaseMemoryTest):
    def setUp(self):
        super().setUp()
        self.manager_patch = mock.patch(
            "ai_mem.mcp_server.MemoryManager", return_value=self.manager
        )
        self.manager_patch.start()
        self.server = MCPServer()

    def tearDown(self):
        self.manager_patch.stop()
        super().tearDown()

    def test_search_tool_returns_scoreboard_and_cache(self):
        observation = self.manager.add_observation(
            content="Shared cross-model knowledge",
            obs_type="note",
            project=self.tmpdir.name,
            summary="Shared knowledge",
        )
        result = self.server.call_tool(
            "search",
            {
                "query": "shared",
                "project": self.tmpdir.name,
                "limit": 5,
                "show_tokens": True,
            },
        )
        self.assertIn("content", result)
        payload_text = result["content"][0]["text"]
        payload = json.loads(payload_text)
        scoreboard = payload.get("scoreboard") or {}
        self.assertIn(observation.id, scoreboard)
        cache = payload.get("cache", {})
        self.assertIn("hits", cache)
        self.assertIn("misses", cache)

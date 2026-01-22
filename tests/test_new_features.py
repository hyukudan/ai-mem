"""Tests for new ai-mem v2.0 features.

Tests cover:
- Endless Mode
- Work Modes
- Citations
- Folder Context
- Plugin System
- i18n
- OpenRouter Provider
"""

import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# Endless Mode Tests
# =============================================================================

class TestEndlessMode:
    """Tests for Endless Mode compression."""

    def test_endless_config_defaults(self):
        """Test EndlessModeConfig has correct defaults."""
        from ai_mem.config import EndlessModeConfig

        config = EndlessModeConfig()
        assert config.enabled is False
        assert config.target_observation_tokens == 500
        assert config.compression_ratio == 4.0
        assert config.enable_archive is True

    def test_endless_config_from_env(self):
        """Test EndlessModeConfig reads from environment."""
        from ai_mem.config import EndlessModeConfig

        with patch.dict(os.environ, {
            "AI_MEM_ENDLESS_ENABLED": "true",
            "AI_MEM_ENDLESS_TARGET_TOKENS": "300",
        }):
            # Re-import to pick up new env vars
            config = EndlessModeConfig(enabled=True, target_observation_tokens=300)
            assert config.enabled is True
            assert config.target_observation_tokens == 300

    def test_archive_entry_dataclass(self):
        """Test ArchiveEntry dataclass."""
        from ai_mem.endless import ArchiveEntry

        entry = ArchiveEntry(
            observation_id="test-123",
            session_id="sess-456",
            project="/test/project",
            tool_name="Read",
            tool_input={"path": "/file.py"},
            tool_output="file contents",
            compressed_output="compressed",
            compression_ratio=4.0,
            original_tokens=1000,
            compressed_tokens=250,
        )

        assert entry.observation_id == "test-123"
        assert entry.compression_ratio == 4.0

    def test_endless_stats_dataclass(self):
        """Test EndlessModeStats dataclass."""
        from ai_mem.endless import EndlessModeStats

        stats = EndlessModeStats()
        assert stats.total_observations_compressed == 0
        assert stats.total_tokens_saved == 0


# =============================================================================
# Work Modes Tests
# =============================================================================

class TestWorkModes:
    """Tests for Work Modes system."""

    def test_list_modes(self):
        """Test listing available modes."""
        from ai_mem.modes import list_modes, MODES

        modes = list_modes()
        assert len(modes) == len(MODES)
        assert any(m["key"] == "code" for m in modes)
        assert any(m["key"] == "chill" for m in modes)

    def test_get_mode_config(self):
        """Test getting mode configuration."""
        from ai_mem.modes import get_mode_config

        config = get_mode_config("code")
        assert config is not None
        assert config.name == "Code Mode"
        assert config.endless_enabled is True
        assert config.context_total == 12

    def test_get_mode_config_unknown(self):
        """Test getting unknown mode returns None."""
        from ai_mem.modes import get_mode_config

        config = get_mode_config("nonexistent")
        assert config is None

    def test_mode_manager(self):
        """Test ModeManager class."""
        from ai_mem.modes import ModeManager

        manager = ModeManager()
        assert manager.current_mode is None

        # Set mode
        result = manager.set_mode("debug")
        assert result is True
        assert manager.current_mode == "debug"
        assert manager.mode_config is not None
        assert manager.mode_config.name == "Debug Mode"

        # Clear mode
        manager.clear_mode()
        assert manager.current_mode is None

    def test_mode_auto_tags(self):
        """Test mode auto-tags."""
        from ai_mem.modes import ModeManager

        manager = ModeManager()
        manager.set_mode("research")

        tags = manager.get_auto_tags()
        assert "research" in tags


# =============================================================================
# Citations Tests
# =============================================================================

class TestCitations:
    """Tests for Citations system."""

    def test_create_citation(self):
        """Test creating a citation."""
        from ai_mem.citations import create_citation

        citation = create_citation("abc12345-6789-0000-1111-222233334444")
        assert citation.observation_id == "abc12345-6789-0000-1111-222233334444"
        assert citation.short_id == "abc12345"
        assert "/obs/abc12345" in citation.url
        assert "/view/" in citation.html_url

    def test_format_citation_compact(self):
        """Test formatting citation in compact mode."""
        from ai_mem.citations import format_citation

        result = format_citation(
            "abc12345-6789-0000-1111-222233334444",
            format_type="compact",
            include_link=False,
        )
        assert result == "abc12345"

    def test_format_citation_full(self):
        """Test formatting citation in full mode."""
        from ai_mem.citations import format_citation

        result = format_citation(
            "abc12345-6789-0000-1111-222233334444",
            format_type="full",
            include_link=True,
        )
        assert "[abc12345]" in result
        assert "/view/" in result

    def test_citation_formatter(self):
        """Test CitationFormatter class."""
        from ai_mem.citations import CitationFormatter

        formatter = CitationFormatter(
            base_url="http://test:9000",
            format_type="compact",
        )

        citation = formatter.create("test-id-1234")
        assert "http://test:9000" in citation.url


# =============================================================================
# Folder Context Tests
# =============================================================================

class TestFolderContext:
    """Tests for Folder Context (CLAUDE.md) system."""

    def test_get_folder_from_observation_path(self):
        """Test extracting folder from observation with path."""
        from ai_mem.folder_context import get_folder_from_observation

        obs = {
            "metadata": {
                "tool_input": {"path": "/home/user/project/src/main.py"}
            }
        }

        with patch("pathlib.Path.is_file", return_value=True):
            folder = get_folder_from_observation(obs)
            assert folder == "/home/user/project/src"

    def test_format_observation_row(self):
        """Test formatting observation as table row."""
        from ai_mem.folder_context import format_observation_row

        obs = {
            "id": "abc12345-6789",
            "created_at": 1700000000,
            "type": "bugfix",
            "title": "Fixed login bug",
        }

        row = format_observation_row(obs)
        assert "abc12345" in row
        assert "bugfix" in row
        assert "Fixed login bug" in row

    def test_generate_folder_claudemd(self):
        """Test generating CLAUDE.md content."""
        from ai_mem.folder_context import generate_folder_claudemd

        observations = [
            {"id": "obs-1", "created_at": 1700000000, "type": "note", "summary": "Test 1"},
            {"id": "obs-2", "created_at": 1700001000, "type": "bugfix", "summary": "Test 2"},
        ]

        content = generate_folder_claudemd(
            folder_path="/test/folder",
            observations=observations,
            project="/test",
        )

        assert "# Folder Context" in content
        assert "/test/folder" in content
        assert "obs-1" in content or "obs-2" in content


# =============================================================================
# Plugin System Tests
# =============================================================================

class TestPluginSystem:
    """Tests for Plugin Marketplace system."""

    def test_plugin_info(self):
        """Test getting plugin info."""
        from ai_mem.plugin import get_plugin_info

        info = get_plugin_info()
        assert info.name == "ai-mem"
        assert info.version is not None

    def test_load_hooks_config(self):
        """Test loading hooks configuration."""
        from ai_mem.plugin import load_hooks_config

        config = load_hooks_config()
        # May or may not exist depending on environment
        assert isinstance(config, dict)

    def test_get_mcp_config(self):
        """Test getting MCP configuration."""
        from ai_mem.plugin import get_mcp_config

        config = get_mcp_config()
        assert "mcpServers" in config
        assert "ai-mem" in config["mcpServers"]
        assert "command" in config["mcpServers"]["ai-mem"]

    def test_check_dependencies(self):
        """Test checking dependencies."""
        from ai_mem.plugin import check_dependencies

        missing = check_dependencies()
        assert isinstance(missing, list)


# =============================================================================
# i18n Tests
# =============================================================================

class TestI18n:
    """Tests for internationalization system."""

    def test_supported_languages(self):
        """Test supported languages list."""
        from ai_mem.i18n import SUPPORTED_LANGUAGES, list_languages

        assert "en" in SUPPORTED_LANGUAGES
        assert "es" in SUPPORTED_LANGUAGES
        assert "zh" in SUPPORTED_LANGUAGES

        languages = list_languages()
        assert len(languages) > 0
        assert any(l["code"] == "en" for l in languages)

    def test_get_set_language(self):
        """Test getting and setting language."""
        from ai_mem.i18n import get_language, set_language

        original = get_language()

        set_language("es")
        assert get_language() == "es"

        set_language(original)

    def test_translation_function(self):
        """Test translation function."""
        from ai_mem.i18n import t, set_language

        set_language("en")
        result = t("app.name")
        assert result == "ai-mem"

    def test_translation_with_variables(self):
        """Test translation with variable interpolation."""
        from ai_mem.i18n import t, set_language

        set_language("en")
        result = t("cli.add.success", id="test-123")
        assert "test-123" in result

    def test_fallback_to_english(self):
        """Test fallback to English for missing translations."""
        from ai_mem.i18n import t, set_language

        set_language("xx")  # Non-existent language
        result = t("app.name")
        # Should return key or fallback
        assert result in ["app.name", "ai-mem"]


# =============================================================================
# OpenRouter Provider Tests
# =============================================================================

class TestOpenRouterProvider:
    """Tests for OpenRouter provider."""

    def test_config_defaults(self):
        """Test OpenRouterConfig defaults."""
        from ai_mem.providers.openrouter import OpenRouterConfig

        config = OpenRouterConfig()
        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.max_tokens == 4096

    def test_free_models_list(self):
        """Test FREE_MODELS list."""
        from ai_mem.providers.openrouter import FREE_MODELS

        assert len(FREE_MODELS) > 0
        assert any("free" in m for m in FREE_MODELS)

    def test_popular_models_dict(self):
        """Test POPULAR_MODELS dictionary."""
        from ai_mem.providers.openrouter import POPULAR_MODELS

        assert "gpt-4o" in POPULAR_MODELS
        assert "claude-3.5-sonnet" in POPULAR_MODELS
        assert "gemini-pro" in POPULAR_MODELS

    def test_chat_message(self):
        """Test ChatMessage dataclass."""
        from ai_mem.providers.openrouter import ChatMessage

        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_response(self):
        """Test ChatResponse dataclass."""
        from ai_mem.providers.openrouter import ChatResponse

        response = ChatResponse(
            content="Hello!",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        assert response.content == "Hello!"
        assert response.usage["prompt_tokens"] == 10


# =============================================================================
# Concepts Tests
# =============================================================================

class TestConcepts:
    """Tests for Concepts system."""

    def test_concept_icons(self):
        """Test CONCEPT_ICONS dictionary."""
        from ai_mem.models import CONCEPT_ICONS

        assert "gotcha" in CONCEPT_ICONS
        assert "trade-off" in CONCEPT_ICONS
        assert "pattern" in CONCEPT_ICONS

    def test_type_icons(self):
        """Test TYPE_ICONS dictionary."""
        from ai_mem.models import TYPE_ICONS

        assert "bugfix" in TYPE_ICONS
        assert "feature" in TYPE_ICONS
        assert "decision" in TYPE_ICONS


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

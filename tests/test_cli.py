"""Tests for the CLI commands.

These tests verify the CLI functionality works correctly by invoking
commands through typer's testing interface.
"""

import os
import tempfile
from unittest.mock import patch

from typer.testing import CliRunner

from ai_mem.cli import app


runner = CliRunner()


# =============================================================================
# Test: Help commands (no mocking needed)
# =============================================================================


def test_app_help():
    """Test the app help displays correctly."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AI Memory" in result.stdout


def test_add_help():
    """Test the add command help."""
    result = runner.invoke(app, ["add", "--help"])
    assert result.exit_code == 0
    assert "content" in result.stdout.lower()
    assert "obs-type" in result.stdout


def test_search_help():
    """Test the search command help."""
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0
    assert "query" in result.stdout.lower()
    assert "limit" in result.stdout.lower()


def test_context_help():
    """Test the context command help."""
    result = runner.invoke(app, ["context", "--help"])
    assert result.exit_code == 0
    assert "query" in result.stdout.lower()


def test_stats_help():
    """Test the stats command help."""
    result = runner.invoke(app, ["stats", "--help"])
    assert result.exit_code == 0
    assert "project" in result.stdout.lower()


def test_tags_help():
    """Test the tags command help."""
    result = runner.invoke(app, ["tags", "--help"])
    assert result.exit_code == 0


def test_sessions_help():
    """Test the sessions command help."""
    result = runner.invoke(app, ["sessions", "--help"])
    assert result.exit_code == 0


def test_session_start_help():
    """Test the session-start command help."""
    result = runner.invoke(app, ["session-start", "--help"])
    assert result.exit_code == 0
    assert "project" in result.stdout.lower()


def test_session_end_help():
    """Test the session-end command help."""
    result = runner.invoke(app, ["session-end", "--help"])
    assert result.exit_code == 0


def test_delete_help():
    """Test the delete command help."""
    result = runner.invoke(app, ["delete", "--help"])
    assert result.exit_code == 0
    assert "obs-id" in result.stdout.lower() or "observation" in result.stdout.lower()


def test_export_help():
    """Test the export command help."""
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0


def test_import_help():
    """Test the import command help."""
    result = runner.invoke(app, ["import", "--help"])
    assert result.exit_code == 0


def test_get_help():
    """Test the get command help."""
    result = runner.invoke(app, ["get", "--help"])
    assert result.exit_code == 0


def test_tag_add_help():
    """Test the tag-add command help."""
    result = runner.invoke(app, ["tag-add", "--help"])
    assert result.exit_code == 0


def test_tag_rename_help():
    """Test the tag-rename command help."""
    result = runner.invoke(app, ["tag-rename", "--help"])
    assert result.exit_code == 0


def test_tag_delete_help():
    """Test the tag-delete command help."""
    result = runner.invoke(app, ["tag-delete", "--help"])
    assert result.exit_code == 0


def test_cleanup_events_help():
    """Test the cleanup-events command help."""
    result = runner.invoke(app, ["cleanup-events", "--help"])
    assert result.exit_code == 0
    assert "days" in result.stdout.lower()


def test_ingest_help():
    """Test the ingest command help."""
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0


def test_hook_help():
    """Test the hook command help."""
    result = runner.invoke(app, ["hook", "--help"])
    assert result.exit_code == 0


def test_server_help():
    """Test the server command help."""
    result = runner.invoke(app, ["server", "--help"])
    assert result.exit_code == 0
    assert "port" in result.stdout.lower()


def test_mcp_help():
    """Test the mcp command help."""
    result = runner.invoke(app, ["mcp", "--help"])
    assert result.exit_code == 0


def test_proxy_help():
    """Test the proxy command help."""
    result = runner.invoke(app, ["proxy", "--help"])
    assert result.exit_code == 0


# =============================================================================
# Test: Config commands (minimal mocking)
# =============================================================================


def test_mcp_config_command():
    """Test the mcp-config command shows MCP configuration."""
    result = runner.invoke(app, ["mcp-config"])
    assert result.exit_code == 0
    # Should output JSON configuration
    assert "{" in result.stdout or "ai-mem" in result.stdout.lower()


def test_hook_config_command():
    """Test the hook-config command shows hook configuration."""
    result = runner.invoke(app, ["hook-config"])
    assert result.exit_code == 0


def test_config_command_show():
    """Test the config command shows configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = {"AI_MEM_DATA_DIR": tmpdir}
        with patch.dict(os.environ, env):
            result = runner.invoke(app, ["config"])
            assert result.exit_code == 0


# =============================================================================
# Test: Command argument validation
# =============================================================================


def test_add_requires_content():
    """Test the add command requires content argument."""
    result = runner.invoke(app, ["add"])
    assert result.exit_code != 0


def test_search_requires_query():
    """Test the search command requires query argument."""
    result = runner.invoke(app, ["search"])
    assert result.exit_code != 0


def test_delete_requires_obs_id():
    """Test the delete command requires obs-id argument."""
    result = runner.invoke(app, ["delete"])
    assert result.exit_code != 0


def test_get_requires_obs_id():
    """Test the get command requires obs-id argument."""
    result = runner.invoke(app, ["get"])
    assert result.exit_code != 0


def test_tag_add_requires_tag():
    """Test the tag-add command requires tag argument."""
    result = runner.invoke(app, ["tag-add"])
    assert result.exit_code != 0


def test_tag_rename_requires_args():
    """Test the tag-rename command requires old and new tag arguments."""
    result = runner.invoke(app, ["tag-rename"])
    assert result.exit_code != 0


def test_tag_delete_requires_tag():
    """Test the tag-delete command requires tag argument."""
    result = runner.invoke(app, ["tag-delete"])
    assert result.exit_code != 0


# =============================================================================
# Test: Snapshot subcommands
# =============================================================================


def test_snapshot_help():
    """Test the snapshot command help."""
    result = runner.invoke(app, ["snapshot", "--help"])
    assert result.exit_code == 0
    assert "export" in result.stdout.lower()
    assert "import" in result.stdout.lower()


def test_snapshot_export_help():
    """Test the snapshot export command help."""
    result = runner.invoke(app, ["snapshot", "export", "--help"])
    assert result.exit_code == 0


def test_snapshot_import_help():
    """Test the snapshot import command help."""
    result = runner.invoke(app, ["snapshot", "import", "--help"])
    assert result.exit_code == 0


# =============================================================================
# Test: Proxy commands help
# =============================================================================


def test_gemini_proxy_help():
    """Test the gemini-proxy command help."""
    result = runner.invoke(app, ["gemini-proxy", "--help"])
    assert result.exit_code == 0


def test_anthropic_proxy_help():
    """Test the anthropic-proxy command help."""
    result = runner.invoke(app, ["anthropic-proxy", "--help"])
    assert result.exit_code == 0


def test_azure_proxy_help():
    """Test the azure-proxy command help."""
    result = runner.invoke(app, ["azure-proxy", "--help"])
    assert result.exit_code == 0


def test_bedrock_proxy_help():
    """Test the bedrock-proxy command help."""
    result = runner.invoke(app, ["bedrock-proxy", "--help"])
    assert result.exit_code == 0


# =============================================================================
# Test: Command options are recognized
# =============================================================================


def test_search_options_recognized():
    """Test that search command options are recognized."""
    # Should not fail due to unrecognized option
    result = runner.invoke(app, ["search", "--help"])
    assert "--limit" in result.stdout
    assert "--project" in result.stdout
    assert "--obs-type" in result.stdout
    assert "--tag" in result.stdout
    assert "--since" in result.stdout


def test_add_options_recognized():
    """Test that add command options are recognized."""
    result = runner.invoke(app, ["add", "--help"])
    assert "--obs-type" in result.stdout
    assert "--project" in result.stdout
    assert "--tag" in result.stdout
    assert "--event-id" in result.stdout
    assert "--host" in result.stdout


def test_context_options_recognized():
    """Test that context command options are recognized."""
    result = runner.invoke(app, ["context", "--help"])
    assert "--query" in result.stdout
    assert "--project" in result.stdout
    assert "--total" in result.stdout


def test_stats_options_recognized():
    """Test that stats command options are recognized."""
    result = runner.invoke(app, ["stats", "--help"])
    assert "--project" in result.stdout
    assert "--format" in result.stdout
    assert "--tag" in result.stdout


def test_tags_options_recognized():
    """Test that tags command options are recognized."""
    result = runner.invoke(app, ["tags", "--help"])
    assert "--project" in result.stdout or "--limit" in result.stdout


# =============================================================================
# Test: Invalid command handling
# =============================================================================


def test_invalid_command():
    """Test that invalid command returns error."""
    result = runner.invoke(app, ["nonexistent-command"])
    assert result.exit_code != 0


def test_invalid_option():
    """Test that invalid option returns error."""
    result = runner.invoke(app, ["search", "query", "--invalid-option", "value"])
    assert result.exit_code != 0

"""Plugin System - Claude Code Marketplace compatibility.

This module provides compatibility with Claude Code's plugin marketplace,
enabling installation, update, and lifecycle management.

Usage:
    # Via Claude Code Marketplace
    /plugin marketplace add sergiocayuqueo/ai-mem
    /plugin install ai-mem

    # Or directly
    pip install ai-mem
    ai-mem init
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logging_config import get_logger
from . import __version__

logger = get_logger("plugin")

# Plugin paths
PLUGIN_DIR = Path.home() / ".ai-mem"
VERSION_FILE = PLUGIN_DIR / ".install-version"
CACHE_DIR = PLUGIN_DIR / "cache"
HOOKS_FILE = Path(__file__).parent.parent.parent / "hooks.json"


@dataclass
class PluginInfo:
    """Information about the installed plugin."""

    name: str
    version: str
    installed_at: str
    updated_at: Optional[str]
    hooks_enabled: bool
    mcp_enabled: bool


def get_plugin_info() -> PluginInfo:
    """Get information about the installed plugin.

    Returns:
        PluginInfo object
    """
    version_data = {}
    if VERSION_FILE.exists():
        try:
            version_data = json.loads(VERSION_FILE.read_text())
        except (json.JSONDecodeError, OSError) as e:
            # Graceful degradation - use defaults if file is corrupted
            logger.debug(f"Could not read version file: {e}")

    return PluginInfo(
        name="ai-mem",
        version=__version__,
        installed_at=version_data.get("installed_at", "unknown"),
        updated_at=version_data.get("updated_at"),
        hooks_enabled=version_data.get("hooks_enabled", False),
        mcp_enabled=version_data.get("mcp_enabled", False),
    )


def save_version_info(
    hooks_enabled: bool = False,
    mcp_enabled: bool = False,
) -> None:
    """Save version and installation info.

    Args:
        hooks_enabled: Whether hooks are enabled
        mcp_enabled: Whether MCP is enabled
    """
    PLUGIN_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing data
    version_data = {}
    if VERSION_FILE.exists():
        try:
            version_data = json.loads(VERSION_FILE.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.debug(f"Could not read existing version file: {e}")

    # Update data
    now = datetime.utcnow().isoformat() + "Z"

    if "installed_at" not in version_data:
        version_data["installed_at"] = now

    version_data.update({
        "version": __version__,
        "updated_at": now,
        "hooks_enabled": hooks_enabled,
        "mcp_enabled": mcp_enabled,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    })

    VERSION_FILE.write_text(json.dumps(version_data, indent=2))
    logger.info(f"Saved version info: {__version__}")


def load_hooks_config() -> Dict[str, Any]:
    """Load hooks configuration from hooks.json.

    Returns:
        Hooks configuration dictionary
    """
    if not HOOKS_FILE.exists():
        logger.warning("hooks.json not found")
        return {}

    try:
        return json.loads(HOOKS_FILE.read_text())
    except Exception as e:
        logger.error(f"Failed to load hooks.json: {e}")
        return {}


def get_mcp_config() -> Dict[str, Any]:
    """Get MCP server configuration for Claude Code.

    Returns:
        MCP configuration dictionary
    """
    return {
        "mcpServers": {
            "ai-mem": {
                "command": "ai-mem",
                "args": ["mcp"],
                "env": {
                    "AI_MEM_LOG_LEVEL": os.environ.get("AI_MEM_LOG_LEVEL", "INFO"),
                },
            }
        }
    }


def generate_claude_config() -> str:
    """Generate Claude Code configuration snippet.

    Returns:
        JSON configuration string
    """
    config = get_mcp_config()
    return json.dumps(config, indent=2)


def check_dependencies() -> List[str]:
    """Check for missing dependencies.

    Returns:
        List of missing dependencies
    """
    missing = []

    # Check Python version
    if sys.version_info < (3, 10):
        missing.append(f"Python 3.10+ (current: {sys.version_info.major}.{sys.version_info.minor})")

    # Check optional dependencies
    optional_deps = [
        ("chromadb", "Vector search"),
        ("sentence_transformers", "Embeddings"),
    ]

    for module, feature in optional_deps:
        try:
            __import__(module)
        except ImportError:
            missing.append(f"{module} ({feature})")

    return missing


def install(
    enable_hooks: bool = True,
    enable_mcp: bool = True,
    quiet: bool = False,
) -> bool:
    """Run plugin installation.

    Args:
        enable_hooks: Whether to enable hooks
        enable_mcp: Whether to enable MCP
        quiet: Suppress output

    Returns:
        True if successful
    """
    try:
        # Create directories
        PLUGIN_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Save version info
        save_version_info(
            hooks_enabled=enable_hooks,
            mcp_enabled=enable_mcp,
        )

        # Initialize database
        from .db import init_db
        import asyncio

        asyncio.get_event_loop().run_until_complete(init_db())

        if not quiet:
            print(f"ai-mem v{__version__} installed successfully")
            print(f"  Data directory: {PLUGIN_DIR}")

            missing = check_dependencies()
            if missing:
                print("\n  Optional dependencies not installed:")
                for dep in missing:
                    print(f"    - {dep}")

            print("\n  To configure Claude Code, add to settings:")
            print(generate_claude_config())

        return True

    except Exception as e:
        logger.error(f"Installation failed: {e}")
        if not quiet:
            print(f"Installation failed: {e}")
        return False


def update(quiet: bool = False) -> bool:
    """Run plugin update.

    Args:
        quiet: Suppress output

    Returns:
        True if successful
    """
    try:
        # Get current info
        info = get_plugin_info()

        # Run migrations
        from .db import migrate_db
        import asyncio

        asyncio.get_event_loop().run_until_complete(migrate_db())

        # Update version info
        save_version_info(
            hooks_enabled=info.hooks_enabled,
            mcp_enabled=info.mcp_enabled,
        )

        if not quiet:
            print(f"ai-mem updated to v{__version__}")

        return True

    except Exception as e:
        logger.error(f"Update failed: {e}")
        if not quiet:
            print(f"Update failed: {e}")
        return False


def uninstall(quiet: bool = False) -> bool:
    """Run plugin uninstall.

    Args:
        quiet: Suppress output

    Returns:
        True if successful
    """
    try:
        import shutil

        # Ask for confirmation
        if not quiet:
            confirm = input(f"Remove all ai-mem data from {PLUGIN_DIR}? [y/N]: ")
            if confirm.lower() != "y":
                print("Aborted")
                return False

        # Remove data directory
        if PLUGIN_DIR.exists():
            shutil.rmtree(PLUGIN_DIR)

        if not quiet:
            print("ai-mem uninstalled successfully")
            print("Note: Run 'pip uninstall ai-mem' to remove the package")

        return True

    except Exception as e:
        logger.error(f"Uninstall failed: {e}")
        if not quiet:
            print(f"Uninstall failed: {e}")
        return False


def verify() -> Dict[str, Any]:
    """Verify plugin installation.

    Returns:
        Verification results
    """
    results = {
        "version": __version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "data_dir_exists": PLUGIN_DIR.exists(),
        "version_file_exists": VERSION_FILE.exists(),
        "hooks_file_exists": HOOKS_FILE.exists(),
        "missing_dependencies": check_dependencies(),
        "status": "ok",
    }

    # Check database
    try:
        from .db import Database
        import asyncio

        db = Database()
        asyncio.get_event_loop().run_until_complete(db.connect())
        results["database"] = "ok"
        asyncio.get_event_loop().run_until_complete(db.close())
    except Exception as e:
        results["database"] = f"error: {e}"
        results["status"] = "error"

    # Check MCP
    try:
        from .mcp_server import AiMemMCPServer
        results["mcp_server"] = "ok"
    except Exception as e:
        results["mcp_server"] = f"error: {e}"

    return results


def generate_bug_report() -> str:
    """Generate a bug report with diagnostic information.

    Returns:
        Bug report text
    """
    import platform

    lines = [
        "# ai-mem Bug Report",
        "",
        "## Environment",
        f"- ai-mem version: {__version__}",
        f"- Python: {sys.version}",
        f"- Platform: {platform.platform()}",
        f"- OS: {platform.system()} {platform.release()}",
        "",
        "## Installation",
    ]

    # Verification results
    results = verify()
    for key, value in results.items():
        lines.append(f"- {key}: {value}")

    # Configuration
    lines.extend([
        "",
        "## Configuration",
    ])

    env_vars = [v for v in os.environ if v.startswith("AI_MEM_")]
    if env_vars:
        for var in sorted(env_vars):
            # Mask sensitive values
            value = os.environ[var]
            if "key" in var.lower() or "secret" in var.lower() or "token" in var.lower():
                value = "***"
            lines.append(f"- {var}: {value}")
    else:
        lines.append("- (no AI_MEM_* environment variables set)")

    # Hooks configuration
    lines.extend([
        "",
        "## Hooks Configuration",
    ])

    hooks = load_hooks_config()
    if hooks:
        lines.append(f"- Name: {hooks.get('name', 'unknown')}")
        lines.append(f"- Version: {hooks.get('version', 'unknown')}")
        lines.append(f"- Hooks defined: {len(hooks.get('hooks', {}))}")
    else:
        lines.append("- hooks.json not found")

    # Recent logs
    log_file = PLUGIN_DIR / "ai-mem.log"
    if log_file.exists():
        lines.extend([
            "",
            "## Recent Logs (last 20 lines)",
            "```",
        ])
        try:
            with open(log_file) as f:
                log_lines = f.readlines()
                lines.extend([l.rstrip() for l in log_lines[-20:]])
        except Exception as e:
            lines.append(f"Error reading logs: {e}")
        lines.append("```")

    lines.extend([
        "",
        "---",
        f"Generated: {datetime.utcnow().isoformat()}Z",
    ])

    return "\n".join(lines)

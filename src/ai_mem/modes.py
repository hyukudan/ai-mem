"""Work Modes - Predefined configurations for different workflows.

This module provides predefined modes that adjust ai-mem behavior:
- Code Mode: Optimized for software development
- Email Investigation Mode: For analyzing communications
- Chill Mode: Casual conversation, minimal memory

Configuration:
    AI_MEM_MODE=code|email|chill|custom
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import AppConfig, ContextConfig, load_config
from .logging_config import get_logger

logger = get_logger("modes")


@dataclass
class ModeConfig:
    """Configuration for a work mode."""

    name: str
    description: str
    # Context settings
    context_total: int
    context_full: int
    # Observation types to prioritize
    priority_types: List[str]
    # Concepts to prioritize
    priority_concepts: List[str]
    # Tags to auto-apply
    auto_tags: List[str]
    # Endless mode settings
    endless_enabled: bool
    compression_ratio: float
    # Search settings
    fts_weight: float
    vector_weight: float
    recency_weight: float
    # Disclosure mode
    disclosure_mode: str
    # Description for context injection
    context_description: str


# Predefined modes
MODES: Dict[str, ModeConfig] = {
    "code": ModeConfig(
        name="Code Mode",
        description="Optimized for software development and coding tasks",
        context_total=12,
        context_full=4,
        priority_types=["bugfix", "feature", "decision", "refactor", "tool_output"],
        priority_concepts=["gotcha", "pattern", "trade-off", "how-it-works"],
        auto_tags=["code"],
        endless_enabled=True,
        compression_ratio=4.0,
        fts_weight=0.4,
        vector_weight=0.4,
        recency_weight=0.2,
        disclosure_mode="standard",
        context_description="Development session - prioritizing code decisions and patterns",
    ),
    "email": ModeConfig(
        name="Email Investigation Mode",
        description="For analyzing communications and email threads",
        context_total=20,
        context_full=8,
        priority_types=["note", "discovery", "decision", "interaction"],
        priority_concepts=["problem-solution", "why-it-exists"],
        auto_tags=["investigation", "email"],
        endless_enabled=True,
        compression_ratio=2.0,
        fts_weight=0.5,
        vector_weight=0.3,
        recency_weight=0.2,
        disclosure_mode="standard",
        context_description="Investigation session - capturing communications and findings",
    ),
    "chill": ModeConfig(
        name="Chill Mode",
        description="Casual conversation with minimal memory overhead",
        context_total=5,
        context_full=2,
        priority_types=["note", "preference"],
        priority_concepts=[],
        auto_tags=["casual"],
        endless_enabled=False,
        compression_ratio=8.0,
        fts_weight=0.3,
        vector_weight=0.5,
        recency_weight=0.2,
        disclosure_mode="compact",
        context_description="Casual session - lightweight memory",
    ),
    "research": ModeConfig(
        name="Research Mode",
        description="For deep research and exploration tasks",
        context_total=25,
        context_full=10,
        priority_types=["discovery", "note", "decision", "summary"],
        priority_concepts=["how-it-works", "why-it-exists", "architecture"],
        auto_tags=["research"],
        endless_enabled=True,
        compression_ratio=3.0,
        fts_weight=0.3,
        vector_weight=0.5,
        recency_weight=0.2,
        disclosure_mode="full",
        context_description="Research session - comprehensive context for exploration",
    ),
    "debug": ModeConfig(
        name="Debug Mode",
        description="For debugging and troubleshooting",
        context_total=15,
        context_full=6,
        priority_types=["bugfix", "tool_output", "discovery", "note"],
        priority_concepts=["gotcha", "problem-solution", "workaround"],
        auto_tags=["debug"],
        endless_enabled=True,
        compression_ratio=2.0,
        fts_weight=0.4,
        vector_weight=0.4,
        recency_weight=0.2,
        disclosure_mode="standard",
        context_description="Debug session - focusing on errors and fixes",
    ),
}


def get_current_mode() -> Optional[str]:
    """Get the current mode from environment.

    Returns:
        Mode name or None
    """
    return os.environ.get("AI_MEM_MODE", "").lower() or None


def get_mode_config(mode_name: str) -> Optional[ModeConfig]:
    """Get configuration for a mode.

    Args:
        mode_name: Name of the mode

    Returns:
        ModeConfig or None if not found
    """
    return MODES.get(mode_name.lower())


def list_modes() -> List[Dict[str, Any]]:
    """List all available modes.

    Returns:
        List of mode info dictionaries
    """
    return [
        {
            "name": mode.name,
            "key": key,
            "description": mode.description,
            "context_total": mode.context_total,
            "endless_enabled": mode.endless_enabled,
        }
        for key, mode in MODES.items()
    ]


def apply_mode(config: AppConfig, mode_name: str) -> AppConfig:
    """Apply a mode's settings to a configuration.

    Args:
        config: Base configuration
        mode_name: Mode to apply

    Returns:
        Updated configuration
    """
    mode = get_mode_config(mode_name)
    if not mode:
        logger.warning(f"Unknown mode: {mode_name}")
        return config

    logger.info(f"Applying mode: {mode.name}")

    # Update context settings
    config.context.total_observation_count = mode.context_total
    config.context.full_observation_count = mode.context_full
    config.context.disclosure_mode = mode.disclosure_mode

    # Update endless mode settings
    config.endless.enabled = mode.endless_enabled
    config.endless.compression_ratio = mode.compression_ratio

    # Update search settings
    config.search.fts_weight = mode.fts_weight
    config.search.vector_weight = mode.vector_weight
    config.search.recency_weight = mode.recency_weight

    return config


def get_mode_context_header(mode_name: Optional[str] = None) -> str:
    """Get a context header for the current mode.

    Args:
        mode_name: Mode name (uses current mode if None)

    Returns:
        Context header string
    """
    if mode_name is None:
        mode_name = get_current_mode()

    if not mode_name:
        return ""

    mode = get_mode_config(mode_name)
    if not mode:
        return ""

    return f"[Mode: {mode.name}] {mode.context_description}"


class ModeManager:
    """Manager for work modes.

    Usage:
        manager = ModeManager()
        if manager.current_mode:
            config = manager.apply_to_config(config)
    """

    def __init__(self):
        """Initialize mode manager."""
        self._current_mode = get_current_mode()

    @property
    def current_mode(self) -> Optional[str]:
        """Get the current mode."""
        return self._current_mode

    @property
    def mode_config(self) -> Optional[ModeConfig]:
        """Get the current mode configuration."""
        if self._current_mode:
            return get_mode_config(self._current_mode)
        return None

    def set_mode(self, mode_name: str) -> bool:
        """Set the current mode.

        Args:
            mode_name: Mode to set

        Returns:
            True if mode was set successfully
        """
        if mode_name not in MODES:
            logger.warning(f"Unknown mode: {mode_name}")
            return False

        self._current_mode = mode_name
        os.environ["AI_MEM_MODE"] = mode_name
        logger.info(f"Mode set to: {mode_name}")
        return True

    def clear_mode(self) -> None:
        """Clear the current mode."""
        self._current_mode = None
        os.environ.pop("AI_MEM_MODE", None)
        logger.info("Mode cleared")

    def apply_to_config(self, config: AppConfig) -> AppConfig:
        """Apply the current mode to a configuration.

        Args:
            config: Configuration to modify

        Returns:
            Modified configuration
        """
        if self._current_mode:
            return apply_mode(config, self._current_mode)
        return config

    def get_auto_tags(self) -> List[str]:
        """Get auto-tags for the current mode.

        Returns:
            List of tags to auto-apply
        """
        if self.mode_config:
            return self.mode_config.auto_tags
        return []

    def get_priority_types(self) -> List[str]:
        """Get priority observation types for the current mode.

        Returns:
            List of types to prioritize
        """
        if self.mode_config:
            return self.mode_config.priority_types
        return []

    def get_priority_concepts(self) -> List[str]:
        """Get priority concepts for the current mode.

        Returns:
            List of concepts to prioritize
        """
        if self.mode_config:
            return self.mode_config.priority_concepts
        return []


# Singleton instance
_mode_manager: Optional[ModeManager] = None


def get_mode_manager() -> ModeManager:
    """Get or create the mode manager singleton.

    Returns:
        ModeManager instance
    """
    global _mode_manager

    if _mode_manager is None:
        _mode_manager = ModeManager()

    return _mode_manager

"""Citations System - URL-based observation references.

This module provides URL-based citations for observations, allowing:
- Direct linking to observations via HTTP
- Formatted citation output in context
- Short ID references for compact display

Configuration:
    AI_MEM_CITATIONS_BASE_URL=http://localhost:8000
    AI_MEM_CITATIONS_ENABLED=true
    AI_MEM_CITATIONS_FORMAT=full|compact|id_only
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from .logging_config import get_logger

logger = get_logger("citations")

# Default settings
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_ENABLED = True
DEFAULT_FORMAT = "compact"  # compact, full, id_only

# Configuration via environment
CITATIONS_BASE_URL = os.environ.get("AI_MEM_CITATIONS_BASE_URL", DEFAULT_BASE_URL)
CITATIONS_ENABLED = os.environ.get("AI_MEM_CITATIONS_ENABLED", "true").lower() in ("true", "1", "yes")
CITATIONS_FORMAT = os.environ.get("AI_MEM_CITATIONS_FORMAT", DEFAULT_FORMAT)


@dataclass
class Citation:
    """A citation reference to an observation."""

    observation_id: str
    short_id: str  # First 8 chars
    url: str
    api_url: str
    html_url: str


def get_base_url() -> str:
    """Get the base URL for citations.

    Returns:
        Base URL (e.g., http://localhost:8000)
    """
    return CITATIONS_BASE_URL


def is_enabled() -> bool:
    """Check if citations are enabled.

    Returns:
        True if citations are enabled
    """
    return CITATIONS_ENABLED


def get_format() -> str:
    """Get the citation format.

    Returns:
        Format string (compact, full, id_only)
    """
    return CITATIONS_FORMAT


def create_citation(observation_id: str) -> Citation:
    """Create a citation for an observation.

    Args:
        observation_id: Full observation UUID

    Returns:
        Citation object with URLs
    """
    base = get_base_url().rstrip("/")
    short_id = observation_id[:8]

    return Citation(
        observation_id=observation_id,
        short_id=short_id,
        url=f"{base}/obs/{short_id}",
        api_url=f"{base}/api/observation/{observation_id}",
        html_url=f"{base}/view/{observation_id}",
    )


def format_citation(
    observation_id: str,
    format_type: Optional[str] = None,
    include_link: bool = True,
) -> str:
    """Format a citation for display.

    Args:
        observation_id: Full observation UUID
        format_type: Override format (compact, full, id_only)
        include_link: Whether to include the URL

    Returns:
        Formatted citation string
    """
    if not is_enabled():
        return observation_id[:8]

    citation = create_citation(observation_id)
    fmt = format_type or get_format()

    if fmt == "id_only":
        return citation.short_id

    if fmt == "full":
        if include_link:
            return f"[{citation.short_id}]({citation.html_url})"
        return f"{citation.short_id} ({citation.html_url})"

    # compact (default)
    if include_link:
        return f"[{citation.short_id}]({citation.url})"
    return citation.short_id


def format_observation_with_citation(
    obs: Dict[str, Any],
    include_url: bool = True,
) -> str:
    """Format an observation summary with citation.

    Args:
        obs: Observation dictionary
        include_url: Whether to include URL

    Returns:
        Formatted string with citation
    """
    obs_id = obs.get("id", "")
    summary = obs.get("summary") or obs.get("content", "")[:100]
    obs_type = obs.get("type", "note")

    citation = format_citation(obs_id, include_link=include_url)

    return f"[{citation}] ({obs_type}) {summary}"


def batch_create_citations(observation_ids: List[str]) -> Dict[str, Citation]:
    """Create citations for multiple observations.

    Args:
        observation_ids: List of observation UUIDs

    Returns:
        Dict mapping observation_id to Citation
    """
    return {obs_id: create_citation(obs_id) for obs_id in observation_ids}


class CitationFormatter:
    """Formatter for citations with configurable settings.

    Usage:
        formatter = CitationFormatter(base_url="http://myserver:8000")
        citation = formatter.format(observation_id)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        format_type: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        """Initialize formatter.

        Args:
            base_url: Base URL for links
            format_type: Citation format
            enabled: Whether citations are enabled
        """
        self._base_url = base_url or get_base_url()
        self._format = format_type or get_format()
        self._enabled = enabled if enabled is not None else is_enabled()

    @property
    def base_url(self) -> str:
        """Get base URL."""
        return self._base_url

    @property
    def enabled(self) -> bool:
        """Check if enabled."""
        return self._enabled

    def create(self, observation_id: str) -> Citation:
        """Create a citation.

        Args:
            observation_id: Observation UUID

        Returns:
            Citation object
        """
        base = self._base_url.rstrip("/")
        short_id = observation_id[:8]

        return Citation(
            observation_id=observation_id,
            short_id=short_id,
            url=f"{base}/obs/{short_id}",
            api_url=f"{base}/api/observation/{observation_id}",
            html_url=f"{base}/view/{observation_id}",
        )

    def format(
        self,
        observation_id: str,
        include_link: bool = True,
    ) -> str:
        """Format a citation.

        Args:
            observation_id: Observation UUID
            include_link: Include URL link

        Returns:
            Formatted citation string
        """
        if not self._enabled:
            return observation_id[:8]

        citation = self.create(observation_id)

        if self._format == "id_only":
            return citation.short_id

        if self._format == "full":
            if include_link:
                return f"[{citation.short_id}]({citation.html_url})"
            return f"{citation.short_id} ({citation.html_url})"

        # compact
        if include_link:
            return f"[{citation.short_id}]({citation.url})"
        return citation.short_id


# Global formatter instance
_formatter: Optional[CitationFormatter] = None


def get_formatter() -> CitationFormatter:
    """Get or create the global citation formatter.

    Returns:
        CitationFormatter instance
    """
    global _formatter

    if _formatter is None:
        _formatter = CitationFormatter()

    return _formatter


def reset_formatter() -> None:
    """Reset the global formatter (for testing)."""
    global _formatter
    _formatter = None

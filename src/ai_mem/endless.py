"""Endless Mode - Real-time compression for extended AI coding sessions.

Endless Mode transforms O(N²) context growth into O(N) linear scaling by:
- Compressing tool outputs in real-time (~500 tokens per observation)
- Maintaining full transcripts in archive memory (disk)
- Achieving ~20x more tool uses before context exhaustion

Two-tier memory architecture:
- Working Memory (Context Window): Compressed observations only
- Archive Memory (Disk): Full tool outputs preserved for perfect recall
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .compression import CompressionResult, CompressionService, get_compression_service
from .config import AppConfig, EndlessModeConfig, load_config, resolve_storage_paths
from .logging_config import get_logger
from .providers.base import ChatProvider

logger = get_logger("endless")


@dataclass
class ArchiveEntry:
    """An entry in the archive memory (full transcript on disk)."""

    observation_id: str
    session_id: str
    project: str
    tool_name: str
    tool_input: str
    tool_output: str  # Full, uncompressed output
    compressed_output: str  # Compressed version used in working memory
    created_at: float
    compression_ratio: float
    original_tokens: int
    compressed_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EndlessModeStats:
    """Statistics for Endless Mode performance."""

    total_observations: int = 0
    compressed_observations: int = 0
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0
    tokens_saved: int = 0
    average_compression_ratio: float = 0.0
    archive_entries: int = 0
    archive_size_bytes: int = 0
    enabled_since: Optional[float] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_observations": self.total_observations,
            "compressed_observations": self.compressed_observations,
            "total_original_tokens": self.total_original_tokens,
            "total_compressed_tokens": self.total_compressed_tokens,
            "tokens_saved": self.tokens_saved,
            "average_compression_ratio": round(self.average_compression_ratio, 2),
            "archive_entries": self.archive_entries,
            "archive_size_bytes": self.archive_size_bytes,
            "enabled_since": self.enabled_since,
            "session_id": self.session_id,
        }


class ArchiveManager:
    """Manages the archive memory (full transcripts on disk).

    Archive files are stored as NDJSON (newline-delimited JSON) for efficient
    appending and reading. Each session has its own archive file.
    """

    def __init__(self, archive_dir: str):
        """Initialize archive manager.

        Args:
            archive_dir: Directory to store archive files
        """
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self._current_file: Optional[Path] = None
        self._current_session: Optional[str] = None

    def _get_archive_path(self, session_id: str, project: str) -> Path:
        """Get the archive file path for a session."""
        # Sanitize project name for filesystem
        safe_project = project.replace("/", "_").replace("\\", "_")[-50:]
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"{date_str}_{safe_project}_{session_id[:8]}.ndjson"
        return self.archive_dir / filename

    def append(self, entry: ArchiveEntry) -> None:
        """Append an entry to the archive.

        Args:
            entry: Archive entry to store
        """
        archive_path = self._get_archive_path(entry.session_id, entry.project)

        entry_data = {
            "observation_id": entry.observation_id,
            "session_id": entry.session_id,
            "project": entry.project,
            "tool_name": entry.tool_name,
            "tool_input": entry.tool_input,
            "tool_output": entry.tool_output,
            "compressed_output": entry.compressed_output,
            "created_at": entry.created_at,
            "compression_ratio": entry.compression_ratio,
            "original_tokens": entry.original_tokens,
            "compressed_tokens": entry.compressed_tokens,
            "metadata": entry.metadata,
        }

        try:
            with open(archive_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry_data) + "\n")
            logger.debug(f"Archived entry {entry.observation_id} to {archive_path}")
        except Exception as e:
            logger.warning(f"Failed to archive entry: {e}")

    def get_entry(self, observation_id: str, session_id: str, project: str) -> Optional[ArchiveEntry]:
        """Retrieve a specific entry from the archive.

        Args:
            observation_id: ID of the observation
            session_id: Session ID
            project: Project path

        Returns:
            ArchiveEntry if found, None otherwise
        """
        archive_path = self._get_archive_path(session_id, project)

        if not archive_path.exists():
            return None

        try:
            with open(archive_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("observation_id") == observation_id:
                        return ArchiveEntry(**data)
        except Exception as e:
            logger.warning(f"Failed to read archive entry: {e}")

        return None

    def get_full_output(self, observation_id: str, session_id: str, project: str) -> Optional[str]:
        """Get the full (uncompressed) tool output for an observation.

        Args:
            observation_id: ID of the observation
            session_id: Session ID
            project: Project path

        Returns:
            Full tool output if found, None otherwise
        """
        entry = self.get_entry(observation_id, session_id, project)
        return entry.tool_output if entry else None

    def list_entries(
        self,
        session_id: Optional[str] = None,
        project: Optional[str] = None,
        limit: int = 100,
    ) -> List[ArchiveEntry]:
        """List archive entries.

        Args:
            session_id: Filter by session ID
            project: Filter by project
            limit: Maximum entries to return

        Returns:
            List of archive entries
        """
        entries: List[ArchiveEntry] = []

        for archive_file in sorted(self.archive_dir.glob("*.ndjson"), reverse=True):
            if len(entries) >= limit:
                break

            try:
                with open(archive_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if len(entries) >= limit:
                            break
                        if not line.strip():
                            continue

                        data = json.loads(line)

                        # Apply filters
                        if session_id and data.get("session_id") != session_id:
                            continue
                        if project and data.get("project") != project:
                            continue

                        entries.append(ArchiveEntry(**data))
            except Exception as e:
                logger.warning(f"Failed to read archive file {archive_file}: {e}")

        return entries

    def get_stats(self) -> Dict[str, Any]:
        """Get archive statistics.

        Returns:
            Dictionary with archive stats
        """
        total_entries = 0
        total_size = 0

        for archive_file in self.archive_dir.glob("*.ndjson"):
            try:
                total_size += archive_file.stat().st_size
                with open(archive_file, "r", encoding="utf-8") as f:
                    total_entries += sum(1 for line in f if line.strip())
            except Exception:
                pass

        return {
            "entries": total_entries,
            "size_bytes": total_size,
            "size_mb": round(total_size / (1024 * 1024), 2),
            "archive_dir": str(self.archive_dir),
            "file_count": len(list(self.archive_dir.glob("*.ndjson"))),
        }


class EndlessModeManager:
    """Manager for Endless Mode compression and archiving.

    Handles real-time compression of tool outputs and maintains archive memory.
    """

    def __init__(
        self,
        config: Optional[EndlessModeConfig] = None,
        compression_service: Optional[CompressionService] = None,
        archive_dir: Optional[str] = None,
    ):
        """Initialize Endless Mode manager.

        Args:
            config: Endless mode configuration
            compression_service: Service for compression (uses default if None)
            archive_dir: Directory for archive (uses config if None)
        """
        if config is None:
            app_config = load_config()
            config = app_config.endless
            if archive_dir is None:
                storage = resolve_storage_paths(app_config)
                archive_dir = os.path.join(storage.data_dir, config.archive_dir)

        self.config = config
        self.compression_service = compression_service or get_compression_service()
        self.archive_dir = archive_dir or os.path.expanduser("~/.ai-mem/archive")
        self.archive = ArchiveManager(self.archive_dir) if config.enable_archive else None
        self.stats = EndlessModeStats()
        self._session_id: Optional[str] = None

    @property
    def enabled(self) -> bool:
        """Check if Endless Mode is enabled."""
        return self.config.enabled

    def enable(self, session_id: Optional[str] = None) -> None:
        """Enable Endless Mode.

        Args:
            session_id: Current session ID
        """
        self.config.enabled = True
        self.stats.enabled_since = time.time()
        self.stats.session_id = session_id
        self._session_id = session_id
        logger.info(f"Endless Mode enabled for session {session_id}")

    def disable(self) -> None:
        """Disable Endless Mode."""
        self.config.enabled = False
        logger.info("Endless Mode disabled")

    def should_compress(self, content: str) -> bool:
        """Check if content should be compressed.

        Args:
            content: Content to check

        Returns:
            True if content should be compressed
        """
        if not self.enabled:
            return False
        return len(content) >= self.config.min_output_for_compression

    async def compress_tool_output(
        self,
        tool_name: str,
        tool_input: Any,
        tool_output: str,
        observation_id: str,
        session_id: str,
        project: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, CompressionResult]:
        """Compress a tool output for Endless Mode.

        This is the core method that:
        1. Archives the full output (if archiving enabled)
        2. Compresses the output for working memory
        3. Updates statistics

        Args:
            tool_name: Name of the tool
            tool_input: Tool input (any serializable type)
            tool_output: Full tool output to compress
            observation_id: ID for the observation
            session_id: Current session ID
            project: Project path
            metadata: Additional metadata

        Returns:
            Tuple of (compressed_output, CompressionResult)
        """
        logger.debug(f"Compressing tool output for {tool_name}, length={len(tool_output)}")
        start_time = time.perf_counter()

        # Determine context type based on tool name
        context_type = "tool_output"
        if tool_name.lower() in {"read", "cat", "head", "tail"}:
            context_type = "code_context"

        # Compress the output
        use_ai = self.config.compression_method == "ai"
        result = await self.compression_service.compress(
            text=tool_output,
            context_type=context_type,
            target_ratio=self.config.compression_ratio,
            use_ai=use_ai,
        )

        # Archive the full output
        if self.archive and self.config.enable_archive:
            input_str = json.dumps(tool_input) if not isinstance(tool_input, str) else tool_input
            entry = ArchiveEntry(
                observation_id=observation_id,
                session_id=session_id,
                project=project,
                tool_name=tool_name,
                tool_input=input_str[:10000],  # Limit input size
                tool_output=tool_output,
                compressed_output=result.compressed_text,
                created_at=time.time(),
                compression_ratio=result.compression_ratio,
                original_tokens=result.original_tokens,
                compressed_tokens=result.compressed_tokens,
                metadata=metadata or {},
            )
            self.archive.append(entry)
            self.stats.archive_entries += 1

        # Update statistics
        self.stats.total_observations += 1
        self.stats.compressed_observations += 1
        self.stats.total_original_tokens += result.original_tokens
        self.stats.total_compressed_tokens += result.compressed_tokens
        self.stats.tokens_saved += result.original_tokens - result.compressed_tokens

        if self.stats.compressed_observations > 0:
            self.stats.average_compression_ratio = (
                self.stats.total_original_tokens / self.stats.total_compressed_tokens
            )

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Compressed {tool_name} output: {result.original_tokens} -> "
            f"{result.compressed_tokens} tokens ({result.compression_ratio:.1f}x) "
            f"in {duration_ms:.0f}ms"
        )

        return result.compressed_text, result

    def get_full_output(
        self,
        observation_id: str,
        session_id: str,
        project: str,
    ) -> Optional[str]:
        """Retrieve the full (uncompressed) output from archive.

        Args:
            observation_id: ID of the observation
            session_id: Session ID
            project: Project path

        Returns:
            Full output if found, None otherwise
        """
        if not self.archive:
            return None
        return self.archive.get_full_output(observation_id, session_id, project)

    def get_stats(self) -> EndlessModeStats:
        """Get current Endless Mode statistics."""
        if self.archive:
            archive_stats = self.archive.get_stats()
            self.stats.archive_size_bytes = archive_stats.get("size_bytes", 0)
            self.stats.archive_entries = archive_stats.get("entries", 0)
        return self.stats

    def format_stats_for_context(self) -> str:
        """Format statistics for inclusion in context.

        Returns:
            Formatted string for context injection
        """
        if not self.config.show_compression_stats:
            return ""

        stats = self.get_stats()

        lines = [
            "📊 Endless Mode Stats:",
            f"  Observations: {stats.total_observations}",
            f"  Tokens saved: {stats.tokens_saved:,}",
            f"  Avg compression: {stats.average_compression_ratio:.1f}x",
        ]

        if stats.archive_entries > 0:
            lines.append(f"  Archive: {stats.archive_entries} entries")

        return "\n".join(lines)


# Singleton instance
_endless_manager: Optional[EndlessModeManager] = None


def get_endless_manager(
    config: Optional[EndlessModeConfig] = None,
    force_new: bool = False,
) -> EndlessModeManager:
    """Get or create the Endless Mode manager.

    Args:
        config: Optional configuration override
        force_new: If True, create a new instance

    Returns:
        EndlessModeManager instance
    """
    global _endless_manager

    if force_new or _endless_manager is None:
        _endless_manager = EndlessModeManager(config=config)

    return _endless_manager


async def compress_if_endless_mode(
    content: str,
    tool_name: str,
    tool_input: Any,
    observation_id: str,
    session_id: str,
    project: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Compress content if Endless Mode is enabled.

    Convenience function for use in hooks and memory manager.

    Args:
        content: Content to potentially compress
        tool_name: Name of the tool
        tool_input: Tool input
        observation_id: Observation ID
        session_id: Session ID
        project: Project path
        metadata: Additional metadata

    Returns:
        Tuple of (possibly compressed content, compression metadata or None)
    """
    manager = get_endless_manager()

    if not manager.should_compress(content):
        return content, None

    compressed, result = await manager.compress_tool_output(
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=content,
        observation_id=observation_id,
        session_id=session_id,
        project=project,
        metadata=metadata,
    )

    compression_metadata = {
        "endless_mode": True,
        "original_tokens": result.original_tokens,
        "compressed_tokens": result.compressed_tokens,
        "compression_ratio": result.compression_ratio,
        "compression_method": result.method,
        "has_archive": manager.archive is not None,
    }

    return compressed, compression_metadata

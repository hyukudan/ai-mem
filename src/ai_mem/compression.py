"""AI-Powered Compression Service for ai-mem.

This module provides semantic compression using LLM providers,
enabling 4:1 or higher compression ratios while preserving meaning.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .providers.base import ChatProvider, ChatMessage, NoOpChatProvider


# Compression prompts optimized for different use cases
COMPRESSION_PROMPTS = {
    "default": """Compress the following text to approximately {target_ratio}x shorter while preserving all key information, entities, and technical details. Output ONLY the compressed text, no explanations.

Text to compress:
{text}""",

    "code_context": """Compress this code-related context to ~{target_ratio}x shorter. Keep: file paths, function names, error messages, decisions made. Remove: verbose explanations, filler words.

Text:
{text}""",

    "tool_output": """Compress this tool output to essential information only (~{target_ratio}x shorter). Keep: results, errors, key data. Remove: formatting, timestamps, metadata.

Output:
{text}""",

    "conversation": """Compress this conversation context to key points (~{target_ratio}x shorter). Keep: decisions, questions, action items. Remove: greetings, acknowledgments, filler.

Conversation:
{text}""",
}


@dataclass
class CompressionResult:
    """Result of text compression."""
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    method: str  # "ai" or "heuristic"


class CompressionService:
    """Service for compressing text using AI or heuristics."""

    def __init__(
        self,
        provider: Optional[ChatProvider] = None,
        default_ratio: float = 4.0,
        min_length_for_ai: int = 200,
    ):
        """Initialize compression service.

        Args:
            provider: ChatProvider for AI compression (None = heuristic only)
            default_ratio: Target compression ratio (e.g., 4.0 = 4x smaller)
            min_length_for_ai: Minimum text length to use AI compression
        """
        self.provider = provider or NoOpChatProvider()
        self.default_ratio = default_ratio
        self.min_length_for_ai = min_length_for_ai

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (chars / 4)."""
        return max(1, len(text) // 4)

    def _heuristic_compress(
        self,
        text: str,
        target_ratio: float = 4.0,
    ) -> str:
        """Compress text using heuristics (no LLM).

        Uses the same logic as context.compress_text but more aggressive.
        """
        if not text:
            return text

        # Import here to avoid circular dependency
        from .context import compress_text

        # Calculate target length
        target_length = int(len(text) / target_ratio)

        # Apply aggressive compression
        compressed = compress_text(
            text,
            compression_level=0.9,  # High compression
            max_length=target_length,
        )

        return compressed

    async def compress(
        self,
        text: str,
        context_type: str = "default",
        target_ratio: Optional[float] = None,
        use_ai: bool = True,
    ) -> CompressionResult:
        """Compress text using AI or heuristics.

        Args:
            text: Text to compress
            context_type: Type of content ("default", "code_context", "tool_output", "conversation")
            target_ratio: Target compression ratio (None = use default)
            use_ai: Whether to use AI compression (falls back to heuristic if provider unavailable)

        Returns:
            CompressionResult with original and compressed text
        """
        if not text:
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                original_tokens=0,
                compressed_tokens=0,
                compression_ratio=1.0,
                method="none",
            )

        original_tokens = self._estimate_tokens(text)
        ratio = target_ratio or self.default_ratio

        # Use heuristic for short texts or if AI not available
        if not use_ai or len(text) < self.min_length_for_ai or isinstance(self.provider, NoOpChatProvider):
            compressed = self._heuristic_compress(text, ratio)
            compressed_tokens = self._estimate_tokens(compressed)
            return CompressionResult(
                original_text=text,
                compressed_text=compressed,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                compression_ratio=original_tokens / max(1, compressed_tokens),
                method="heuristic",
            )

        # Use AI compression
        try:
            prompt_template = COMPRESSION_PROMPTS.get(context_type, COMPRESSION_PROMPTS["default"])
            prompt = prompt_template.format(text=text, target_ratio=ratio)

            messages = [ChatMessage(role="user", content=prompt)]
            compressed = await self.provider.chat(messages, temperature=0.1)
            compressed = compressed.strip()

            # Fallback to heuristic if AI returned nothing or longer text
            if not compressed or len(compressed) >= len(text):
                compressed = self._heuristic_compress(text, ratio)
                method = "heuristic_fallback"
            else:
                method = "ai"

            compressed_tokens = self._estimate_tokens(compressed)
            return CompressionResult(
                original_text=text,
                compressed_text=compressed,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                compression_ratio=original_tokens / max(1, compressed_tokens),
                method=method,
            )

        except Exception as e:
            # Fallback to heuristic on error
            compressed = self._heuristic_compress(text, ratio)
            compressed_tokens = self._estimate_tokens(compressed)
            return CompressionResult(
                original_text=text,
                compressed_text=compressed,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                compression_ratio=original_tokens / max(1, compressed_tokens),
                method="heuristic_error",
            )

    async def compress_observations(
        self,
        observations: List[Dict[str, Any]],
        field: str = "content",
        context_type: str = "default",
        target_ratio: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Compress a list of observations.

        Args:
            observations: List of observation dicts
            field: Field to compress
            context_type: Type of content
            target_ratio: Target compression ratio

        Returns:
            Tuple of (compressed observations, stats dict)
        """
        compressed_obs = []
        total_original = 0
        total_compressed = 0
        methods_used = {"ai": 0, "heuristic": 0, "none": 0}

        for obs in observations:
            obs_copy = dict(obs)
            text = obs_copy.get(field) or ""

            if text:
                # Determine context type from observation type
                obs_type = obs.get("type", "")
                if obs_type == "tool_output":
                    ctx_type = "tool_output"
                elif obs_type in ("interaction", "note"):
                    ctx_type = "conversation"
                else:
                    ctx_type = context_type

                result = await self.compress(
                    text,
                    context_type=ctx_type,
                    target_ratio=target_ratio,
                )

                obs_copy[field] = result.compressed_text
                obs_copy["_compression"] = {
                    "original_tokens": result.original_tokens,
                    "compressed_tokens": result.compressed_tokens,
                    "ratio": result.compression_ratio,
                    "method": result.method,
                }

                total_original += result.original_tokens
                total_compressed += result.compressed_tokens

                if "ai" in result.method:
                    methods_used["ai"] += 1
                elif "heuristic" in result.method:
                    methods_used["heuristic"] += 1
                else:
                    methods_used["none"] += 1

            compressed_obs.append(obs_copy)

        stats = {
            "total_observations": len(observations),
            "total_original_tokens": total_original,
            "total_compressed_tokens": total_compressed,
            "overall_ratio": total_original / max(1, total_compressed),
            "tokens_saved": total_original - total_compressed,
            "methods_used": methods_used,
        }

        return compressed_obs, stats


# Singleton instance for convenience
_default_service: Optional[CompressionService] = None


def get_compression_service(provider: Optional[ChatProvider] = None) -> CompressionService:
    """Get or create the default compression service.

    Args:
        provider: Optional ChatProvider to use

    Returns:
        CompressionService instance
    """
    global _default_service

    if provider is not None:
        return CompressionService(provider=provider)

    if _default_service is None:
        _default_service = CompressionService()

    return _default_service

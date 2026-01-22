import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set

from .config import ContextConfig
from .memory import MemoryManager
from .injection import (
    calculate_context_budget,
    determine_optimal_disclosure_mode,
    get_host_config,
)

CHARS_PER_TOKEN_ESTIMATE = 4

# Token budgets for different disclosure modes
DISCLOSURE_TOKEN_TARGETS = {
    "compact": 100,    # Layer 1: ~50-100 tokens (ultra-compact)
    "standard": 500,   # Layer 1 + partial Layer 2
    "full": 2000,      # All layers
    "auto": None,      # Determined by token budget
}

# Default context window sizes for common models
MODEL_CONTEXT_WINDOWS = {
    # Anthropic
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-opus-4": 200000,
    "claude-sonnet-4": 200000,
    # OpenAI
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-4o": 128000,
    "gpt-3.5-turbo": 16384,
    # Google
    "gemini-pro": 32000,
    "gemini-1.5-pro": 1000000,
    "gemini-1.5-flash": 1000000,
    # Default
    "default": 16000,
}

# Warning thresholds (percentage of context window)
TOKEN_WARNING_THRESHOLDS = {
    "info": 0.15,      # >15% - informational
    "warning": 0.30,   # >30% - warning
    "critical": 0.50,  # >50% - critical warning
}


@dataclass
class TokenBudgetWarning:
    """Warning about token budget usage."""
    level: str  # "info", "warning", "critical"
    message: str
    tokens_used: int
    tokens_budget: int
    percentage: float
    recommendations: List[str]


def get_model_context_window(model: Optional[str] = None) -> int:
    """Get the context window size for a model.

    Args:
        model: Model identifier (e.g., "claude-3-opus", "gpt-4")

    Returns:
        Context window size in tokens
    """
    if not model:
        return MODEL_CONTEXT_WINDOWS["default"]

    model_lower = model.lower()

    # Exact match
    if model_lower in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model_lower]

    # Partial match
    for key, value in MODEL_CONTEXT_WINDOWS.items():
        if key in model_lower or model_lower in key:
            return value

    return MODEL_CONTEXT_WINDOWS["default"]


def check_token_budget(
    tokens_used: int,
    model: Optional[str] = None,
    max_budget: Optional[int] = None,
    context_percentage: float = 0.25,  # Recommend using max 25% for context
) -> Optional[TokenBudgetWarning]:
    """Check if token usage exceeds recommended thresholds.

    Args:
        tokens_used: Number of tokens used for context
        model: Model identifier for context window lookup
        max_budget: Override max budget (ignores model lookup)
        context_percentage: Target percentage of context window for memory

    Returns:
        TokenBudgetWarning if threshold exceeded, None otherwise
    """
    if max_budget:
        budget = max_budget
    else:
        context_window = get_model_context_window(model)
        budget = int(context_window * context_percentage)

    if budget <= 0:
        return None

    percentage = tokens_used / budget

    # Determine warning level
    if percentage >= TOKEN_WARNING_THRESHOLDS["critical"]:
        level = "critical"
    elif percentage >= TOKEN_WARNING_THRESHOLDS["warning"]:
        level = "warning"
    elif percentage >= TOKEN_WARNING_THRESHOLDS["info"]:
        level = "info"
    else:
        return None

    # Generate recommendations based on level
    recommendations = []

    if level == "critical":
        recommendations.extend([
            "Switch to 'compact' disclosure mode for ~10x token savings",
            "Reduce --total and --full counts",
            "Use more specific search queries to narrow results",
            "Enable compression with --compression-level 0.7",
        ])
    elif level == "warning":
        recommendations.extend([
            "Consider using 'compact' or 'standard' disclosure mode",
            "Reduce --total count to limit observations",
            "Enable deduplication to merge similar observations",
        ])
    else:  # info
        recommendations.extend([
            "Current usage is within acceptable limits",
            "Consider monitoring if context grows further",
        ])

    # Format message
    if level == "critical":
        message = f"CRITICAL: Context using {percentage:.0%} of budget ({tokens_used}/{budget} tokens)"
    elif level == "warning":
        message = f"WARNING: Context using {percentage:.0%} of budget ({tokens_used}/{budget} tokens)"
    else:
        message = f"INFO: Context using {percentage:.0%} of budget ({tokens_used}/{budget} tokens)"

    return TokenBudgetWarning(
        level=level,
        message=message,
        tokens_used=tokens_used,
        tokens_budget=budget,
        percentage=percentage * 100,
        recommendations=recommendations,
    )


def format_token_warning(warning: TokenBudgetWarning) -> str:
    """Format a token budget warning for display.

    Args:
        warning: TokenBudgetWarning object

    Returns:
        Formatted warning string
    """
    lines = [warning.message]

    if warning.recommendations:
        lines.append("Recommendations:")
        for rec in warning.recommendations[:3]:  # Top 3 recommendations
            lines.append(f"  - {rec}")

    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE))


def _compute_text_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts (word-level).

    This is a fast, LLM-agnostic approximation for deduplication.
    For more accurate semantic similarity, use vector embeddings.
    """
    if not text1 or not text2:
        return 0.0

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


@dataclass
class DeduplicationResult:
    """Result of deduplication operation."""
    observations: List[Dict[str, Any]]
    removed_count: int
    similar_groups: Dict[str, List[str]]  # obs_id -> list of similar obs_ids


def deduplicate_observations(
    observations: List[Dict[str, Any]],
    threshold: float = 0.85,
    key_field: str = "summary",
    track_similar: bool = True,
) -> Tuple[List[Dict[str, Any]], int]:
    """Remove semantically similar observations.

    Uses Jaccard word-level similarity for fast deduplication.
    Optionally tracks similar observations for "[+N similar]" display.

    Args:
        observations: List of observation dicts
        threshold: Similarity threshold (0.0-1.0) above which to consider duplicates
        key_field: Field to use for similarity comparison
        track_similar: If True, add "_similar_count" to retained observations

    Returns:
        Tuple of (deduplicated list, count of removed duplicates)
    """
    if not observations or threshold <= 0:
        return observations, 0

    deduplicated: List[Dict[str, Any]] = []
    seen_texts: List[str] = []
    seen_indices: List[int] = []  # Index in deduplicated list
    removed_count = 0

    for obs in observations:
        text = obs.get(key_field) or obs.get("content") or ""
        if not text:
            deduplicated.append(obs)
            seen_texts.append("")
            seen_indices.append(len(deduplicated) - 1)
            continue

        # Check similarity with already-seen observations
        is_duplicate = False
        best_match_idx = -1
        best_similarity = 0.0

        for idx, seen_text in enumerate(seen_texts):
            if not seen_text:
                continue
            similarity = _compute_text_similarity(text, seen_text)
            if similarity >= threshold and similarity > best_similarity:
                is_duplicate = True
                best_match_idx = seen_indices[idx]
                best_similarity = similarity

        if is_duplicate and best_match_idx >= 0:
            removed_count += 1
            if track_similar:
                # Increment similar count on the retained observation
                if "_similar_count" not in deduplicated[best_match_idx]:
                    deduplicated[best_match_idx]["_similar_count"] = 0
                deduplicated[best_match_idx]["_similar_count"] += 1
                # Track IDs of similar observations
                if "_similar_ids" not in deduplicated[best_match_idx]:
                    deduplicated[best_match_idx]["_similar_ids"] = []
                if obs.get("id"):
                    deduplicated[best_match_idx]["_similar_ids"].append(obs["id"])
        else:
            deduplicated.append(obs)
            seen_texts.append(text)
            seen_indices.append(len(deduplicated) - 1)

    return deduplicated, removed_count


def deduplicate_with_embeddings(
    observations: List[Dict[str, Any]],
    embeddings: List[List[float]],
    threshold: float = 0.85,
    track_similar: bool = True,
) -> Tuple[List[Dict[str, Any]], int]:
    """Remove semantically similar observations using vector embeddings.

    More accurate than Jaccard but requires pre-computed embeddings.

    Args:
        observations: List of observation dicts
        embeddings: Pre-computed embeddings (same order as observations)
        threshold: Cosine similarity threshold (0.0-1.0)
        track_similar: If True, add "_similar_count" to retained observations

    Returns:
        Tuple of (deduplicated list, count of removed duplicates)
    """
    if not observations or not embeddings or threshold <= 0:
        return observations, 0

    if len(observations) != len(embeddings):
        # Fallback to Jaccard if embeddings don't match
        return deduplicate_observations(observations, threshold, track_similar=track_similar)

    deduplicated: List[Dict[str, Any]] = []
    kept_embeddings: List[List[float]] = []
    kept_indices: List[int] = []
    removed_count = 0

    for i, (obs, emb) in enumerate(zip(observations, embeddings)):
        if not emb:  # Empty embedding
            deduplicated.append(obs)
            kept_embeddings.append(emb)
            kept_indices.append(len(deduplicated) - 1)
            continue

        # Check cosine similarity with kept embeddings
        is_duplicate = False
        best_match_idx = -1
        best_similarity = 0.0

        for j, kept_emb in enumerate(kept_embeddings):
            if not kept_emb:
                continue
            similarity = _cosine_similarity(emb, kept_emb)
            if similarity >= threshold and similarity > best_similarity:
                is_duplicate = True
                best_match_idx = kept_indices[j]
                best_similarity = similarity

        if is_duplicate and best_match_idx >= 0:
            removed_count += 1
            if track_similar:
                if "_similar_count" not in deduplicated[best_match_idx]:
                    deduplicated[best_match_idx]["_similar_count"] = 0
                deduplicated[best_match_idx]["_similar_count"] += 1
                if "_similar_ids" not in deduplicated[best_match_idx]:
                    deduplicated[best_match_idx]["_similar_ids"] = []
                if obs.get("id"):
                    deduplicated[best_match_idx]["_similar_ids"].append(obs["id"])
        else:
            deduplicated.append(obs)
            kept_embeddings.append(emb)
            kept_indices.append(len(deduplicated) - 1)

    return deduplicated, removed_count


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


# Common stopwords to remove for compression (LLM-agnostic)
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
    "because", "until", "while", "that", "this", "these", "those", "it",
}


def compress_text(
    text: str,
    compression_level: float = 0.5,
    max_length: Optional[int] = None,
    preserve_keywords: Optional[Set[str]] = None,
) -> str:
    """Compress text by removing stopwords and truncating.

    This is a basic, LLM-agnostic compression that:
    1. Removes common stopwords (proportional to compression level)
    2. Truncates to max_length if specified
    3. Preserves important keywords if provided

    Args:
        text: Text to compress
        compression_level: 0.0 = no compression, 1.0 = max compression
        max_length: Maximum character length (None = no limit)
        preserve_keywords: Keywords to always keep

    Returns:
        Compressed text
    """
    if not text or compression_level <= 0:
        if max_length and len(text) > max_length:
            return text[:max_length - 3] + "..."
        return text

    # Preserve keywords (never remove)
    preserve = preserve_keywords or set()

    words = text.split()
    compressed_words = []

    # Remove stopwords based on compression level
    # At level 0.5, remove ~50% of stopwords
    # At level 1.0, remove all stopwords
    for word in words:
        word_lower = word.lower().strip(".,!?;:'\"()")

        # Always keep preserved keywords
        if word_lower in preserve:
            compressed_words.append(word)
            continue

        # Keep non-stopwords
        if word_lower not in STOPWORDS:
            compressed_words.append(word)
            continue

        # At high compression, remove all stopwords
        if compression_level >= 0.8:
            continue

        # At medium compression, keep some stopwords for readability
        # Keep first occurrence and every Nth after
        if compression_level < 0.3:
            compressed_words.append(word)

    result = " ".join(compressed_words)

    # Apply max length
    if max_length and len(result) > max_length:
        result = result[:max_length - 3] + "..."

    return result


def compress_observations(
    observations: List[Dict[str, Any]],
    compression_level: float = 0.5,
    field: str = "summary",
) -> List[Dict[str, Any]]:
    """Compress observations for more efficient context injection.

    Args:
        observations: List of observation dicts
        compression_level: 0.0-1.0 compression level
        field: Field to compress ("summary" or "content")

    Returns:
        List of observations with compressed text
    """
    if compression_level <= 0:
        return observations

    compressed = []
    for obs in observations:
        obs_copy = dict(obs)
        text = obs_copy.get(field) or ""
        if text:
            obs_copy[field] = compress_text(text, compression_level)
        compressed.append(obs_copy)

    return compressed


def _format_compact_index_line(obs: Dict[str, Any], score: Optional[float] = None) -> str:
    """Format a single observation as a compact index line (Layer 1).

    Target: ~10-15 tokens per line for maximum efficiency.
    Format: ID | type | 1-line summary | score
    """
    obs_id = obs.get("id", "?")[:8]  # Truncate ID for compactness
    obs_type = (obs.get("type") or "-")[:10]  # Truncate type
    summary = obs.get("summary") or obs.get("content") or ""

    # Truncate summary to ~50 chars for compactness
    if len(summary) > 50:
        summary = summary[:47] + "..."

    # Single line, minimal formatting
    score_str = f" [{score:.2f}]" if score is not None else ""
    return f"- {obs_id} | {obs_type} | {summary}{score_str}"


def format_context_ref(obs_id: str, summary: Optional[str] = None) -> str:
    """Format a context reference placeholder for Layer 1 compact mode.

    These placeholders can be expanded to full content by the LLM
    using the get_observations tool/command.

    Example: <context-ref id="obs_abc12345" hint="JWT auth config"/>
    """
    hint = ""
    if summary:
        # Truncate hint to ~30 chars
        hint_text = summary[:27] + "..." if len(summary) > 30 else summary
        hint = f' hint="{hint_text}"'

    return f'<context-ref id="{obs_id}"{hint}/>'


def format_layer1_index(
    observations: List[Dict[str, Any]],
    scoreboard: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
    include_refs: bool = True,
) -> Tuple[str, int]:
    """Format Layer 1: Compact index with optional context-ref placeholders.

    Returns: (formatted_text, token_estimate)
    """
    lines = ["## Quick Index (Layer 1)"]
    lines.append("Request full details for any ID using `get_observation <id>`\n")

    for obs in observations:
        obs_id = obs.get("id", "")
        score = None
        if scoreboard and obs_id in scoreboard:
            score = scoreboard[obs_id].get("vector_score")

        line = _format_compact_index_line(obs, score)

        # Optionally add context-ref for easy expansion
        if include_refs and obs_id:
            ref = format_context_ref(obs_id, obs.get("summary"))
            line = f"{line}  {ref}"

        lines.append(line)

    text = "\n".join(lines)
    tokens = estimate_tokens(text)
    return text, tokens


def format_layer2_timeline(
    anchor: Dict[str, Any],
    before: List[Dict[str, Any]],
    after: List[Dict[str, Any]],
) -> Tuple[str, int]:
    """Format Layer 2: Chronological timeline around anchor observation.

    Shows temporal context to help the model understand sequence of events.

    Returns: (formatted_text, token_estimate)
    """
    lines = ["## Timeline Context (Layer 2)"]
    lines.append(f"Anchor: {anchor.get('id', '?')[:8]}\n")

    # Before context
    if before:
        lines.append("**Before:**")
        for obs in reversed(before):  # Show oldest first
            ts = obs.get("created_at", "")[:16] if obs.get("created_at") else "?"
            summary = obs.get("summary") or obs.get("content") or ""
            if len(summary) > 80:
                summary = summary[:77] + "..."
            lines.append(f"  {ts} | {obs.get('id', '?')[:8]} | {summary}")

    # Anchor observation (highlighted)
    lines.append("\n**→ Current:**")
    ts = anchor.get("created_at", "")[:16] if anchor.get("created_at") else "?"
    summary = anchor.get("summary") or anchor.get("content") or ""
    if len(summary) > 80:
        summary = summary[:77] + "..."
    lines.append(f"  {ts} | {anchor.get('id', '?')[:8]} | {summary}")

    # After context
    if after:
        lines.append("\n**After:**")
        for obs in after:
            ts = obs.get("created_at", "")[:16] if obs.get("created_at") else "?"
            summary = obs.get("summary") or obs.get("content") or ""
            if len(summary) > 80:
                summary = summary[:77] + "..."
            lines.append(f"  {ts} | {obs.get('id', '?')[:8]} | {summary}")

    text = "\n".join(lines)
    tokens = estimate_tokens(text)
    return text, tokens


def format_layer3_full(
    observations: List[Dict[str, Any]],
    field: str = "content",
) -> Tuple[str, int]:
    """Format Layer 3: Full observation details.

    Only used when model explicitly requests full content.

    Returns: (formatted_text, token_estimate)
    """
    lines = ["## Full Details (Layer 3)"]

    for obs in observations:
        obs_id = obs.get("id", "?")
        obs_type = obs.get("type", "-")
        content = obs.get(field) or obs.get("content") or obs.get("summary") or ""

        lines.append(f"\n### {obs_id} ({obs_type})")
        lines.append(content)

        # Add metadata if available
        if obs.get("tags"):
            lines.append(f"\nTags: {', '.join(obs['tags'])}")

    text = "\n".join(lines)
    tokens = estimate_tokens(text)
    return text, tokens


def _match_tags(obs_tags: List[str], filters: List[str]) -> bool:
    if not filters:
        return True
    return any(tag in obs_tags for tag in filters)


def _filter_observations(
    observations: List[Dict[str, Any]],
    obs_types: List[str],
    tag_filters: List[str],
) -> List[Dict[str, Any]]:
    filtered = []
    for obs in observations:
        if obs_types and obs.get("type") not in obs_types:
            continue
        if not _match_tags(obs.get("tags") or [], tag_filters):
            continue
        filtered.append(obs)
    return filtered


def _format_index_line(obs: Dict[str, Any], token_count: int, show_tokens: bool) -> str:
    from .models import CONCEPT_ICONS

    summary = obs.get("summary") or ""
    obs_type = obs.get("type") or "-"
    concept = obs.get("concept")
    tags = obs.get("tags") or []
    tag_text = f" | tags: {', '.join(tags)}" if tags else ""
    tokens = f" (~{token_count} tok)" if show_tokens else ""

    # Show similar count if deduplication found similar observations
    similar_count = obs.get("_similar_count", 0)
    similar_text = f" [+{similar_count} similar]" if similar_count > 0 else ""

    # Get icon for concept (if present) or use type icon
    icon = ""
    if concept and concept in CONCEPT_ICONS:
        icon = f"{CONCEPT_ICONS[concept]} "
    elif obs_type:
        # Default type icons
        type_icons = {
            "bugfix": "\U0001f41e",        # Bug
            "feature": "\u2728",           # Sparkles
            "decision": "\U0001f914",      # Thinking
            "gotcha": "\U0001f534",        # Red circle
            "trade-off": "\u2696\ufe0f",   # Balance scale
        }
        icon = type_icons.get(obs_type, "")
        if icon:
            icon = f"{icon} "

    # Format concept/type display
    type_display = f"{obs_type}"
    if concept:
        type_display = f"{obs_type}/{concept}"

    return f"- {icon}{obs.get('id')} | {type_display} | {summary}{tag_text}{tokens}{similar_text}"


def _determine_disclosure_mode(
    cfg: ContextConfig,
    total_observations: int,
    estimated_full_tokens: int,
) -> str:
    """Determine the best disclosure mode based on token budget.

    Returns: "compact", "standard", or "full"
    """
    mode = cfg.disclosure_mode.lower()

    # If explicit mode set, use it
    if mode in ("compact", "standard", "full"):
        return mode

    # Auto mode: decide based on token budget
    max_budget = cfg.max_token_budget

    if max_budget is None:
        # No budget set, use standard
        return "standard"

    # If full content would exceed budget, use compact
    if estimated_full_tokens > max_budget:
        return "compact"

    # If we have room, use standard
    target = DISCLOSURE_TOKEN_TARGETS["standard"]
    if estimated_full_tokens <= target:
        return "full"

    return "standard"


def _render_header(
    project: str,
    query: Optional[str],
    show_tokens: bool,
    totals: Dict[str, int],
    economics: Optional[Dict[str, float]] = None,
    disclosure_mode: str = "standard",
    dedup_count: int = 0,
) -> List[str]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# [{project}] context ({timestamp})"]
    if query:
        lines.append(f"Query: {query}")
    if show_tokens:
        lines.append(
            "Token estimate: index ~{index} tok, full ~{full} tok, total ~{total} tok".format(**totals)
        )
        if economics and economics.get("full_total", 0) > 0:
            savings = economics.get("savings", 0.0)
            savings_pct = economics.get("savings_pct", 0.0)
            lines.append(
                "Context economics: full ~{full_total} tok, saved ~{savings} tok ({savings_pct:.1f}%)".format(
                    full_total=int(economics["full_total"]),
                    savings=int(savings),
                    savings_pct=savings_pct,
                )
            )
        if dedup_count > 0:
            lines.append(f"Deduplication: {dedup_count} similar items merged")

    lines.append("")

    # Different instructions based on disclosure mode
    if disclosure_mode == "compact":
        lines.append(
            "Quick Index (compact mode). Use observation IDs to request full details."
        )
    else:
        lines.append(
            "Context Index: summaries for quick recall. Use IDs to pull full details when needed."
        )
    lines.append("")
    return lines


async def build_compact_context(
    manager: MemoryManager,
    observations: List[Dict[str, Any]],
    scoreboard: Dict[str, Dict[str, Optional[float]]],
    project: str,
    query: Optional[str] = None,
    show_tokens: bool = True,
    wrap: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """Build Layer 1 compact context (~50-100 tokens).

    This is the most token-efficient format, containing only:
    - Observation IDs
    - Types
    - 1-line summaries (~50 chars)
    - Relevance scores

    The model can use IDs to request full details via get_observations.
    """
    lines: List[str] = []
    total_tokens = 0

    for obs in observations:
        obs_id = obs.get("id")
        score = None
        if obs_id and obs_id in scoreboard:
            sb = scoreboard[obs_id]
            # Use combined score if available
            fts = sb.get("fts_score") or 0
            vec = sb.get("vector_score") or 0
            score = (fts + vec) / 2 if fts or vec else None

        line = _format_compact_index_line(obs, score)
        lines.append(line)
        total_tokens += estimate_tokens(line)

    totals = {"index": total_tokens, "full": 0, "total": total_tokens}

    header = _render_header(
        project, query, show_tokens, totals,
        disclosure_mode="compact"
    )

    context_text = "\n".join(header + lines)
    if wrap:
        context_text = f"<ai-mem-context mode=\"compact\">\n{context_text}\n</ai-mem-context>"

    metadata = {
        "project": project,
        "query": query,
        "disclosure_mode": "compact",
        "observation_count": len(observations),
        "observation_ids": [obs.get("id") for obs in observations],
        "tokens": totals,
        "scoreboard": scoreboard,
    }

    return context_text, metadata


async def build_context(
    manager: MemoryManager,
    project: Optional[str] = None,
    query: Optional[str] = None,
    obs_type: Optional[str] = None,
    obs_types: Optional[List[str]] = None,
    tag_filters: Optional[List[str]] = None,
    concepts: Optional[List[str]] = None,  # Filter by concepts (gotcha, trade-off, etc.)
    session_id: Optional[str] = None,
    total_count: Optional[int] = None,
    full_count: Optional[int] = None,
    full_field: Optional[str] = None,
    show_tokens: Optional[bool] = None,
    wrap: Optional[bool] = None,
    config: Optional[ContextConfig] = None,
    disclosure_mode: Optional[str] = None,  # Override: "compact", "standard", "full"
    host: Optional[str] = None,  # Host identifier for adaptive behavior
    model: Optional[str] = None,  # Model identifier for token budget
) -> Tuple[str, Dict[str, Any]]:
    """Build context for injection into LLM prompts.

    Supports Progressive Disclosure with 3 layers:
    - compact: Only IDs + 1-line summaries (~50-100 tokens)
    - standard: Index + selected full content (default)
    - full: All content included

    Args:
        manager: MemoryManager instance
        project: Filter by project
        query: Search query for semantic retrieval
        obs_type: Single observation type filter
        obs_types: Multiple observation type filters
        tag_filters: Filter by tags
        session_id: Filter by session
        total_count: Number of observations in index
        full_count: Number of observations with full content
        full_field: Field for full content ("content" or "summary")
        show_tokens: Show token estimates
        wrap: Wrap in <ai-mem-context> tags
        config: Override ContextConfig
        disclosure_mode: Override disclosure mode

    Returns:
        Tuple of (context_text, metadata)
    """
    cfg = config or manager.config.context
    total_count = total_count if total_count is not None else cfg.total_observation_count
    full_count = full_count if full_count is not None else cfg.full_observation_count
    if total_count < 0:
        total_count = 0
    if full_count < 0:
        full_count = 0
    obs_type_filters = cfg.observation_types
    if obs_types is not None:
        obs_type_filters = [item for item in obs_types if item]
    if obs_type:
        obs_type_filters = [obs_type]
    tag_filters = tag_filters if tag_filters is not None else cfg.tag_filters
    full_field = (full_field or cfg.full_observation_field).lower()
    if full_field not in {"content", "summary"}:
        full_field = "content"
    show_tokens = cfg.show_token_estimates if show_tokens is None else show_tokens
    wrap = cfg.wrap_context_tag if wrap is None else wrap

    project_name = project or (f"session:{session_id}" if session_id else "all-projects")
    observations: List[Dict[str, Any]] = []
    scoreboard: Dict[str, Dict[str, Optional[float]]] = {}

    if query:
        index_items = await manager.search(
            query,
            limit=total_count,
            project=project,
            session_id=session_id,
            obs_type=obs_type,
            tag_filters=tag_filters,
        )
        if index_items:
            ids = [item.id for item in index_items]
            details = {item["id"]: item for item in await manager.get_observations(ids)}
            scoreboard = {
                item.id: {
                    "fts_score": item.fts_score,
                    "vector_score": item.vector_score,
                    "recency_factor": item.recency_factor,
                }
                for item in index_items
                if item.id
            }
            for item in index_items:
                detail = details.get(item.id)
                if not detail:
                    continue
                if scoreboard.get(item.id):
                    detail["_scoreboard"] = scoreboard[item.id]
                observations.append(detail)
    else:
        index_items = await manager.db.list_observations(
            project=project if not session_id else None,
            session_id=session_id,
            limit=total_count,
            tag_filters=tag_filters,
        )
        if index_items:
             ids = [item.id for item in index_items]
             observations = await manager.get_observations(ids)

    observations = _filter_observations(observations, obs_type_filters, tag_filters)

    # Apply concept filter if specified
    if concepts:
        observations = [
            obs for obs in observations
            if obs.get("concept") in concepts
        ]

    # Apply deduplication if enabled
    dedup_count = 0
    if cfg.enable_deduplication and observations:
        observations, dedup_count = deduplicate_observations(
            observations,
            threshold=cfg.deduplication_threshold,
            key_field="summary",
        )

    # Apply compression if enabled
    if cfg.compression_level > 0 and observations:
        observations = compress_observations(
            observations,
            compression_level=cfg.compression_level,
            field="summary",
        )

    # Estimate full tokens for auto mode determination
    estimated_full_tokens = sum(
        estimate_tokens(obs.get("content") or obs.get("summary") or "")
        for obs in observations
    )

    # Calculate adaptive token budget based on host/model
    if cfg.max_token_budget is not None:
        max_budget = cfg.max_token_budget
    elif host or model:
        max_budget = calculate_context_budget(model=model, host=host)
    else:
        max_budget = None

    # Determine disclosure mode (use host-aware logic if available)
    if disclosure_mode:
        mode = disclosure_mode
    elif host or model:
        mode = determine_optimal_disclosure_mode(
            host=host,
            model=model,
            observation_count=len(observations),
            estimated_tokens=estimated_full_tokens,
            max_budget=max_budget,
        )
    else:
        mode = _determine_disclosure_mode(cfg, len(observations), estimated_full_tokens)

    # If compact mode or compact_index_only, use compact builder
    if mode == "compact" or cfg.compact_index_only:
        return await build_compact_context(
            manager,
            observations[:total_count],
            scoreboard,
            project_name,
            query=query,
            show_tokens=show_tokens,
            wrap=wrap,
        )

    index_observations = observations[:total_count]
    # In full mode, include all as full; in standard, use full_count
    if mode == "full":
        full_observations = observations[:total_count]
    else:
        full_observations = observations[:full_count]

    totals = {"index": 0, "full": 0, "total": 0}
    index_lines: List[str] = []
    full_baseline_tokens = 0
    for obs in index_observations:
        summary_text = obs.get("summary") or obs.get("content") or ""
        token_count = estimate_tokens(summary_text)
        totals["index"] += token_count
        full_text = obs.get("content") or summary_text
        full_baseline_tokens += estimate_tokens(full_text)
        index_lines.append(_format_index_line({**obs, "summary": summary_text}, token_count, show_tokens))

    full_lines: List[str] = []
    if full_observations:
        full_lines.append("Full Context:")
        full_lines.append("")

    for obs in full_observations:
        detail_text = obs.get(full_field) or ""
        token_count = estimate_tokens(detail_text)
        totals["full"] += token_count
        full_lines.append(f"## {obs.get('id')} | {obs.get('type') or '-'}")
        full_lines.append(detail_text.strip())
        full_lines.append("")

    totals["total"] = totals["index"] + totals["full"]
    economics = {}
    if show_tokens and full_baseline_tokens > 0:
        savings = max(full_baseline_tokens - totals["total"], 0)
        savings_pct = (savings / full_baseline_tokens) * 100 if full_baseline_tokens else 0.0
        economics = {
            "full_total": full_baseline_tokens,
            "savings": savings,
            "savings_pct": savings_pct,
        }

    lines = _render_header(
        project_name, query, show_tokens, totals, economics or None,
        disclosure_mode=mode,
        dedup_count=dedup_count,
    )
    lines.extend(index_lines)
    lines.append("")
    lines.extend(full_lines)

    context_text = "\n".join(line for line in lines if line is not None)
    if wrap:
        context_text = f"<ai-mem-context mode=\"{mode}\">\n{context_text}\n</ai-mem-context>"

    metadata = {
        "project": project_name,
        "session_id": session_id,
        "query": query,
        "disclosure_mode": mode,
        "host": host,
        "model": model,
        "token_budget": max_budget,
        "index_count": len(index_observations),
        "full_count": len(full_observations),
        "dedup_count": dedup_count,
        "tokens": totals,
        "economics": economics,
        "full_field": full_field,
        "obs_types": obs_type_filters,
        "concepts": concepts,
        "scoreboard": scoreboard if query else {},
    }
    return context_text, metadata

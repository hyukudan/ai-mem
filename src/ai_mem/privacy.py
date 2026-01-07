import re
from typing import List, Optional, Tuple

MAX_TAG_COUNT = 100
PRIVATE_TAG = "private"
CONTEXT_TAGS = ("ai-mem-context", "claude-mem-context")
REDACTED_PLACEHOLDER = "[REDACTED]"


def _count_tags(content: str) -> int:
    lower_content = content.lower()
    return sum(lower_content.count(f"<{tag}>") for tag in (PRIVATE_TAG, *CONTEXT_TAGS))


def strip_memory_tags(content: str) -> Tuple[str, bool]:
    """Remove <private> and context tags from content.

    Returns:
        Tuple of (cleaned_content, was_stripped)
    """
    if not content:
        return content, False

    text = str(content)
    tag_count = _count_tags(text)
    stripped = False
    if tag_count > MAX_TAG_COUNT:
        raise ValueError(f"Too many tags ({tag_count}) found in content. Max is {MAX_TAG_COUNT}.")

    for tag in CONTEXT_TAGS + (PRIVATE_TAG,):
        pattern = re.compile(rf"<{tag}>[\s\S]*?</{tag}>", re.IGNORECASE)
        if pattern.search(text):
            stripped = True
        text = pattern.sub("", text)

    return text.strip(), stripped


def apply_redaction_patterns(
    content: str,
    patterns: Optional[List[str]] = None,
    placeholder: str = REDACTED_PLACEHOLDER,
) -> Tuple[str, bool, int]:
    """Apply regex redaction patterns to content.

    Args:
        content: The text to redact
        patterns: List of regex patterns to match and redact
        placeholder: Replacement text for redacted content

    Returns:
        Tuple of (redacted_content, was_redacted, redaction_count)
    """
    if not content or not patterns:
        return content, False, 0

    text = str(content)
    redaction_count = 0

    for pattern_str in patterns:
        try:
            pattern = re.compile(pattern_str)
            matches = pattern.findall(text)
            if matches:
                redaction_count += len(matches)
                text = pattern.sub(placeholder, text)
        except re.error:
            # Skip invalid regex patterns
            continue

    was_redacted = redaction_count > 0
    return text, was_redacted, redaction_count


def truncate_content(
    content: str,
    max_chars: int,
    suffix: str = "... [TRUNCATED]",
) -> Tuple[str, bool]:
    """Truncate content to max_chars if needed.

    Args:
        content: The text to truncate
        max_chars: Maximum allowed characters
        suffix: Text to append when truncating

    Returns:
        Tuple of (truncated_content, was_truncated)
    """
    if not content or max_chars <= 0:
        return content, False

    if len(content) <= max_chars:
        return content, False

    truncate_at = max_chars - len(suffix)
    if truncate_at < 0:
        truncate_at = 0

    return content[:truncate_at] + suffix, True


def sanitize_for_storage(
    content: str,
    redaction_patterns: Optional[List[str]] = None,
    max_chars: int = 0,
) -> Tuple[str, dict]:
    """Full sanitization pipeline for content before storage.

    Applies in order:
    1. Strip memory tags (<private>, <ai-mem-context>, etc.)
    2. Apply redaction patterns (API keys, passwords, etc.)
    3. Truncate if exceeds max_chars

    Args:
        content: The raw content to sanitize
        redaction_patterns: Optional list of regex patterns for redaction
        max_chars: Maximum characters (0 = no limit)

    Returns:
        Tuple of (sanitized_content, metadata_dict)
        metadata_dict contains: {stripped, redacted, redaction_count, truncated}
    """
    metadata = {
        "stripped": False,
        "redacted": False,
        "redaction_count": 0,
        "truncated": False,
    }

    if not content:
        return "", metadata

    # Step 1: Strip memory tags
    text, stripped = strip_memory_tags(content)
    metadata["stripped"] = stripped

    # Step 2: Apply redaction patterns
    if redaction_patterns:
        text, redacted, count = apply_redaction_patterns(text, redaction_patterns)
        metadata["redacted"] = redacted
        metadata["redaction_count"] = count

    # Step 3: Truncate if needed
    if max_chars > 0:
        text, truncated = truncate_content(text, max_chars)
        metadata["truncated"] = truncated

    return text, metadata


def should_skip_tool(
    tool_name: str,
    skip_names: Optional[List[str]] = None,
    skip_prefixes: Optional[List[str]] = None,
    skip_categories: Optional[List[str]] = None,
    tool_category: Optional[str] = None,
) -> bool:
    """Check if a tool should be skipped based on filtering rules.

    Args:
        tool_name: The name of the tool
        skip_names: List of exact tool names to skip
        skip_prefixes: List of tool name prefixes to skip
        skip_categories: List of tool categories to skip
        tool_category: The category of the tool (if provided by host)

    Returns:
        True if the tool should be skipped
    """
    if not tool_name:
        return True

    name = str(tool_name).strip()
    if not name:
        return True

    # Check exact name match
    if skip_names and name in skip_names:
        return True

    # Check prefix match
    if skip_prefixes:
        for prefix in skip_prefixes:
            if name.startswith(prefix):
                return True

    # Check category match
    if skip_categories and tool_category:
        if tool_category in skip_categories:
            return True

    return False

"""Inline Context Tags for ai-mem.

This module provides parsing and expansion of inline memory tags in prompts.
Tags like `<mem query="auth" limit="5"/>` are expanded to actual memory context.

This is LLM-agnostic and works with any prompt-based system.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Pattern to match <mem .../> tags
# Supports both self-closing and paired tags:
#   <mem query="auth"/>
#   <mem query="auth" limit="5"/>
#   <mem query="auth">additional context</mem>
MEM_TAG_PATTERN = re.compile(
    r'<mem\s+([^>]*?)\s*(?:/\s*>|>(.*?)</mem>)',
    re.IGNORECASE | re.DOTALL
)

# Pattern to extract attributes from tag
ATTR_PATTERN = re.compile(
    r'(\w+)\s*=\s*["\']([^"\']*)["\']'
)


@dataclass
class MemTag:
    """Represents a parsed <mem> tag."""
    raw: str  # Original tag text
    query: Optional[str] = None
    limit: int = 5
    project: Optional[str] = None
    obs_type: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    mode: str = "standard"  # compact, standard, full
    inner_text: Optional[str] = None  # Text between <mem>...</mem>
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class ParsedPrompt:
    """Result of parsing a prompt for mem tags."""
    original: str
    tags: List[MemTag]
    has_tags: bool = False
    cleaned: str = ""  # Prompt with tags removed


def parse_mem_tag_attrs(attrs_str: str) -> Dict[str, str]:
    """Parse attributes from a mem tag.

    Args:
        attrs_str: Attribute string like 'query="auth" limit="5"'

    Returns:
        Dict of attribute name -> value
    """
    attrs = {}
    for match in ATTR_PATTERN.finditer(attrs_str):
        name = match.group(1).lower()
        value = match.group(2)
        attrs[name] = value
    return attrs


def parse_mem_tag(match: re.Match, prompt: str) -> MemTag:
    """Parse a single mem tag match.

    Args:
        match: Regex match object
        prompt: Original prompt text

    Returns:
        MemTag object
    """
    raw = match.group(0)
    attrs_str = match.group(1)
    inner_text = match.group(2) if match.lastindex >= 2 else None

    attrs = parse_mem_tag_attrs(attrs_str)

    # Parse limit as integer
    limit = 5
    if "limit" in attrs:
        try:
            limit = int(attrs["limit"])
        except ValueError:
            pass

    # Parse tags as comma-separated list
    tag_list = []
    if "tags" in attrs:
        tag_list = [t.strip() for t in attrs["tags"].split(",") if t.strip()]

    return MemTag(
        raw=raw,
        query=attrs.get("query"),
        limit=limit,
        project=attrs.get("project"),
        obs_type=attrs.get("type") or attrs.get("obs_type"),
        tags=tag_list,
        mode=attrs.get("mode", "standard"),
        inner_text=inner_text.strip() if inner_text else None,
        start_pos=match.start(),
        end_pos=match.end(),
    )


def parse_prompt(prompt: str) -> ParsedPrompt:
    """Parse a prompt for inline mem tags.

    Looks for tags like:
    - <mem query="auth"/>
    - <mem query="database" limit="10" mode="compact"/>
    - <mem query="user" type="decision" tags="api,backend"/>
    - <mem query="config">Use these settings</mem>

    Args:
        prompt: User prompt text

    Returns:
        ParsedPrompt with extracted tags and cleaned prompt
    """
    tags: List[MemTag] = []

    for match in MEM_TAG_PATTERN.finditer(prompt):
        tag = parse_mem_tag(match, prompt)
        tags.append(tag)

    # Create cleaned prompt (tags removed)
    cleaned = prompt
    for tag in reversed(tags):  # Reverse to preserve positions
        cleaned = cleaned[:tag.start_pos] + cleaned[tag.end_pos:]

    # Clean up extra whitespace
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned).strip()

    return ParsedPrompt(
        original=prompt,
        tags=tags,
        has_tags=len(tags) > 0,
        cleaned=cleaned,
    )


def has_mem_tags(prompt: str) -> bool:
    """Quick check if prompt contains mem tags.

    Args:
        prompt: User prompt text

    Returns:
        True if prompt contains at least one mem tag
    """
    return bool(MEM_TAG_PATTERN.search(prompt))


async def expand_mem_tags(
    prompt: str,
    manager: Any,  # MemoryManager - avoid circular import
    default_project: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Expand mem tags in a prompt with actual memory content.

    This is the main entry point for inline tag expansion. It:
    1. Parses the prompt for <mem> tags
    2. Queries memory for each tag
    3. Replaces tags with actual memory context
    4. Returns the expanded prompt

    Args:
        prompt: User prompt with potential mem tags
        manager: MemoryManager instance
        default_project: Default project if not specified in tag

    Returns:
        Tuple of (expanded_prompt, list of expansion metadata)
    """
    from .context import build_context  # Avoid circular import

    parsed = parse_prompt(prompt)

    if not parsed.has_tags:
        return prompt, []

    expansions: List[Dict[str, Any]] = []
    result = prompt

    # Process tags in reverse order to preserve positions
    for tag in reversed(parsed.tags):
        # Build context for this tag
        context_text, metadata = await build_context(
            manager=manager,
            project=tag.project or default_project,
            query=tag.query,
            obs_type=tag.obs_type,
            tag_filters=tag.tags if tag.tags else None,
            total_count=tag.limit,
            full_count=min(tag.limit, 3),  # Full details for top 3
            disclosure_mode=tag.mode,
            wrap=True,
        )

        # If tag has inner text, append it to context
        if tag.inner_text:
            context_text = f"{context_text}\n\nNote: {tag.inner_text}"

        # Replace tag with expanded context
        result = result[:tag.start_pos] + context_text + result[tag.end_pos:]

        expansions.append({
            "tag": tag.raw,
            "query": tag.query,
            "limit": tag.limit,
            "mode": tag.mode,
            "results": metadata.get("index_count", 0),
            "tokens": metadata.get("tokens", {}).get("total", 0),
        })

    return result, expansions


def format_expansion_summary(expansions: List[Dict[str, Any]]) -> str:
    """Format a summary of tag expansions for display.

    Args:
        expansions: List of expansion metadata dicts

    Returns:
        Formatted summary string
    """
    if not expansions:
        return "No mem tags found in prompt"

    lines = [f"Expanded {len(expansions)} mem tag(s):"]

    for i, exp in enumerate(expansions, 1):
        query = exp.get("query", "none")
        results = exp.get("results", 0)
        tokens = exp.get("tokens", 0)
        mode = exp.get("mode", "standard")

        lines.append(
            f"  {i}. query=\"{query}\" → {results} results, ~{tokens} tokens ({mode})"
        )

    return "\n".join(lines)

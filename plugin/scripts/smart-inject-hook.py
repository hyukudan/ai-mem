#!/usr/bin/env python3
"""Smart context injection hook for ai-mem.

This hook provides intelligent context injection based on:
1. Intent detection - analyzes user prompt to determine if context is needed
2. Query expansion - uses synonyms and variations for better recall
3. Inline tag expansion - expands <mem> tags in prompts

Environment variables:
    AI_MEM_SMART_INJECT: Set to "1" to enable smart injection (default: enabled)
    AI_MEM_EXPAND_TAGS: Set to "1" to enable inline tag expansion (default: enabled)
    AI_MEM_MIN_PROMPT_LENGTH: Minimum prompt length to trigger injection (default: 20)
    AI_MEM_CONTEXT_LIMIT: Max observations to inject (default: 5)
    AI_MEM_CONTEXT_MODE: Disclosure mode: compact, standard, full (default: compact)
"""

import json
import os
import sys
from typing import Any, Dict, Optional

from common import (
    emit_continue,
    load_payload,
    resolve_project,
    resolve_session_id,
    run_ai_mem,
    stringify,
    first_present,
    get_host_identifier,
)


def get_smart_inject_enabled() -> bool:
    """Check if smart injection is enabled."""
    env_value = os.environ.get("AI_MEM_SMART_INJECT", "1").lower()
    return env_value in ("1", "true", "yes", "on")


def get_expand_tags_enabled() -> bool:
    """Check if inline tag expansion is enabled."""
    env_value = os.environ.get("AI_MEM_EXPAND_TAGS", "1").lower()
    return env_value in ("1", "true", "yes", "on")


def get_min_prompt_length() -> int:
    """Get minimum prompt length to trigger injection."""
    env_value = os.environ.get("AI_MEM_MIN_PROMPT_LENGTH", "20")
    try:
        return int(env_value)
    except ValueError:
        return 20


def get_context_limit() -> int:
    """Get max observations to inject."""
    env_value = os.environ.get("AI_MEM_CONTEXT_LIMIT", "5")
    try:
        return int(env_value)
    except ValueError:
        return 5


def get_context_mode() -> str:
    """Get disclosure mode for context."""
    return os.environ.get("AI_MEM_CONTEXT_MODE", "compact")


def detect_intent_cli(prompt: str) -> Dict[str, Any]:
    """Detect intent using CLI command."""
    result = run_ai_mem(["detect-intent", prompt, "-f", "json"])
    if result.returncode == 0 and result.stdout:
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            pass
    return {"should_inject_context": True, "query": prompt[:50]}


def expand_query_cli(query: str) -> str:
    """Expand query using CLI command."""
    result = run_ai_mem(["expand-query", query, "-f", "json"])
    if result.returncode == 0 and result.stdout:
        try:
            data = json.loads(result.stdout)
            # Use first expanded query variant
            queries = data.get("all_queries", [query])
            return queries[0] if queries else query
        except json.JSONDecodeError:
            pass
    return query


def has_mem_tags(prompt: str) -> bool:
    """Quick check if prompt contains <mem> tags."""
    return "<mem " in prompt.lower() or "<mem>" in prompt.lower()


def expand_tags_cli(prompt: str, project: str) -> str:
    """Expand inline <mem> tags using CLI command."""
    result = run_ai_mem(["expand-tags", prompt, "--project", project, "-f", "json"])
    if result.returncode == 0 and result.stdout:
        try:
            data = json.loads(result.stdout)
            return data.get("expanded", prompt)
        except json.JSONDecodeError:
            pass
    return prompt


def get_context_cli(query: str, project: str, session_id: Optional[str], limit: int, mode: str) -> str:
    """Get context using CLI command."""
    args = [
        "context",
        "--query", query,
        "--total", str(limit),
        "--full", str(min(limit, 2)),
    ]

    if session_id:
        args.extend(["--session-id", session_id])
    else:
        args.extend(["--project", project])

    # Mode is handled by disclosure_mode in context config
    # For now, use compact for efficiency

    result = run_ai_mem(args)
    if result.returncode == 0 and result.stdout:
        return result.stdout.strip()
    return ""


def emit_with_context(context: str) -> None:
    """Emit continue with injected context."""
    payload = {
        "continue": True,
        "suppressOutput": True,
    }

    if context:
        payload["systemPrompt"] = context

    print(json.dumps(payload))


def main() -> int:
    payload = load_payload()
    content_value = first_present(
        payload,
        ["prompt", "user_prompt", "userMessage", "user_message", "message", "input"],
    )
    content = stringify(content_value).strip()

    if not content:
        emit_continue()
        return 0

    project = resolve_project(payload)
    session_id = resolve_session_id(payload)
    min_length = get_min_prompt_length()

    # Check for inline <mem> tags first
    if get_expand_tags_enabled() and has_mem_tags(content):
        expanded = expand_tags_cli(content, project)
        if expanded != content:
            # Tags were expanded - emit with expanded content
            # Note: This modifies the prompt, which may need special handling
            emit_continue()
            return 0

    # Skip short prompts
    if len(content) < min_length:
        emit_continue()
        return 0

    # Check if smart injection is enabled
    if not get_smart_inject_enabled():
        emit_continue()
        return 0

    # Detect intent
    intent = detect_intent_cli(content)
    should_inject = intent.get("should_inject_context", True)

    if not should_inject:
        emit_continue()
        return 0

    # Get query from intent or expand it
    query = intent.get("query", "")
    if not query:
        query = content[:100]

    # Optionally expand query for better recall
    expanded_query = expand_query_cli(query)

    # Get context
    limit = get_context_limit()
    mode = get_context_mode()
    context = get_context_cli(expanded_query, project, session_id, limit, mode)

    if context:
        emit_with_context(context)
    else:
        emit_continue()

    # Also store the user prompt as interaction (existing behavior)
    args = [
        "add",
        content,
        "--obs-type",
        "interaction",
        "--tag",
        "user",
        "--tag",
        "claude-code",
    ]
    if session_id:
        args += ["--session-id", session_id]
    else:
        args += ["--project", project]
    run_ai_mem(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

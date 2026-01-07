#!/usr/bin/env python3
"""Tool hook for ai-mem: captures tool executions via the /api/events endpoint.

This hook implements:
- Skip list filtering (configurable via AI_MEM_SKIP_TOOL_NAMES)
- Prefix filtering (configurable via AI_MEM_SKIP_TOOL_PREFIXES)
- Failed tool filtering (configurable via AI_MEM_IGNORE_FAILED_TOOLS)
- Minimum output filter (configurable via AI_MEM_MIN_OUTPUT_CHARS)
- LLM-agnostic event ingestion (uses AI_MEM_HOST for adapter selection)
- Event ID idempotency (prevents duplicate observations from retried hooks)

The hook sends raw payloads to /api/events, which uses host-specific adapters
to parse tool events into the canonical Event Schema v1 format.
"""

import json
import os
import sys
import urllib.request
import urllib.error
import uuid

from common import (
    emit_continue,
    first_present,
    get_host_identifier,
    get_ignore_failed_tools,
    get_min_output_chars,
    load_payload,
    resolve_project,
    resolve_session_id,
    should_skip_tool,
    stringify,
)


def get_api_url() -> str:
    """Get the ai-mem API URL from environment or use default."""
    return os.environ.get("AI_MEM_API_URL", "http://localhost:37777")


def get_api_token() -> str:
    """Get the API token from environment."""
    return os.environ.get("AI_MEM_API_TOKEN", "")


def send_event_to_api(host: str, payload: dict, session_id: str = None, project: str = None) -> bool:
    """Send a raw event payload to the /api/events endpoint.

    Returns True if successful, False otherwise.
    """
    api_url = get_api_url()
    token = get_api_token()

    request_body = {
        "host": host,
        "payload": payload,
        "summarize": True,
    }

    if session_id:
        request_body["session_id"] = session_id
    if project:
        request_body["project"] = project

    data = json.dumps(request_body).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = urllib.request.Request(
            f"{api_url}/api/events",
            data=data,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status == 200
    except urllib.error.URLError:
        # Server not running or network error - fall back to CLI
        return False
    except Exception:
        return False


def fallback_to_cli(payload: dict, host: str, session_id: str, project: str) -> None:
    """Fall back to using ai-mem CLI if API is not available."""
    from common import (
        get_default_tags,
        get_max_input_chars,
        get_max_output_chars,
        run_ai_mem,
        truncate_text,
    )

    # Extract tool info manually (old behavior)
    tool_name = first_present(payload, ["tool_name", "tool", "name"])
    tool_input = first_present(payload, ["tool_input", "input", "tool_args", "arguments"])
    tool_output = first_present(payload, ["tool_response", "response", "result", "output"])

    # Truncate if needed
    max_input = get_max_input_chars()
    max_output = get_max_output_chars()

    input_str = stringify(tool_input) if tool_input is not None else ""
    output_str = stringify(tool_output) if tool_output is not None else ""

    if max_input > 0:
        input_str = truncate_text(input_str, max_input)
    if max_output > 0:
        output_str = truncate_text(output_str, max_output)

    # Build content
    parts = []
    if tool_name:
        parts.append(f"Tool: {stringify(tool_name)}")
    if input_str:
        parts.append(f"Input: {input_str}")
    if output_str:
        parts.append(f"Output: {output_str}")

    content = "\n".join(parts).strip()
    if not content:
        return

    # Build tags
    tags = get_default_tags()
    if host and host != "unknown" and host not in tags:
        tags.append(host)

    event_id = str(uuid.uuid4())

    args = ["add", content, "--obs-type", "tool_output"]
    for tag in tags:
        args += ["--tag", tag]

    if session_id:
        args += ["--session-id", session_id]
    else:
        args += ["--project", project]

    args += ["--event-id", event_id]
    if host:
        args += ["--host", host]

    run_ai_mem(args)


def main() -> int:
    payload = load_payload()

    # Extract tool name for filtering (before sending to API)
    tool_name = first_present(payload, ["tool_name", "tool", "name"])
    tool_success = first_present(payload, ["success", "tool_success", "succeeded"])
    tool_output = first_present(payload, ["tool_response", "response", "result", "output"])

    # === FILTERING LOGIC ===

    # 1. Skip if tool name is in skip list or has skip prefix
    if should_skip_tool(tool_name):
        emit_continue()
        return 0

    # 2. Skip failed tools if configured
    if get_ignore_failed_tools() and tool_success is False:
        emit_continue()
        return 0

    # 3. Check minimum output length
    output_str = stringify(tool_output) if tool_output is not None else ""
    min_output = get_min_output_chars()
    if min_output > 0 and len(output_str) < min_output:
        emit_continue()
        return 0

    # 4. Skip if no tool name
    if not tool_name:
        emit_continue()
        return 0

    # === SEND TO API ===

    host = get_host_identifier()
    session_id = resolve_session_id(payload)
    project = resolve_project(payload)

    # Try to send via API (uses adapters server-side)
    success = send_event_to_api(host, payload, session_id, project)

    if not success:
        # Fall back to CLI if API is not available
        fallback_to_cli(payload, host, session_id, project)

    emit_continue()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

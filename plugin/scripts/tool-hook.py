#!/usr/bin/env python3
"""Tool hook for ai-mem: captures tool executions with filtering and redaction.

This hook implements:
- Skip list filtering (configurable via AI_MEM_SKIP_TOOL_NAMES)
- Prefix filtering (configurable via AI_MEM_SKIP_TOOL_PREFIXES)
- Output truncation (configurable via AI_MEM_MAX_OUTPUT_CHARS)
- Input truncation (configurable via AI_MEM_MAX_INPUT_CHARS)
- Minimum output filter (configurable via AI_MEM_MIN_OUTPUT_CHARS)
- Failed tool filtering (configurable via AI_MEM_IGNORE_FAILED_TOOLS)
- LLM-agnostic tagging (uses AI_MEM_HOST instead of hardcoded "claude-code")
"""

from common import (
    emit_continue,
    first_present,
    get_default_tags,
    get_host_identifier,
    get_ignore_failed_tools,
    get_max_input_chars,
    get_max_output_chars,
    get_min_output_chars,
    load_payload,
    resolve_project,
    resolve_session_id,
    run_ai_mem,
    should_skip_tool,
    stringify,
    truncate_text,
)


def main() -> int:
    payload = load_payload()

    # Extract tool information from payload (multiple possible field names for compatibility)
    tool_name = first_present(payload, ["tool_name", "tool", "name"])
    tool_input = first_present(payload, ["tool_input", "input", "tool_args", "arguments"])
    tool_output = first_present(payload, ["tool_response", "response", "result", "output"])
    tool_success = first_present(payload, ["success", "tool_success", "succeeded"])

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

    # === CONTENT PREPARATION ===

    # Truncate input/output if too large
    max_input = get_max_input_chars()
    max_output = get_max_output_chars()

    input_str = stringify(tool_input) if tool_input is not None else ""
    if max_input > 0:
        input_str = truncate_text(input_str, max_input)
    if max_output > 0:
        output_str = truncate_text(output_str, max_output)

    # Build content only if we have something meaningful
    if not tool_name and not input_str and not output_str:
        emit_continue()
        return 0

    parts = []
    if tool_name:
        parts.append(f"Tool: {stringify(tool_name)}")
    if input_str:
        parts.append(f"Input: {input_str}")
    if output_str:
        parts.append(f"Output: {output_str}")

    content = "\n".join(parts).strip()
    if not content:
        emit_continue()
        return 0

    # === STORAGE ===

    project = resolve_project(payload)
    session_id = resolve_session_id(payload)

    # Build tags: use configurable defaults + host identifier
    tags = get_default_tags()
    host = get_host_identifier()
    if host and host != "unknown" and host not in tags:
        tags.append(host)

    args = ["add", content, "--obs-type", "tool_output"]
    for tag in tags:
        args += ["--tag", tag]

    if session_id:
        args += ["--session-id", session_id]
    else:
        args += ["--project", project]

    run_ai_mem(args)
    emit_continue()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import json

from common import emit_continue, load_payload, resolve_project, resolve_session_id, run_ai_mem, stringify, first_present


def main() -> int:
    payload = load_payload()
    tool_name = first_present(payload, ["tool_name", "tool", "name"])
    tool_input = first_present(payload, ["tool_input", "input", "tool_args", "arguments"])
    tool_output = first_present(payload, ["tool_response", "response", "result", "output"])

    if tool_name or tool_input or tool_output:
        parts = []
        if tool_name:
            parts.append(f"Tool: {stringify(tool_name)}")
        if tool_input is not None:
            parts.append(f"Input: {stringify(tool_input)}")
        if tool_output is not None:
            parts.append(f"Output: {stringify(tool_output)}")
        content = "\n".join(parts).strip()
        if content:
            project = resolve_project(payload)
            session_id = resolve_session_id(payload)
            args = [
                "add",
                content,
                "--obs-type",
                "tool_output",
                "--tag",
                "tool",
                "--tag",
                "claude-code",
            ]
            if session_id:
                args += ["--session-id", session_id]
            else:
                args += ["--project", project]
            run_ai_mem(
                args
            )

    emit_continue()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

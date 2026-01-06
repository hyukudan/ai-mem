#!/usr/bin/env python3
import os

from common import emit_continue, load_payload, resolve_project, resolve_session_id, run_ai_mem, stringify, first_present


def main() -> int:
    payload = load_payload()
    content_value = first_present(
        payload,
        ["last_assistant_message", "assistant_message", "response", "output", "message"],
    )
    content = stringify(content_value).strip()
    project = resolve_project(payload)
    session_id = resolve_session_id(payload)
    if content:
        args = [
            "add",
            content,
            "--obs-type",
            "interaction",
            "--tag",
            "assistant",
            "--tag",
            "claude-code",
        ]
        if session_id:
            args += ["--session-id", session_id]
        else:
            args += ["--project", project]
        run_ai_mem(args)
    if os.environ.get("AI_MEM_SESSION_TRACKING"):
        if session_id:
            run_ai_mem(["session-end", "--id", session_id])
        else:
            run_ai_mem(["session-end", "--project", project])
    emit_continue()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

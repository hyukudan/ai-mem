#!/usr/bin/env python3
import json

from common import emit_continue, load_payload, resolve_project, resolve_session_id, run_ai_mem, stringify, first_present


def main() -> int:
    payload = load_payload()
    content_value = first_present(
        payload,
        ["prompt", "user_prompt", "userMessage", "user_message", "message", "input"],
    )
    content = stringify(content_value).strip()
    if content:
        project = resolve_project(payload)
        session_id = resolve_session_id(payload)
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
        run_ai_mem(
            args
        )
    emit_continue()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

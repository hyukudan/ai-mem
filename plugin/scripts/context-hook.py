#!/usr/bin/env python3
import json
import sys

import os

from common import load_payload, resolve_project, resolve_session_id, run_ai_mem


def main() -> int:
    payload = load_payload()
    project = resolve_project(payload)
    session_id = resolve_session_id(payload)
    if os.environ.get("AI_MEM_SESSION_TRACKING"):
        if session_id:
            run_ai_mem(["session-start", "--project", project, "--id", session_id])
        else:
            run_ai_mem(["session-start", "--project", project])
    args = ["context"]
    if session_id:
        args += ["--session-id", session_id]
    else:
        args += ["--project", project]
    query = os.environ.get("AI_MEM_QUERY")
    total = os.environ.get("AI_MEM_CONTEXT_TOTAL")
    full = os.environ.get("AI_MEM_CONTEXT_FULL")
    full_field = os.environ.get("AI_MEM_CONTEXT_FULL_FIELD")
    tags = os.environ.get("AI_MEM_CONTEXT_TAGS")
    no_wrap = os.environ.get("AI_MEM_CONTEXT_NO_WRAP")
    if query:
        args += ["--query", query]
    if total:
        args += ["--total", total]
    if full:
        args += ["--full", full]
    if full_field:
        args += ["--full-field", full_field]
    if tags:
        for tag in tags.split(","):
            tag = tag.strip()
            if tag:
                args += ["--tag", tag]
    if no_wrap:
        args += ["--no-wrap"]
    result = run_ai_mem(args)
    context_text = result.stdout.strip()
    if sys.stdin.isatty():
        if context_text:
            print(context_text)
        return 0

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context_text,
        }
    }
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

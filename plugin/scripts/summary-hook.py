#!/usr/bin/env python3
import os

from common import emit_continue, load_payload, resolve_project, resolve_session_id, run_ai_mem


def _env_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    value = value.strip().lower()
    return value in {"1", "true", "yes", "on"}


def main() -> int:
    if not _env_enabled("AI_MEM_SUMMARY_ON_STOP"):
        emit_continue()
        return 0

    payload = load_payload()
    project = resolve_project(payload)
    session_id = resolve_session_id(payload)
    count = os.environ.get("AI_MEM_SUMMARY_COUNT", "20")
    obs_type = os.environ.get("AI_MEM_SUMMARY_OBS_TYPE")
    args = ["summarize", "--count", count]
    if session_id:
        args += ["--session-id", session_id]
    else:
        args += ["--project", project]
    if obs_type:
        args += ["--obs-type", obs_type]
    args += ["--tag", "claude-code"]
    run_ai_mem(args)
    emit_continue()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import json
import os
import sys
import subprocess
from typing import Any, Dict, Optional


def load_payload() -> Dict[str, Any]:
    if sys.stdin is None or sys.stdin.isatty():
        return {}
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        return {}
    return {}


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True)
    except TypeError:
        return str(value)


def first_present(payload: Dict[str, Any], keys: list[str]) -> Optional[Any]:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def resolve_project(payload: Dict[str, Any]) -> str:
    env_project = os.environ.get("AI_MEM_PROJECT")
    if env_project:
        return env_project
    candidate = first_present(payload, ["project", "cwd", "working_dir", "workspace"])
    if candidate:
        return str(candidate)
    return os.getcwd()


def resolve_session_id(payload: Dict[str, Any]) -> Optional[str]:
    env_session = os.environ.get("AI_MEM_SESSION_ID")
    if env_session:
        return env_session
    candidate = first_present(payload, ["session_id", "sessionId", "session"])
    if candidate:
        return str(candidate)
    return None


def ai_mem_bin() -> str:
    return os.environ.get("AI_MEM_BIN", "ai-mem")


def run_ai_mem(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [ai_mem_bin(), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def emit_continue() -> None:
    payload = {"continue": True, "suppressOutput": True}
    print(json.dumps(payload))

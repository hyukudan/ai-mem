import json
import os
import sys
import subprocess
from typing import Any, Dict, List, Optional


# Default skip list for tools that produce noise
DEFAULT_SKIP_TOOL_NAMES = [
    "SlashCommand",
    "Skill",
    "TodoWrite",
    "TodoRead",
    "AskFollowupQuestion",
    "AttemptCompletion",
    "EnterPlanMode",
    "ExitPlanMode",
]

DEFAULT_SKIP_TOOL_PREFIXES: List[str] = []

# Default redaction patterns for sensitive data
DEFAULT_REDACTION_PATTERNS = [
    r"(?i)(api[_-]?key|apikey|secret[_-]?key|password|passwd|token|bearer)\s*[=:]\s*['\"]?[\w\-\.]+['\"]?",
    r"(?i)authorization:\s*bearer\s+[\w\-\.]+",
    r"sk-[a-zA-Z0-9]{20,}",  # OpenAI API keys
    r"AIza[a-zA-Z0-9_\-]{35}",  # Google API keys
]


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


def get_skip_tool_names() -> List[str]:
    """Get list of tool names to skip from env or defaults."""
    env_value = os.environ.get("AI_MEM_SKIP_TOOL_NAMES")
    if env_value:
        return [name.strip() for name in env_value.split(",") if name.strip()]
    return DEFAULT_SKIP_TOOL_NAMES


def get_skip_tool_prefixes() -> List[str]:
    """Get list of tool name prefixes to skip from env or defaults."""
    env_value = os.environ.get("AI_MEM_SKIP_TOOL_PREFIXES")
    if env_value:
        return [prefix.strip() for prefix in env_value.split(",") if prefix.strip()]
    return DEFAULT_SKIP_TOOL_PREFIXES


def get_max_output_chars() -> int:
    """Get max output chars from env or default."""
    env_value = os.environ.get("AI_MEM_MAX_OUTPUT_CHARS")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass
    return 50000


def get_max_input_chars() -> int:
    """Get max input chars from env or default."""
    env_value = os.environ.get("AI_MEM_MAX_INPUT_CHARS")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass
    return 10000


def get_min_output_chars() -> int:
    """Get minimum output chars required to store (filter noise)."""
    env_value = os.environ.get("AI_MEM_MIN_OUTPUT_CHARS")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass
    return 0


def get_ignore_failed_tools() -> bool:
    """Check if failed tools should be ignored."""
    env_value = os.environ.get("AI_MEM_IGNORE_FAILED_TOOLS", "").lower()
    return env_value in ("1", "true", "yes", "on")


def should_skip_tool(tool_name: Optional[str]) -> bool:
    """Check if a tool should be skipped based on name/prefix rules."""
    if not tool_name:
        return True

    name = str(tool_name).strip()
    skip_names = get_skip_tool_names()
    skip_prefixes = get_skip_tool_prefixes()

    # Check exact name match
    if name in skip_names:
        return True

    # Check prefix match
    for prefix in skip_prefixes:
        if name.startswith(prefix):
            return True

    return False


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars if needed."""
    if not text or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    suffix = "... [TRUNCATED]"
    truncate_at = max_chars - len(suffix)
    if truncate_at < 0:
        truncate_at = 0
    return text[:truncate_at] + suffix


def get_default_tags() -> List[str]:
    """Get default tags for tool observations."""
    env_value = os.environ.get("AI_MEM_INGESTION_DEFAULT_TAGS")
    if env_value:
        return [tag.strip() for tag in env_value.split(",") if tag.strip()]
    return ["tool", "auto-ingested"]


def get_host_identifier() -> str:
    """Get host identifier from env (e.g., claude-code, gemini, cursor)."""
    return os.environ.get("AI_MEM_HOST", "unknown")

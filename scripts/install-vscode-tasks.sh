#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/install-vscode-tasks.sh [--dest PATH] [--bin PATH]

Adds ai-mem tasks to VS Code tasks.json in the current project.

Options:
  --dest PATH   Target tasks.json path (default: ./.vscode/tasks.json)
  --bin PATH    ai-mem binary path (default: ai-mem)
EOF
}

dest_path=".vscode/tasks.json"
bin_path="ai-mem"

while [ $# -gt 0 ]; do
  case "$1" in
    --dest)
      dest_path="${2:-}"
      shift 2
      ;;
    --bin)
      bin_path="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

dest_dir="$(dirname "$dest_path")"
mkdir -p "$dest_dir"

python3 - "$dest_path" "$bin_path" <<'PY'
import json
import os
import sys

dest_path = sys.argv[1]
bin_path = sys.argv[2]

tasks = [
    {
        "label": "ai-mem: Hook Session Start",
        "type": "shell",
        "command": bin_path,
        "args": ["hook", "session_start", "--project", "${workspaceFolder}", "--session-tracking"],
        "problemMatcher": [],
    },
    {
        "label": "ai-mem: Hook Session End",
        "type": "shell",
        "command": bin_path,
        "args": [
            "hook",
            "session_end",
            "--project",
            "${workspaceFolder}",
            "--session-tracking",
            "--summary-on-end",
        ],
        "problemMatcher": [],
    },
    {
        "label": "ai-mem: Start Server",
        "type": "shell",
        "command": bin_path,
        "args": ["server"],
        "problemMatcher": [],
        "isBackground": True,
    },
    {
        "label": "ai-mem: Start MCP",
        "type": "shell",
        "command": bin_path,
        "args": ["mcp"],
        "problemMatcher": [],
        "isBackground": True,
    },
    {
        "label": "ai-mem: Start OpenAI Proxy",
        "type": "shell",
        "command": bin_path,
        "args": ["proxy", "--port", "8081"],
        "problemMatcher": [],
        "isBackground": True,
    },
    {
        "label": "ai-mem: Start Gemini Proxy",
        "type": "shell",
        "command": bin_path,
        "args": ["gemini-proxy", "--port", "8090"],
        "problemMatcher": [],
        "isBackground": True,
    },
    {
        "label": "ai-mem: Start Anthropic Proxy",
        "type": "shell",
        "command": bin_path,
        "args": ["anthropic-proxy", "--port", "8095"],
        "problemMatcher": [],
        "isBackground": True,
    },
    {
        "label": "ai-mem: Start Azure Proxy",
        "type": "shell",
        "command": bin_path,
        "args": ["azure-proxy", "--port", "8092"],
        "problemMatcher": [],
        "isBackground": True,
    },
]

data = {"version": "2.0.0", "tasks": []}
if os.path.exists(dest_path):
    try:
        with open(dest_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            data.update({k: v for k, v in loaded.items() if k != "tasks"})
            if isinstance(loaded.get("tasks"), list):
                data["tasks"] = loaded["tasks"]
    except json.JSONDecodeError:
        pass

existing_labels = {task.get("label") for task in data.get("tasks", []) if isinstance(task, dict)}
for task in tasks:
    if task["label"] not in existing_labels:
        data.setdefault("tasks", []).append(task)

with open(dest_path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")

print(f"Updated {dest_path} with ai-mem tasks.")
PY

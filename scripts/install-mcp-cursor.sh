#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/install-mcp-cursor.sh [--config PATH] [--bin PATH] [--name NAME] [--project PATH]

Adds ai-mem to Cursor MCP config.

Options:
  --config PATH   Path to Cursor MCP config (mcp.json)
  --bin PATH      ai-mem binary path (default: ai-mem)
  --name NAME     MCP server name (default: ai-mem)
  --project PATH  Default AI_MEM_PROJECT value
EOF
}

config_path=""
bin_path="ai-mem"
server_name="ai-mem"
project_path=""

while [ $# -gt 0 ]; do
  case "$1" in
    --config)
      config_path="${2:-}"
      shift 2
      ;;
    --bin)
      bin_path="${2:-}"
      shift 2
      ;;
    --name)
      server_name="${2:-}"
      shift 2
      ;;
    --project)
      project_path="${2:-}"
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

if [ -z "$config_path" ]; then
  case "$(uname -s)" in
    Darwin)
      config_path="$HOME/Library/Application Support/Cursor/mcp.json"
      ;;
    Linux)
      if [ -d "$HOME/.cursor" ]; then
        config_path="$HOME/.cursor/mcp.json"
      else
        config_path="${XDG_CONFIG_HOME:-$HOME/.config}/Cursor/mcp.json"
      fi
      ;;
    *)
      echo "Unknown OS. Provide --config PATH." >&2
      exit 1
      ;;
  esac
fi

config_dir="$(dirname "$config_path")"
mkdir -p "$config_dir"

python3 - "$config_path" "$bin_path" "$server_name" "$project_path" <<'PY'
import json
import os
import sys

config_path = sys.argv[1]
bin_path = sys.argv[2]
server_name = sys.argv[3]
project_path = sys.argv[4]

data = {}
if os.path.exists(config_path):
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        data = {}

if not isinstance(data, dict):
    data = {}

mcp_servers = data.get("mcpServers")
if not isinstance(mcp_servers, dict):
    mcp_servers = {}

entry = {
    "command": bin_path,
    "args": ["mcp"],
}
if project_path:
    entry["env"] = {"AI_MEM_PROJECT": project_path}

mcp_servers[server_name] = entry
data["mcpServers"] = mcp_servers

with open(config_path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")

print(f"Updated {config_path} with MCP server '{server_name}'.")
PY

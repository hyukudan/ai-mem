#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/install-jetbrains-tools.sh [--dest PATH] [--bin PATH] [--force]

Adds ai-mem External Tools to a JetBrains IDE project.

Options:
  --dest PATH   Target XML path (default: ./.idea/tools/ai-mem.xml)
  --bin PATH    ai-mem binary path (default: ai-mem)
  --force       Overwrite existing file
EOF
}

dest_path=".idea/tools/ai-mem.xml"
bin_path="ai-mem"
force="false"

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
    --force)
      force="true"
      shift 1
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

if [ -f "$dest_path" ] && [ "$force" != "true" ]; then
  echo "File already exists: $dest_path (use --force to overwrite)"
  exit 0
fi

python3 - "$dest_path" "$bin_path" <<'PY'
import sys

dest_path = sys.argv[1]
bin_path = sys.argv[2]

tools = [
    ("ai-mem: Hook Session Start", "hook session_start --project $ProjectFileDir$ --session-tracking"),
    ("ai-mem: Hook Session End", "hook session_end --project $ProjectFileDir$ --session-tracking --summary-on-end"),
    ("ai-mem: Start Server", "server"),
    ("ai-mem: Start MCP", "mcp"),
    ("ai-mem: Start OpenAI Proxy", "proxy --port 8081"),
    ("ai-mem: Start Gemini Proxy", "gemini-proxy --port 8090"),
    ("ai-mem: Start Anthropic Proxy", "anthropic-proxy --port 8095"),
    ("ai-mem: Start Azure Proxy", "azure-proxy --port 8092"),
]

tool_entries = []
for name, params in tools:
    tool_entries.append(
        f"""  <tool name="{name}" description="" showInMainMenu="true" showInEditor="true" showInProject="true" showInSearchPopup="true" disabled="false" useConsole="true" showConsoleOnStdOut="false" showConsoleOnStdErr="false" synchronizeAfterRun="false">
    <exec>
      <option name="COMMAND" value="{bin_path}"/>
      <option name="PARAMETERS" value="{params}"/>
      <option name="WORKING_DIRECTORY" value="$ProjectFileDir$"/>
    </exec>
  </tool>"""
    )

xml = """<toolSet name="ai-mem">
{tools}
</toolSet>
""".format(tools="\n".join(tool_entries))

with open(dest_path, "w", encoding="utf-8") as handle:
    handle.write(xml)

print(f"Wrote {dest_path}")
PY

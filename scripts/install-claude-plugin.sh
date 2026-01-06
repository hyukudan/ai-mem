#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT="${CLAUDE_PLUGIN_ROOT:-$HOME/.claude/plugins/local}"
TARGET_DIR="$TARGET_ROOT/ai-mem"

mkdir -p "$TARGET_ROOT"

if [ -e "$TARGET_DIR" ]; then
  echo "Plugin already exists at $TARGET_DIR"
  exit 0
fi

ln -s "$(pwd)/plugin" "$TARGET_DIR"
echo "Installed ai-mem Claude Code plugin at $TARGET_DIR"
echo "Set AI_MEM_BIN if needed, e.g.:"
echo "  export AI_MEM_BIN=\"$(pwd)/.venv/bin/ai-mem\""

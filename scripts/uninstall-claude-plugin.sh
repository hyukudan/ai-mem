#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT="${CLAUDE_PLUGIN_ROOT:-$HOME/.claude/plugins/local}"
TARGET_DIR="$TARGET_ROOT/ai-mem"

if [ ! -e "$TARGET_DIR" ]; then
  echo "No ai-mem plugin found at $TARGET_DIR"
  exit 0
fi

rm -rf "$TARGET_DIR"
echo "Removed ai-mem Claude Code plugin from $TARGET_DIR"

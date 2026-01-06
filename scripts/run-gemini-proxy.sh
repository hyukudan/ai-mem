#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run ./scripts/bootstrap.sh first."
  exit 1
fi

source .venv/bin/activate

PORT="${AI_MEM_GEMINI_PROXY_PORT:-8090}"

has_port=false
for arg in "$@"; do
  if [ "$arg" = "--port" ]; then
    has_port=true
    break
  fi
done

if [ "$has_port" = false ]; then
  set -- "$@" --port "$PORT"
fi

exec ai-mem gemini-proxy "$@"

#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  ./scripts/bootstrap.sh
fi

source .venv/bin/activate

proxy_port="${AI_MEM_ANTHROPIC_PROXY_PORT:-8095}"

if [ -z "${AI_MEM_ANTHROPIC_API_KEY:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "Warning: no Anthropic API key set (AI_MEM_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY)." >&2
fi

ai-mem anthropic-proxy --port "$proxy_port"

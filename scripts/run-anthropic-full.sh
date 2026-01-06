#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  ./scripts/bootstrap.sh
fi

source .venv/bin/activate

server_port="${AI_MEM_PORT:-8000}"
proxy_port="${AI_MEM_ANTHROPIC_PROXY_PORT:-8095}"

if [ -z "${AI_MEM_ANTHROPIC_API_KEY:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "Warning: no Anthropic API key set (AI_MEM_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY)." >&2
fi

ai-mem server --port "$server_port" &
server_pid=$!

ai-mem anthropic-proxy --port "$proxy_port" &
proxy_pid=$!

cleanup() {
  if [ -n "${server_pid:-}" ]; then
    kill "$server_pid" 2>/dev/null || true
  fi
  if [ -n "${proxy_pid:-}" ]; then
    kill "$proxy_pid" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

echo "ai-mem server running at http://localhost:${server_port}"
echo "ai-mem Anthropic proxy running at http://localhost:${proxy_port}"
echo "Starting MCP stdio server..."
ai-mem mcp

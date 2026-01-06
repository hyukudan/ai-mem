#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  ./scripts/bootstrap.sh
fi

source .venv/bin/activate

server_port="${AI_MEM_PORT:-8000}"
proxy_port="${AI_MEM_PROXY_PORT:-8081}"
proxy_upstream="${AI_MEM_PROXY_UPSTREAM_BASE_URL:-}"

if [ -z "$proxy_upstream" ]; then
  echo "AI_MEM_PROXY_UPSTREAM_BASE_URL is required to start the proxy." >&2
  exit 1
fi

ai-mem server --port "$server_port" &
server_pid=$!

ai-mem proxy --port "$proxy_port" --upstream "$proxy_upstream" &
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
echo "ai-mem proxy running at http://localhost:${proxy_port}"
wait

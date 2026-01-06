#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  ./scripts/bootstrap.sh
fi

source .venv/bin/activate

server_port="${AI_MEM_PORT:-8000}"
proxy_port="${AI_MEM_GEMINI_PROXY_PORT:-8090}"

if [ -z "${AI_MEM_GEMINI_API_KEY:-}" ] && [ -z "${GOOGLE_API_KEY:-}" ]; then
  echo "Warning: no Gemini API key set (AI_MEM_GEMINI_API_KEY or GOOGLE_API_KEY)." >&2
fi

ai-mem server --port "$server_port" &
server_pid=$!

ai-mem gemini-proxy --port "$proxy_port" &
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
echo "ai-mem Gemini proxy running at http://localhost:${proxy_port}"
wait

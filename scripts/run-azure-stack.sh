#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  ./scripts/bootstrap.sh
fi

source .venv/bin/activate

server_port="${AI_MEM_PORT:-8000}"
proxy_port="${AI_MEM_AZURE_PROXY_PORT:-8092}"

if [ -z "${AI_MEM_AZURE_UPSTREAM_BASE_URL:-}" ] && [ -z "${AZURE_OPENAI_ENDPOINT:-}" ]; then
  echo "Warning: no Azure endpoint set (AI_MEM_AZURE_UPSTREAM_BASE_URL or AZURE_OPENAI_ENDPOINT)." >&2
fi
if [ -z "${AI_MEM_AZURE_API_KEY:-}" ] && [ -z "${AZURE_OPENAI_API_KEY:-}" ]; then
  echo "Warning: no Azure API key set (AI_MEM_AZURE_API_KEY or AZURE_OPENAI_API_KEY)." >&2
fi
if [ -z "${AI_MEM_AZURE_DEPLOYMENT:-}" ] && [ -z "${AZURE_OPENAI_DEPLOYMENT:-}" ]; then
  echo "Warning: no Azure deployment set (AI_MEM_AZURE_DEPLOYMENT or AZURE_OPENAI_DEPLOYMENT)." >&2
fi

ai-mem server --port "$server_port" &
server_pid=$!

ai-mem azure-proxy --port "$proxy_port" &
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
echo "ai-mem Azure proxy running at http://localhost:${proxy_port}"
wait

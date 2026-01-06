#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  ./scripts/bootstrap.sh
fi

source .venv/bin/activate

server_port="${AI_MEM_PORT:-8000}"
openai_port="${AI_MEM_PROXY_PORT:-8081}"
gemini_port="${AI_MEM_GEMINI_PROXY_PORT:-8090}"
openai_upstream="${AI_MEM_PROXY_UPSTREAM_BASE_URL:-}"

if [ -z "$openai_upstream" ]; then
  echo "AI_MEM_PROXY_UPSTREAM_BASE_URL is required to start the OpenAI-compatible proxy." >&2
  exit 1
fi

if [ -z "${AI_MEM_GEMINI_API_KEY:-}" ] && [ -z "${GOOGLE_API_KEY:-}" ]; then
  echo "Warning: no Gemini API key set (AI_MEM_GEMINI_API_KEY or GOOGLE_API_KEY)." >&2
fi

ai-mem server --port "$server_port" &
server_pid=$!

ai-mem proxy --port "$openai_port" --upstream "$openai_upstream" &
openai_pid=$!

ai-mem gemini-proxy --port "$gemini_port" &
gemini_pid=$!

cleanup() {
  if [ -n "${server_pid:-}" ]; then
    kill "$server_pid" 2>/dev/null || true
  fi
  if [ -n "${openai_pid:-}" ]; then
    kill "$openai_pid" 2>/dev/null || true
  fi
  if [ -n "${gemini_pid:-}" ]; then
    kill "$gemini_pid" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

echo "ai-mem server running at http://localhost:${server_port}"
echo "OpenAI-compatible proxy running at http://localhost:${openai_port}"
echo "Gemini proxy running at http://localhost:${gemini_port}"
echo "Starting MCP stdio server..."
ai-mem mcp

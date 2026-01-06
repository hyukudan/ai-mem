#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  ./scripts/bootstrap.sh
fi

source .venv/bin/activate

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

ai-mem azure-proxy --port "$proxy_port"

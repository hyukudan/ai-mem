#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  ./scripts/bootstrap.sh
fi

source .venv/bin/activate

proxy_port="${AI_MEM_BEDROCK_PROXY_PORT:-8094}"

if [ -z "${AI_MEM_BEDROCK_MODEL:-}" ]; then
  echo "Warning: no Bedrock model set (AI_MEM_BEDROCK_MODEL)." >&2
fi

ai-mem bedrock-proxy --port "$proxy_port"

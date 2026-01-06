#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run ./scripts/bootstrap.sh first."
  exit 1
fi

source .venv/bin/activate
exec ai-mem proxy "$@"

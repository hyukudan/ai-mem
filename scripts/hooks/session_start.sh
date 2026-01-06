#!/usr/bin/env bash
set -euo pipefail

AI_MEM_BIN="${AI_MEM_BIN:-ai-mem}"
PROJECT="${AI_MEM_PROJECT:-${PWD}}"
SESSION_ID="${AI_MEM_SESSION_ID:-}"
QUERY="${AI_MEM_QUERY:-}"
TOTAL="${AI_MEM_CONTEXT_TOTAL:-}"
FULL="${AI_MEM_CONTEXT_FULL:-}"
FULL_FIELD="${AI_MEM_CONTEXT_FULL_FIELD:-}"
CONTEXT_TAGS="${AI_MEM_CONTEXT_TAGS:-}"
NO_WRAP="${AI_MEM_CONTEXT_NO_WRAP:-}"

if [ -n "${AI_MEM_SESSION_TRACKING:-}" ]; then
  if [ -n "$SESSION_ID" ]; then
    "$AI_MEM_BIN" session-start --project "$PROJECT" --id "$SESSION_ID" >/dev/null 2>&1 || true
  else
    "$AI_MEM_BIN" session-start --project "$PROJECT" >/dev/null 2>&1 || true
  fi
fi

CMD=("$AI_MEM_BIN" context)
if [ -n "$SESSION_ID" ]; then
  CMD+=(--session-id "$SESSION_ID")
else
  CMD+=(--project "$PROJECT")
fi

if [ -n "$QUERY" ]; then
  CMD+=(--query "$QUERY")
fi
if [ -n "$TOTAL" ]; then
  CMD+=(--total "$TOTAL")
fi
if [ -n "$FULL" ]; then
  CMD+=(--full "$FULL")
fi
if [ -n "$FULL_FIELD" ]; then
  CMD+=(--full-field "$FULL_FIELD")
fi
if [ -n "$CONTEXT_TAGS" ]; then
  IFS=',' read -r -a TAGS <<< "$CONTEXT_TAGS"
  for tag in "${TAGS[@]}"; do
    if [ -n "$tag" ]; then
      CMD+=(--tag "$tag")
    fi
  done
fi
if [ -n "$NO_WRAP" ]; then
  CMD+=(--no-wrap)
fi

"${CMD[@]}"

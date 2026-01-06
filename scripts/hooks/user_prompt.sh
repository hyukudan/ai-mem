#!/usr/bin/env bash
set -euo pipefail

CONTENT="${AI_MEM_CONTENT:-}"
if [ -z "$CONTENT" ]; then
  exit 0
fi

AI_MEM_BIN="${AI_MEM_BIN:-ai-mem}"
PROJECT="${AI_MEM_PROJECT:-${PWD}}"
SESSION_ID="${AI_MEM_SESSION_ID:-}"
OBS_TYPE="${AI_MEM_OBS_TYPE:-interaction}"
NO_SUMMARY="${AI_MEM_NO_SUMMARY:-}"

TAG_ARGS=(--tag user)
if [ -n "${AI_MEM_TAGS:-}" ]; then
  IFS=',' read -r -a TAGS <<< "$AI_MEM_TAGS"
  for tag in "${TAGS[@]}"; do
    if [ -n "$tag" ]; then
      TAG_ARGS+=(--tag "$tag")
    fi
  done
fi

CMD=("$AI_MEM_BIN" add "$CONTENT" --obs-type "$OBS_TYPE")
if [ -n "$SESSION_ID" ]; then
  CMD+=(--session-id "$SESSION_ID")
else
  CMD+=(--project "$PROJECT")
fi
if [ -n "$NO_SUMMARY" ]; then
  CMD+=(--no-summary)
fi
CMD+=("${TAG_ARGS[@]}")

"${CMD[@]}"

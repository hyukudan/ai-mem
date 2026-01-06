#!/usr/bin/env bash
set -euo pipefail

CONTENT="${AI_MEM_CONTENT:-}"
AI_MEM_BIN="${AI_MEM_BIN:-ai-mem}"
PROJECT="${AI_MEM_PROJECT:-${PWD}}"
SESSION_ID="${AI_MEM_SESSION_ID:-}"
OBS_TYPE="${AI_MEM_OBS_TYPE:-note}"
NO_SUMMARY="${AI_MEM_NO_SUMMARY:-}"

TAG_ARGS=(--tag session_end)
if [ -n "${AI_MEM_TAGS:-}" ]; then
  IFS=',' read -r -a TAGS <<< "$AI_MEM_TAGS"
  for tag in "${TAGS[@]}"; do
    if [ -n "$tag" ]; then
      TAG_ARGS+=(--tag "$tag")
    fi
  done
fi

if [ -n "$CONTENT" ]; then
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
fi

if [ -n "${AI_MEM_SESSION_TRACKING:-}" ]; then
  if [ -n "$SESSION_ID" ]; then
    "$AI_MEM_BIN" session-end --id "$SESSION_ID" >/dev/null 2>&1 || true
  else
    "$AI_MEM_BIN" session-end --project "$PROJECT" >/dev/null 2>&1 || true
  fi
fi

if [ -n "${AI_MEM_SUMMARY_ON_END:-}" ]; then
  SUMMARY_COUNT="${AI_MEM_SUMMARY_COUNT:-20}"
  SUMMARY_TYPE="${AI_MEM_SUMMARY_OBS_TYPE:-}"
  SUMMARY_CMD=("$AI_MEM_BIN" summarize --count "$SUMMARY_COUNT")
  if [ -n "$SESSION_ID" ]; then
    SUMMARY_CMD+=(--session-id "$SESSION_ID")
  else
    SUMMARY_CMD+=(--project "$PROJECT")
  fi
  if [ -n "$SUMMARY_TYPE" ]; then
    SUMMARY_CMD+=(--obs-type "$SUMMARY_TYPE")
  fi
  "${SUMMARY_CMD[@]}" >/dev/null 2>&1 || true
fi

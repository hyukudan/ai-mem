# Claude Code Plugin (Local)

This folder contains a minimal Claude Code plugin that wires Claude hooks to ai-mem.

## Install (local plugin)

1. Create a local plugin folder for Claude Code (example path):
   `~/.claude/plugins/local/ai-mem`
2. Copy the `plugin/` contents into that folder (or symlink the folder).
3. Set `AI_MEM_BIN` to your ai-mem binary (for venvs):

```bash
export AI_MEM_BIN="$PWD/.venv/bin/ai-mem"
```

## What it does

- Injects ai-mem context at `SessionStart`.
- Stores user prompts at `UserPromptSubmit`.
- Stores tool input/output at `PostToolUse`.
- Stores assistant messages at `Stop` (best-effort).

The scripts are defensive: if fields are missing, they skip storage.
All stored items are tagged with `claude-code` for easy filtering.

Env overrides:
- `AI_MEM_BIN`: path to ai-mem binary.
- `AI_MEM_PROJECT`: force project path.
- `AI_MEM_SESSION_ID`: scope context + stored observations to a specific session.
- `AI_MEM_QUERY`: context query for SessionStart.
- `AI_MEM_CONTEXT_TOTAL`, `AI_MEM_CONTEXT_FULL`, `AI_MEM_CONTEXT_FULL_FIELD`, `AI_MEM_CONTEXT_TAGS`, `AI_MEM_CONTEXT_NO_WRAP`
- `AI_MEM_SESSION_TRACKING`: when set, SessionStart opens a session and Stop closes it.
- `AI_MEM_SUMMARY_ON_STOP`: enable summary generation on Stop hook.
- `AI_MEM_SUMMARY_COUNT`: number of recent observations to summarize (default 20).
- `AI_MEM_SUMMARY_OBS_TYPE`: optional observation type filter for summaries.

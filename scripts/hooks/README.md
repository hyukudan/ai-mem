# Hook Scripts

These scripts provide a lightweight hook layer similar to `claude-mem`, but usable with any LLM client that can call shell hooks.

## Available Hooks

- `session_start.sh`: print context to inject at session start.
- `user_prompt.sh`: store user prompts.
- `assistant_response.sh`: store assistant responses.
- `tool_output.sh`: store tool output or logs.
- `session_end.sh`: store a session summary or closing note.

## Environment Variables

Common:
- `AI_MEM_BIN`: override `ai-mem` binary (e.g. `.venv/bin/ai-mem`).
- `AI_MEM_PROJECT`: override project path (defaults to current working directory).
- `AI_MEM_SESSION_ID`: scope context + stored observations to a specific session.
- `AI_MEM_CONTENT`: content to store (required for prompt/response/tool/end hooks).
- `AI_MEM_TAGS`: comma-separated tags to add.
- `AI_MEM_OBS_TYPE`: override observation type (defaults to sensible values per hook).
- `AI_MEM_NO_SUMMARY`: disable summarization (set to any value to enable).
- `AI_MEM_SESSION_TRACKING`: when set, `session_start.sh` opens a session and `session_end.sh` closes the latest one.

Context hook only:
- `AI_MEM_QUERY`: optional query to bias context.
- `AI_MEM_CONTEXT_TOTAL`: max index items.
- `AI_MEM_CONTEXT_FULL`: max full items.
- `AI_MEM_CONTEXT_FULL_FIELD`: `content` or `summary`.
- `AI_MEM_CONTEXT_TAGS`: comma-separated tag filters for context.
- `AI_MEM_CONTEXT_NO_WRAP`: disable `<ai-mem-context>` wrapper.

## Example

```bash
export AI_MEM_PROJECT="$PWD"
export AI_MEM_CONTENT="User asked about OAuth refresh tokens."
scripts/hooks/user_prompt.sh
```

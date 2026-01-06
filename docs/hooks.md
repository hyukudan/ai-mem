# Hooks

## Hooks (Model Agnostic)

Use scripts in `scripts/hooks` with any client that supports shell hooks.

Examples:

```bash
# Session start: print context
AI_MEM_PROJECT="$PWD" scripts/hooks/session_start.sh

# Store a user prompt
AI_MEM_PROJECT="$PWD" AI_MEM_CONTENT="Fix OAuth flow" scripts/hooks/user_prompt.sh
```

You can also use the CLI hook runner (no shell scripts required):

```bash
# Session start: print context
ai-mem hook session_start --project "$PWD"

# Store a user prompt from stdin
echo "Fix OAuth flow" | ai-mem hook user_prompt --project "$PWD" --content-file -
```

You can pass metadata and attachments to the hook runner as well, for example:

```bash
ai-mem hook user_prompt --metadata author=sergio --file docs/notes.md
```

Generate a hook config snippet for clients that accept command hooks:

```bash
ai-mem hook-config --bin "$PWD/.venv/bin/ai-mem" --project "$PWD"
```

Hook environment variables (common):
- `AI_MEM_BIN`: path to ai-mem binary
- `AI_MEM_PROJECT`: project path
- `AI_MEM_SESSION_ID`: scope context + storage to a specific session
- `AI_MEM_CONTENT`: content to store
- `AI_MEM_TAGS`: comma-separated tags
- `AI_MEM_OBS_TYPE`: override type
- `AI_MEM_NO_SUMMARY`: disable summarization
- `AI_MEM_SESSION_TRACKING`: session_start opens, session_end closes
- `AI_MEM_SUMMARY_ON_END`: run summarize after session_end
- `AI_MEM_SUMMARY_COUNT`: number of observations to summarize (default 20)
- `AI_MEM_SUMMARY_OBS_TYPE`: filter observation type for summaries
- `AI_MEM_METADATA`: JSON or `key=value` pairs to attach extra metadata
- `AI_MEM_ASSET_FILES`: comma-separated file paths whose contents are stored as file assets
- `AI_MEM_ASSET_DIFFS`: comma-separated patch/diff paths for diff assets

Hooks can auto-load a local env file before running:

- Default: `hooks.env` in the same directory as the hook script.
- Override with `AI_MEM_HOOKS_ENV=/path/to/hooks.env`.

Install the hooks into a shared location:

```bash
./scripts/install-hooks.sh
```

## IDE Integrations

Helper scripts are included for popular IDEs:

- VS Code tasks: `./scripts/install-vscode-tasks.sh`
- JetBrains External Tools: `./scripts/install-jetbrains-tools.sh`
- Cursor MCP: `./scripts/install-mcp-cursor.sh`

These install project-local tasks or config entries so you can launch ai-mem servers, proxies, and hooks from inside the IDE.

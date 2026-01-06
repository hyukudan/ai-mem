# MCP Tools and Integrations

## MCP Tools (Claude Desktop and others)

Start MCP stdio server:

```bash
ai-mem mcp
```

Add `ai-mem` to Claude Desktop MCP config:

```bash
./scripts/install-mcp-claude-desktop.sh --bin "$PWD/.venv/bin/ai-mem"
```

Add `ai-mem` to Cursor MCP config:

```bash
./scripts/install-mcp-cursor.sh --bin "$PWD/.venv/bin/ai-mem"
```

Add `ai-mem` tasks to VS Code:

```bash
./scripts/install-vscode-tasks.sh --bin "$PWD/.venv/bin/ai-mem"
```

Add `ai-mem` External Tools to JetBrains:

```bash
./scripts/install-jetbrains-tools.sh --bin "$PWD/.venv/bin/ai-mem"
```

Generate an MCP config snippet for any MCP client:

```bash
ai-mem mcp-config --bin "$PWD/.venv/bin/ai-mem"
ai-mem mcp-config --full --name ai-mem --project "$PWD"
```

Tools:
- `search`
- `mem-search` (alias)
- `timeline`
- `get_observations`
- `summarize`
- `context`
- `stats`
- `tags`
- `tag-add`
- `tag-rename`
- `tag-delete`

Search and timeline accept `session_id` to scope results.

### Ranking metadata

`mem-search` and `timeline` now return structured JSON with three top-level keys: `results` / `timeline` (arrays of observation indices), `scoreboard` (FTS/vector/recency scores per ID), and `cache` (summary of search cache hits/misses/TTL). Clients can parse the `scoreboard` to explain why a memory matched and surface the raw scores back to the LLM.

Use the `context` tool with `output=json` (or `format=json`) to receive both the wrapped context text and the metadata that includes the same `scoreboard` map for the injected observations.

## Claude Code Plugin

A local plugin is available in `plugin/`.

```bash
./scripts/install-claude-plugin.sh
```

Set `AI_MEM_BIN` for venvs:

```bash
export AI_MEM_BIN="$PWD/.venv/bin/ai-mem"
```

See `plugin/README.md` for details.

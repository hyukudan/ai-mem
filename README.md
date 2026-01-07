# ai-mem: Universal Long-Term Memory for LLMs

![ai-mem banner](assets/banner.png)
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Status](https://img.shields.io/badge/status-active-success)

 ai-mem is a local-first memory layer that serves **any large language model**, whether it's Gemini (CLI/proxy), Claude, ChatGPT, Anthropic, Azure OpenAI, AWS Bedrock, or another vLLM setup. Observations live in SQLite (with FTS5) + a vector store, memory context is generated on demand, and the same context feeds every client through CLI, REST, or the MCP protocol.

## Why ai-mem matters

- **Open memory space for every model** – Claude, Gemini, ChatGPT, and other assistants consume the same context stream, so a discovery you capture from one model appears automatically in all other agents that query the shared store.
- **Transparent relevance insights** – Each context exposure returns a token scoreboard + cache health details, making it easy to see why memories were selected and to tune injection windows without opening the UI.
- **Private, local-first runtime** – All data is stored on disk via SQLite/FTS5 and Chroma (vector store), so you can run without cloud APIs and keep everything inside your workstation.
- **Composable integrations** – Includes hooks for IDEs (VS Code, JetBrains, Antigravity, Claude Desktop, etc.), MCP clients, shell scripts, and CLI helpers, so no manual instrumentation is required.
- **Persistent, adaptive memory** – “Endless Mode” auto-refreshes context, while `snapshot merge` lets you surface long-lived checkpoints across tasks.

## Getting started (local-first)

1. **Bootstrap the environment**

   ```bash
   ./scripts/bootstrap.sh
   source .venv/bin/activate
   ```

   This script creates a Python virtualenv, pins tooling, and installs `ai-mem` in editable mode.

2. **Capture memory**

   ```bash
   ai-mem add "We use Python 3.11 and pandas 2.0"
   ai-mem search "Python dependencies"
   ```

3. **Share context across models**

   - `ai-mem context ...` formats context with `<ai-mem-context>` wrappers for easy injection into Claude Desktop, Gemini CLI, or other assistants.
   - `ai-mem timeline` provides progressive disclosure (search → timeline → full detail).
   - `ai-mem endless` keeps regenerating context, prints token totals/scoreboard, and adapts the window automatically to stay within your token budget.

4. **Persist checkpoints**

   ```bash
   ai-mem snapshot export path/to/snapshot.ndjson
   ai-mem snapshot merge <checkpoint-id>
   ```

   Snapshots dump observations so you can sync across sessions or share with other team members. `snapshot merge` imports an existing checkpoint into the local store while keeping provenance metadata.

5. **Launch the UI**

   ```bash
   ./scripts/run.sh        # starts the web viewer and MCP server at http://localhost:37777
   ./scripts/run-gemini-full.sh  # proxy + Gemini-native integration
   ```

   The UI streams live observations, shows cache health, exposes scoreboard metadata, and provides links to `/api/observation/{id}` for citations.

## Featured capabilities

- **Model-agnostic context streaming** – Context chunks are formatted via the same helper no matter which model requests them, so your Claude Desktop session, Gemini CLI, or ChatGPT plugin can all read from a single source of truth.
- **Multi-LLM simultaneous workflows** – Run Claude in one terminal, Gemini in another, Cursor in your IDE – all writing to the same shared memory. Each host tags its events so you can trace provenance.
- **Ingestion filtering & redaction** – Skip noisy tools (TodoWrite, SlashCommand), truncate large outputs, and auto-redact secrets (API keys, passwords) before storage. See [Configuration Guide](docs/configuration.md#ingestion-filtering).
- **Event Schema v1 + idempotency** – Canonical event format for all hosts, with event_id-based deduplication to prevent duplicate observations from retried hooks.
- **Host adapters** – Built-in adapters translate Claude, Gemini, and custom payload formats to the unified Event Schema, so any LLM host can plug in without code changes.
- **Shared metadata & scoreboard** – Responses include metadata that enumerates vector vs. FTS scores, recency, and cache hits/misses, letting you introspect model prompts before each completion.
- **Context deduplication** – Automatically removes semantically similar observations from context, showing `[+N similar]` indicators to reduce token waste while preserving information.
- **Two-stage retrieval** – Configurable reranking with bi-encoder, TF-IDF, or cross-encoder strategies for improved search precision.
- **Memory consolidation & decay** – Merge duplicate observations and automatically clean up old, rarely-accessed memories to keep the store lean.
- **Endless Mode** – `ai-mem endless` polls the store at a configurable interval to keep total tokens within limits. See [Endless Mode Guide](docs/endless_mode.md).
- **Snapshot-based syncing** – `ai-mem snapshot export`/`import`/`merge` round-trips checkpoints. See [Snapshots Guide](docs/snapshots.md).
- **Hooks & IDE scripts** – Run `./scripts/install-hooks.sh`, `./scripts/install-vscode-tasks.sh`, `./scripts/install-jetbrains-tools.sh`, or the Antigravity/Claude installers to make every hook call `ai-mem hook ...` automatically.
- **Web viewer + MCP server** – Live stream + scoreboard at `http://localhost:37777`, plus REST endpoints that expose observations, scorecards, and citations for external agents like Gemini, Claude, or Antigravity.
- **CLI-first control** – All key flows (add/search/context/timeline/endless/snapshot) are available via `ai-mem` so you can script onboarding via a single CLI.
- **Scripts library** – `./scripts/run*.sh` cover full/stack/proxy setups for Gemini, Claude, Bedrock, Azure, and Anthropic deployments, ensuring the UI + MCP server is always upstream.

## Advanced Features

### Knowledge Graph

Extract and traverse entity relationships from your observations:

```bash
# List entities in the knowledge graph
ai-mem entities --name "Auth"

# Find related entities (multi-hop traversal)
ai-mem related-entities <entity-id> --depth 2

# View graph statistics
ai-mem graph-stats
```

Entity types: `file`, `function`, `class`, `module`, `endpoint`, `error`, `concept`, `technology`.

### User-Level Memory

Store global preferences that persist across projects:

```bash
# Add a user preference
ai-mem user-add "I prefer TypeScript over JavaScript" --type preference

# List user memories
ai-mem user-list

# Export/import for backup or sync
ai-mem user-export -o ~/my-preferences.json
ai-mem user-import -i ~/my-preferences.json
```

### Query Expansion

Improve search recall with automatic synonym expansion:

```bash
# Expand a query with technical synonyms
ai-mem expand-query "auth login"
# → auth, authentication, authorization, login, signin, oauth...

# Detect intent from a prompt
ai-mem detect-intent "Fix the authentication bug in login.py"
```

### Inline Context Tags

Embed memory queries directly in prompts using `<mem>` tags:

```bash
# Parse tags in a prompt
ai-mem parse-tags 'Help with <mem query="auth" limit="5"/> implementation'

# Expand tags with actual memory content
ai-mem expand-tags 'Explain <mem query="config" mode="compact"/>'
```

### Token Budget & Context Preview

Monitor and optimize token usage:

```bash
# Preview context without generating it
ai-mem context --preview --model claude-3-opus

# Check token budget warnings
ai-mem context --query "auth" --model gpt-4
```

### Incremental Indexing

Only reindex new or modified observations:

```bash
# View indexing statistics
ai-mem index-stats

# Incrementally reindex
ai-mem reindex

# Force full reindex
ai-mem reindex --force
```

### Async Batch Embeddings

For bulk operations, use the async batch embedder:

```python
from ai_mem.embeddings import AsyncBatchEmbedder, BatchConfig

batch_embedder = AsyncBatchEmbedder(
    provider,
    config=BatchConfig(max_batch_size=32, max_concurrent_batches=4)
)
result = await batch_embedder.embed_batch(texts)
print(f"Embedded {result.total_texts} texts in {result.elapsed_seconds:.2f}s")
```

### Memory Consolidation & Decay

Keep your memory store lean by consolidating duplicates and removing stale entries:

```bash
# Find and consolidate similar observations
ai-mem consolidate --project /path/to/project --threshold 0.85

# Preview what would be consolidated (dry run)
ai-mem consolidate -p myproject --dry-run

# Keep highest importance observations instead of newest
ai-mem consolidate -p myproject --strategy highest_importance

# Clean up old, rarely-accessed memories (decay)
ai-mem cleanup-stale --project myproject --max-age 90 --min-access 0

# Find similar observations without consolidating
ai-mem find-similar --project myproject --threshold 0.8
```

Strategies: `newest` (default), `oldest`, `highest_importance`.

### Two-Stage Retrieval

Improve search precision with configurable reranking:

```bash
# Search with automatic two-stage retrieval
ai-mem search "authentication flow" --rerank

# Configure reranker type in config
# Options: "biencoder" (default), "tfidf" (fast), "crossencoder" (most accurate)
```

Configuration in `~/.config/ai-mem/config.json`:
```json
{
  "search": {
    "enable_reranking": true,
    "reranker_type": "biencoder",
    "stage1_candidates": 50,
    "rerank_weight": 0.4
  }
}
```

### Progressive Disclosure (3-Layer Context)

Optimize token usage with layered context disclosure:

```bash
# Layer 1: Compact index only (~50-100 tokens)
ai-mem context --query "auth" --mode compact

# Layer 2: Timeline context around results
ai-mem timeline --query "auth" --depth-before 3 --depth-after 3

# Layer 3: Full details on demand
ai-mem get <observation-id>
```

The 3-layer workflow enables ~10x token savings:
- **Layer 1**: Quick index with IDs and summaries
- **Layer 2**: Chronological context around anchor observation
- **Layer 3**: Full content only when needed

### Host Configuration

Configure behavior per LLM host:

```bash
# Set default host
export AI_MEM_DEFAULT_HOST=claude-code

# Override host-specific settings
export AI_MEM_HOST_CLAUDE_CODE_INJECTION_METHOD=mcp_tools
export AI_MEM_HOST_GEMINI_FORMAT=markdown
```

Configuration in `~/.config/ai-mem/config.json`:
```json
{
  "hosts": {
    "default_host": "generic",
    "hosts": {
      "claude-code": {
        "injection_method": "mcp_tools",
        "supports_mcp": true,
        "context_position": "system_prompt",
        "format": "markdown",
        "progressive_disclosure": true
      },
      "gemini": {
        "injection_method": "prompt_prefix",
        "supports_mcp": false,
        "context_position": "user_prompt_start",
        "format": "markdown"
      }
    }
  }
}
```

### AI Compression

Compress observations to reduce token usage:

```python
from ai_mem.compression import CompressionService

# Heuristic compression (no LLM required)
service = CompressionService()
result = await service.compress(text, target_ratio=4.0)
print(f"Compressed {result.original_tokens} → {result.compressed_tokens} tokens")

# With LLM for semantic compression
service = CompressionService(provider=chat_provider)
result = await service.compress(text, context_type="code_context")
```

Context types: `default`, `code_context`, `tool_output`, `conversation`.

## Multi-LLM Architecture

ai-mem is designed from the ground up to support **multiple LLMs simultaneously**. Whether you're using Claude Code in one terminal, Gemini CLI in another, and Cursor in your IDE, all observations flow into the same shared memory store.

### Event Schema v1

All events are normalized to a canonical format before storage:

```python
from ai_mem.events import ToolEvent, EventSource, ToolExecution

event = ToolEvent(
    event_id="unique-uuid",           # For idempotency
    session_id="session-123",
    source=EventSource(host="gemini"), # Host identifier
    tool=ToolExecution(
        name="Read",
        input={"path": "/src/main.py"},
        output="file contents...",
        success=True,
    ),
)
```

This schema is **LLM-agnostic** - no Claude or Gemini specific fields. Any host can generate events in this format.

### Host Adapters

Adapters translate host-specific payloads to the unified Event Schema:

| Adapter | Host | Use Case |
|---------|------|----------|
| `ClaudeAdapter` | `claude-code`, `claude-desktop` | Claude Code CLI, Claude Desktop app |
| `GeminiAdapter` | `gemini`, `gemini-cli` | Gemini CLI, Gemini API integrations |
| `GenericAdapter` | Any other | Cursor, VS Code, custom integrations |

**Usage via API:**

```bash
# Send a tool event from any host
curl -X POST http://localhost:37777/api/events \
  -H "Content-Type: application/json" \
  -d '{
    "host": "gemini",
    "payload": {
      "tool_name": "Read",
      "tool_input": {"path": "/src/main.py"},
      "tool_response": "file contents..."
    }
  }'
```

**Usage via CLI:**

```bash
# The --host flag identifies the source
ai-mem add "Tool: Read\nInput: /src/main.py" --host gemini --event-id "uuid-123"
```

**Programmatic usage:**

```python
from ai_mem.adapters import get_adapter

# Auto-select adapter based on host
adapter = get_adapter("gemini")
event = adapter.parse_tool_event({
    "tool_name": "Bash",
    "tool_input": {"command": "ls -la"},
    "tool_response": "file list...",
})

# Event is now in canonical ToolEvent format
print(event.tool.name)  # "Bash"
print(event.source.host)  # "gemini"
```

### Idempotency

Each event has an `event_id` that prevents duplicate observations:

```bash
# First call - creates observation
ai-mem add "test" --event-id "abc-123" --host claude-code
# Memory stored! (ID: obs_xyz)

# Second call with same event_id - returns existing observation
ai-mem add "test" --event-id "abc-123" --host claude-code
# Memory stored! (ID: obs_xyz)  # Same ID, no duplicate
```

This is critical for hooks that may be retried on network failures.

### Environment Variables

Configure host behavior via environment variables (see `.env.example`):

```bash
# Host identification
AI_MEM_HOST=claude-code           # Default host identifier

# Ingestion filtering
AI_MEM_SKIP_TOOL_NAMES=TodoWrite,SlashCommand,Skill
AI_MEM_SKIP_TOOL_PREFIXES=mcp__
AI_MEM_MAX_OUTPUT_CHARS=50000
AI_MEM_IGNORE_FAILED_TOOLS=false

# Redaction patterns (comma-separated regex)
AI_MEM_REDACTION_PATTERNS=(?i)api[_-]?key.*,sk-[a-zA-Z0-9]{20,}
```

## Cross-model handoff

1. Start the server/panel stack so the MCP tools, REST API, and UI feed a shared memory graph:

   ```bash
   ./scripts/run-all.sh
   ```

2. Keep a live stream via `ai-mem endless` to continuously refresh context; the same `scoreboard` and cache entries that appear in the UI are printed on every iteration so assistants can explain why memories were selected:

   ```bash
   ai-mem endless --query "next feature" --interval 30 --token-limit 1000
   ```

3. Point Claude Desktop, Gemini CLI, Antigravity, or any custom vLLM at the MCP endpoint—each client consumes the same `<ai-mem-context>` blocks, scoreboard, and citations so your investigations stay synchronized.

4. Export and merge checkpoints when you want to hand off context between machines or sessions:

   ```bash
   ai-mem snapshot export /tmp/ai-mem-checkpoint.ndjson
   ai-mem snapshot merge /tmp/ai-mem-checkpoint.ndjson
   ```

This workflow keeps Claude, Gemini, and other assistants aligned with the same persistent history, token budgets, and metadata traceability across every interaction.

## Documentation

See the `docs/` folder for targeted guides:

- [🚀 Getting Started](docs/getting-started.md) – Installation, configuration, quick-start, and baseline workflows.
- [⚙️ Configuration](docs/configuration.md) – Vector stores, cache policies, connectors, and token budget controls.
- [🔌 Proxies & Hooks](docs/proxies.md) – OpenAI, Gemini, Anthropic, Azure, Bedrock proxies plus IDE integrations.
- [🛠️ MCP Tools & Integrations](docs/mcp-tools.md) – MCP Server, Claude plugins, Antigravity, VS Code, cursor hooks, and how to consume context over MCP.
- [🎣 Hooks](docs/hooks.md) – Lifecycle hooks for command-line, IDE, and MCP-based ingestion.
- [🧩 Presets](docs/presets.md) – Scripts for installing Claude mem-search skills, VS Code tasks, JetBrains tools, and Antigravity/VS integrations so everything wires into `ai-mem hook ...`.
- [📖 API Reference](docs/api-reference.md) – REST routes, observation schema, scoreboard payloads.
- [🏛️ Architecture](docs/architecture.md) – Component diagrams, data flow, and search strategy.
- [∞ Endless Mode](docs/endless_mode.md) – Continuous context injection and scoreboard monitoring.
- [📸 Snapshots](docs/snapshots.md) – Backup, export, and merging strategies.
- [💻 Development](docs/development.md) – Testing, roadmap, contributor guide, and venv tips.

## Credits & inspiration

Inspired by [claude-mem](https://github.com/thedotmack/claude-mem). We highlight the shared-memory story, CLI helpers, and documentation so any LLM can plug in and tap into persistent context.

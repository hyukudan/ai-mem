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
- **Shared metadata & scoreboard** – Responses include metadata that enumerates vector vs. FTS scores, recency, and cache hits/misses, letting you introspect model prompts before each completion.
- **Endless Mode** – `ai-mem endless` polls the store at a configurable interval to keep total tokens within limits. See [Endless Mode Guide](docs/endless_mode.md).
- **Snapshot-based syncing** – `ai-mem snapshot export`/`import`/`merge` round-trips checkpoints. See [Snapshots Guide](docs/snapshots.md).
- **Hooks & IDE scripts** – Run `./scripts/install-hooks.sh`, `./scripts/install-vscode-tasks.sh`, `./scripts/install-jetbrains-tools.sh`, or the Antigravity/Claude installers to make every hook call `ai-mem hook ...` automatically.
- **Web viewer + MCP server** – Live stream + scoreboard at `http://localhost:37777`, plus REST endpoints that expose observations, scorecards, and citations for external agents like Gemini, Claude, or Antigravity.
- **CLI-first control** – All key flows (add/search/context/timeline/endless/snapshot) are available via `ai-mem` so you can script onboarding via a single CLI.
- **Scripts library** – `./scripts/run*.sh` cover full/stack/proxy setups for Gemini, Claude, Bedrock, Azure, and Anthropic deployments, ensuring the UI + MCP server is always upstream.

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

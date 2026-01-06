# ai-mem: Universal Long-Term Memory for LLMs

ai-mem is a local-first memory layer for any LLM. It stores observations in a SQLite + FTS5 database, adds semantic search via a local vector store (Chroma), and injects the right context when you start a new task. It works with Gemini, OpenAI-compatible APIs (vLLM/LM Studio/Ollama), and any client that can call hooks or an MCP tool.

This project is inspired by claude-mem (https://github.com/thedotmack/claude-mem). We borrow ideas and adapt them for a multi-provider, local-first workflow. See Credits and Attribution for details.

License: AGPL-3.0

## Table of Contents

- Overview
- Key Features
- Core Concepts
- Architecture
- Installation
- Quick Start
- Convenience Scripts
- Configuration
- Using the CLI
- Sessions
- Context Injection
- Web Viewer
- Hooks (Model Agnostic)
- Claude Code Plugin
- MCP Tools (Claude Desktop and others)
- Proxies (OpenAI-compatible and Gemini)
- API Token (Optional)
- REST API
- Storage Layout
- Privacy and Redaction
- Troubleshooting
- Roadmap
- Development and Testing
- Credits and Attribution

## Overview

LLM chats forget context when you start a new session. ai-mem acts as a memory layer that:

1. Captures observations (prompts, outputs, notes, tool logs).
2. Summarizes long content to keep memory compact.
3. Stores knowledge locally (SQLite + FTS5 + Chroma).
4. Retrieves relevant context on demand (search -> timeline -> full details).
5. Injects that context into any LLM client.

You can use ai-mem entirely locally with fast embeddings, or connect it to Gemini or any OpenAI-compatible model. It is designed to work in terminals, IDEs, or agent frameworks.

## Key Features

- Model agnostic: Gemini native + OpenAI-compatible local models.
- Local and private: SQLite + FTS5 + ChromaDB stored on disk.
- Semantic + keyword search: hybrid retrieval for relevance.
- Progressive disclosure: search -> timeline -> full detail to control token cost.
- Web viewer UI: browse, search, and manage memory at http://localhost:8000.
- Sessions: track goals and scope retrieval to a session.
- Privacy tags: <private>...</private> is stripped before storage.
- Tag management: add, edit, rename, and delete tags.
- Context injection: generate <ai-mem-context> blocks for any model.
- Hooks and proxies: automatic storage and injection.
- MCP tools: search memory from Claude Desktop or other MCP clients.
- Citations: reference observations via /api/observation/{id}.

## Core Concepts

- Observation: a single memory item. It has an id, session_id, project, type, content, summary, tags, metadata, and timestamps.
- Project: a folder path used to group observations.
- Session: a scoped work unit with a goal and timestamps. Sessions help you filter memory for a task.
- Progressive disclosure:
  - search: compact summaries
  - timeline: surrounding context
  - get: full details

Observation types are flexible, but common values include:
- note, decision, bugfix, feature, refactor, discovery, change, interaction, tool_output, file_content, summary

## Architecture

ai-mem uses a layered retrieval pipeline:

1. SQLite + FTS5 for keyword search and fast filtering.
2. Chroma vector store for semantic search over chunks.
3. Progressive disclosure to control context size.

Core components:

- MemoryManager: orchestrates storage, retrieval, and session tracking.
- SQLite DB: observations and sessions.
- Vector store: embeddings and semantic search.
- Context builder: formats <ai-mem-context> blocks with token estimates.
- Web server: REST API + viewer UI.
- Proxies: OpenAI-compatible proxy and Gemini proxy.
- MCP server: tools for external clients.
- Hook scripts: for any lifecycle-based client.

Architecture diagram (simplified):

```
                     +-----------------------+
                     |  CLI / UI / REST API  |
                     +-----------+-----------+
                                 |
                                 v
                     +-----------------------+
   Hooks / MCP <---->|  MemoryManager + RAG  |<---- Proxies (OpenAI/Gemini)
                     +-----------+-----------+
                                 |
                    +------------+------------+
                    |                         |
                    v                         v
             +--------------+         +---------------+
             | SQLite + FTS |         | Chroma Vector |
             | observations |         | embeddings    |
             +--------------+         +---------------+
```

## Installation

Requirements:
- Python 3.10+
- pip, venv

Clone and install:

```bash
git clone https://github.com/yourusername/ai-mem.git
cd ai-mem
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or use the bootstrap helper:

```bash
./scripts/bootstrap.sh
```

## Quick Start

CLI only (no API keys required):

```bash
ai-mem add "We use Python 3.11 and pandas 2.0 in Omega."
ai-mem search "Omega dependencies"
```

Start the server and UI:

```bash
./scripts/run.sh
```

Open http://localhost:8000

## Convenience Scripts

Scripts in scripts/ start common stacks:

- ./scripts/run.sh: server + UI only
- ./scripts/run-all.sh: server + UI + MCP
- ./scripts/run-stack.sh: server + OpenAI-compatible proxy
- ./scripts/run-full.sh: server + OpenAI-compatible proxy + MCP
- ./scripts/run-gemini-stack.sh: server + Gemini proxy
- ./scripts/run-gemini-full.sh: server + Gemini proxy + MCP
- ./scripts/run-dual-stack.sh: server + OpenAI-compatible proxy + Gemini proxy
- ./scripts/run-dual-full.sh: server + OpenAI-compatible proxy + Gemini proxy + MCP
- ./scripts/run-proxy.sh: OpenAI-compatible proxy only
- ./scripts/run-gemini-proxy.sh: Gemini proxy only
- ./scripts/run-mcp.sh: MCP server only

## Configuration

Config is stored in:
- ~/.config/ai-mem/config.json
- Override with AI_MEM_CONFIG=/path/to/config.json

Show current config:

```bash
ai-mem config --show
```

Configure LLM and embeddings:

```bash
# Gemini + local embeddings
ai-mem config --llm-provider gemini --llm-model gemini-1.5-flash --llm-api-key YOUR_KEY
ai-mem config --embeddings-provider fastembed

# OpenAI-compatible model (vLLM, LM Studio, Ollama)
ai-mem config --llm-provider openai-compatible --llm-base-url http://localhost:8000/v1 --llm-model YOUR_MODEL

# If your OpenAI-compatible server exposes embeddings
ai-mem config --embeddings-provider openai-compatible --embeddings-base-url http://localhost:8000/v1 --embeddings-model YOUR_EMBED_MODEL

# Auto embeddings: use OpenAI-compatible if configured, else fastembed
ai-mem config --embeddings-provider auto
```

Context defaults:

```bash
ai-mem config --context-total 12 --context-full 4 --context-show-tokens
```

Environment overrides (context only):

```bash
export AI_MEM_CONTEXT_TOTAL=12
export AI_MEM_CONTEXT_FULL=4
export AI_MEM_CONTEXT_TYPES=note,bugfix
export AI_MEM_CONTEXT_TAGS=auth,infra
export AI_MEM_CONTEXT_FULL_FIELD=content
export AI_MEM_CONTEXT_SHOW_TOKENS=true
export AI_MEM_CONTEXT_WRAP=true
```

Date filters accept epoch, ISO-8601, or relative durations like 7d, 24h, 30m.

## Using the CLI

Common commands:

```bash
# Add
ai-mem add "We decided to use Redis for session cache." --obs-type decision --tag architecture

# Search
ai-mem search "Redis session cache" --limit 10
ai-mem mem-search "Redis session cache" --limit 10

# Timeline
ai-mem timeline --query "Redis" --depth-before 3 --depth-after 3

# Full observation details
ai-mem get <observation_id>

# Update tags
ai-mem update-tags <observation_id> --tag infra --tag followup
ai-mem update-tags <observation_id> --clear

# Stats
ai-mem stats --tag infra

# Tags
ai-mem tags
ai-mem tag-add triage --filter-tag bug
ai-mem tag-rename infra infrastructure
ai-mem tag-delete legacy --force
ai-mem tag-rename infra infrastructure --filter-tag important

# Export / import
ai-mem export memories.json
ai-mem import memories.json
ai-mem export memories.jsonl --format jsonl
ai-mem import memories.jsonl
ai-mem export memories.csv --format csv
ai-mem import memories.csv
ai-mem export memories.json --tag bug --date-start 2024-01-01
ai-mem export memories.json --since 24h

# Watch a command
ai-mem watch --command "pytest -q"

# Ingest a codebase
ai-mem ingest .
```

## Sessions

Sessions let you scope memory to a task.

```bash
# Start / end
ai-mem session-start --goal "Fix OAuth flow"
ai-mem sessions --active-only
ai-mem session-end --id <session_id>

# Add directly to a session
ai-mem add "Session-specific note" --session-id <session_id>

# Search within a session
ai-mem search "OAuth" --session-id <session_id>
```

Session summaries:

```bash
ai-mem summarize --session-id <session_id> --count 50
```
Stored session summaries also update the session summary field (visible in the UI).

## Context Injection

Generate a formatted context block for any client:

```bash
# Default project (cwd)
ai-mem context

# With a query
ai-mem context --query "OAuth flow" --tag auth --tag bug

# Session scope
ai-mem context --session-id <session_id>
```

By default ai-mem wraps output in <ai-mem-context> to prevent recursive ingestion.
Use --no-wrap if you need raw text.

## Web Viewer

Launch the server UI:

```bash
ai-mem server
```

Viewer features:
- Search, timeline, stats, and context preview.
- Session start/end, summaries, and export.
- Export JSON/JSONL/CSV and import JSON/JSONL/CSV from the viewer.
- Export since (relative) for incremental snapshots.
- Stream view for new observations.
- Session ID filter for search, timeline, stats, context, and stream.
- Observation details include a Copy URL button for citations.
- Observation detail view lets you edit tags.
- Click tags in the detail view to filter (shift-click to add).
- Search and timeline results show a rough token estimate per summary.
- Session list previews the latest session summary when available.
- Session detail view shows totals, top types, and top tags.
- Save and re-apply search filters per project or globally (stored locally in the browser).
- Tag management panel to add, list, rename, and delete tags.
- Context presets per project in the viewer UI.

## Hooks (Model Agnostic)

Use scripts in scripts/hooks with any client that supports shell hooks.

Examples:

```bash
# Session start: print context
AI_MEM_PROJECT="$PWD" scripts/hooks/session_start.sh

# Store a user prompt
AI_MEM_PROJECT="$PWD" AI_MEM_CONTENT="Fix OAuth flow" scripts/hooks/user_prompt.sh
```

Hook environment variables (common):
- AI_MEM_BIN: path to ai-mem binary
- AI_MEM_PROJECT: project path
- AI_MEM_SESSION_ID: scope context + storage to a specific session
- AI_MEM_CONTENT: content to store
- AI_MEM_TAGS: comma-separated tags
- AI_MEM_OBS_TYPE: override type
- AI_MEM_NO_SUMMARY: disable summarization
- AI_MEM_SESSION_TRACKING: session_start opens, session_end closes
- AI_MEM_SUMMARY_ON_END: run summarize after session_end
- AI_MEM_SUMMARY_COUNT: number of observations to summarize (default 20)
- AI_MEM_SUMMARY_OBS_TYPE: filter observation type for summaries

## Claude Code Plugin

A local plugin is available in plugin/.

```bash
./scripts/install-claude-plugin.sh
```

Set AI_MEM_BIN for venvs:

```bash
export AI_MEM_BIN="$PWD/.venv/bin/ai-mem"
```

See plugin/README.md for details.

## MCP Tools (Claude Desktop and others)

Start MCP stdio server:

```bash
ai-mem mcp
```

Tools:
- search
- mem-search (alias)
- timeline
- get_observations
- summarize
- tags
- tag-add
- tag-rename
- tag-delete

Search and timeline accept session_id to scope results.

## Proxies

### OpenAI-compatible proxy

```bash
AI_MEM_PROXY_UPSTREAM_BASE_URL="http://localhost:8000" ai-mem proxy --port 8081
```

Point your client to http://localhost:8081/v1

Supported endpoints:
- /v1/chat/completions
- /v1/completions
- /v1/responses

Headers:
- x-ai-mem-project: override project
- x-ai-mem-session-id: scope context + storage to a session
- x-ai-mem-query: override query for context
- x-ai-mem-inject: true/false
- x-ai-mem-store: true/false
- x-ai-mem-obs-type
- x-ai-mem-obs-types
- x-ai-mem-tags
- x-ai-mem-total
- x-ai-mem-full
- x-ai-mem-full-field
- x-ai-mem-show-tokens
- x-ai-mem-wrap

### Gemini proxy

```bash
AI_MEM_GEMINI_API_KEY="YOUR_KEY" ai-mem gemini-proxy --port 8090
```

Point your Gemini client to http://localhost:8090

Supported endpoints:
- :generateContent
- :streamGenerateContent

## API Token (Optional)

Set AI_MEM_API_TOKEN to require a bearer token on all API routes (including the UI).
The viewer includes a token field that stores the value in localStorage.

```bash
AI_MEM_API_TOKEN=your-token ./scripts/run.sh
```

## REST API

When you run ai-mem server, see docs at:
- http://localhost:8000/docs

Key endpoints:
- POST /api/memories: add a memory (project, session_id, obs_type, tags, metadata, title, summarize)
- GET /api/search: search (project, session_id, obs_type, tags, date_start, date_end)
- GET /api/timeline: timeline (project, session_id, obs_type, tags, date_start, date_end, depth_before, depth_after)
- GET /api/observations: list observations (project, session_id, obs_type, tags, date_start, date_end, since, limit)
- GET /api/observations/{id}: single observation
- GET /api/observation/{id}: alias
- PATCH /api/observations/{id}: update tags (tags)
- DELETE /api/observations/{id}
- GET /api/projects
- GET /api/sessions (project, active_only, goal, date_start, date_end, limit)
- GET /api/sessions/{id}
- GET /api/sessions/{id}/observations
- POST /api/sessions/start (project, goal, session_id)
- POST /api/sessions/end (session_id or latest)
- GET /api/stats (project, session_id, obs_type, tags, date_start, date_end, tag_limit, day_limit, type_tag_limit)
- GET /api/tags (project, session_id, obs_type, tags, date_start, date_end, limit)
- POST /api/tags/add (tag, project, session_id, obs_type, tags, date_start, date_end)
- POST /api/tags/rename (old_tag, new_tag, project, session_id, obs_type, tags, date_start, date_end)
- POST /api/tags/delete (tag, project, session_id, obs_type, tags, date_start, date_end)
- GET /api/context/preview (project, session_id, query, obs_type, obs_types, tags, total, full, full_field, show_tokens, wrap)
- GET /api/context/inject (same as preview)
- GET /api/context (alias)
- GET /api/context/config
- POST /api/context/preview
- POST /api/context/inject
- GET /api/stream (project, session_id, obs_type, tags, query, token)
- GET /api/export (project, session_id, obs_type, tags, date_start, date_end, since, limit, format=json|jsonl|csv)
- POST /api/import (preserves session_id unless project override changes scope)
- POST /api/summarize (project, session_id, count, obs_type, store, tags)
- GET /api/health
- GET /api/readiness
- GET /api/version

## Storage Layout

Default data directory: ~/.ai-mem

- SQLite: ~/.ai-mem/ai-mem.sqlite
- Vector DB: ~/.ai-mem/vector-db

You can override these in config:

```bash
ai-mem config --data-dir /path/to/data
ai-mem config --sqlite-path /path/to/ai-mem.sqlite
ai-mem config --vector-dir /path/to/vector-db
```

## Privacy and Redaction

Use <private>...</private> to prevent sensitive content from being stored.
The private segments are removed before hashing, indexing, and storage.

<ai-mem-context> is reserved and stripped to prevent recursive ingestion.

## Troubleshooting

- Proxy fails to start: set AI_MEM_PROXY_UPSTREAM_BASE_URL.
- Gemini proxy warns about API key: set AI_MEM_GEMINI_API_KEY or GOOGLE_API_KEY.
- No results: check project path, session_id, or tags.
- Token estimates are rough: they are based on character count heuristics.
- Port in use: change AI_MEM_PORT, AI_MEM_PROXY_PORT, or AI_MEM_GEMINI_PROXY_PORT.

## Roadmap

Planned or proposed improvements:

- Provider adapters: Anthropic, Azure OpenAI, and additional local runtimes.
- Pluggable vector stores (pgvector, Qdrant) and faster hybrid search.
- Richer observations: attachments, file diffs, structured metadata.
- Incremental sync/export formats (JSONL, snapshots, merge tools).
- UI enhancements: session analytics, tag management, saved filters.
- Hook presets for more clients and IDEs.

## Development and Testing

Run tests:

```bash
.venv/bin/python -m pytest tests/test_db.py tests/test_summary.py tests/test_privacy.py tests/test_chunking.py tests/test_dates.py
```

## Credits and Attribution

ai-mem is inspired by claude-mem (https://github.com/thedotmack/claude-mem).
We adopt similar memory concepts and extend them with multi-provider support and local-first defaults.

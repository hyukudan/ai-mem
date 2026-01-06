# ai-mem: Universal Long-Term Memory for LLMs

**ai-mem** is an open-source tool designed to give **Long-Term Memory** to any Large Language Model (LLM), including Google Gemini, OpenAI-compatible local models (vLLM/LM Studio/Ollama), and more.

Inspired by [claude-mem](https://github.com/thedotmack/claude-mem), this project generalizes the concept of persistent context, allowing you to maintain knowledge across sessions, projects, and different AI providers.

![License](https://img.shields.io/badge/license-AGPL%203.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)

## 🚀 Why ai-mem?

Current LLM interfaces have a limited context window. Once you close a chat or start a new session, the model "forgets" everything you taught it about your coding style, project architecture, or specific requirements.

**ai-mem** solves this by acting as an external **Memory Layer** that:
1.  **Captures** important interactions, facts, and decisions.
2.  **Compresses** verbose logs into concise insights using an AI summarizer.
3.  **Stores** knowledge in a local Vector Database (ChromaDB) with project-level isolation.
4.  **Retrieves** the most relevant context automatically via Semantic Search (RAG) when you start a new task.

## ✨ Key Features

*   **🤖 Model Agnostic:** Gemini native + OpenAI-compatible local models out of the box.
*   **🔒 Local & Private:** Your data stays on your machine (SQLite + FTS5 + ChromaDB).
*   **🧩 Local Embeddings:** Fast, offline embeddings via `fastembed` by default.
*   **📂 Project Scoped:** Automatically detects your current project directory to keep memories organized.
*   **🕸️ Web Viewer UI:** A built-in dashboard to visualize, search, and manage your memory stream in real-time.
*   **🔌 API Server:** Exposes a REST API so other agents, scripts, or IDE extensions can query memory programmatically.
*   **🧠 Semantic Search:** Finds information by *meaning*, not just keywords.
*   **📉 Cost & Token Management:** Summarizes old memories to save on context window usage.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-mem.git
cd ai-mem

# Install dependencies (global)
pip install -e .
```

## 🧪 Virtualenv (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or use the bootstrap script:

```bash
./scripts/bootstrap.sh
```

Start API + UI (single process):

```bash
./scripts/run.sh
```

Start API + UI alongside the MCP stdio server:

```bash
./scripts/run-all.sh
```

## ⚡ Quick Start

### 1. Zero-config (CLI only)
By default, `ai-mem` runs with **local embeddings** and **no LLM** required. This lets you use the CLI without API keys.

```bash
ai-mem add "We use Python 3.11 and pandas 2.0 in Omega."
ai-mem search "Omega dependencies"
```

### 2. Configuration (Optional)
Configure an LLM provider and embeddings. Embeddings are local (`fastembed`) unless you change them.

```bash
# Gemini (LLM) + local embeddings
ai-mem config --llm-provider gemini --llm-model gemini-1.5-flash --llm-api-key YOUR_GEMINI_API_KEY
ai-mem config --embeddings-provider fastembed

# Local OpenAI-compatible LLM (vLLM/LM Studio/Ollama) + local embeddings
ai-mem config --llm-provider openai-compatible --llm-base-url http://localhost:8000/v1 --llm-model YOUR_MODEL_NAME
ai-mem config --embeddings-provider fastembed

# If your OpenAI-compatible server exposes embeddings
ai-mem config --embeddings-provider openai-compatible --embeddings-base-url http://localhost:8000/v1 --embeddings-model YOUR_EMBED_MODEL

# Auto: use OpenAI-compatible embeddings when configured, otherwise local fastembed
ai-mem config --embeddings-provider auto
```

Example vLLM server:

```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8000
```

### 3. Adding Memories
You can manually add knowledge or pipe output from other commands.

```bash
# Manual addition
ai-mem add "The project 'Omega' uses Python 3.11 and requires pandas 2.0. Avoid using numpy directly."

# Ingest a codebase
ai-mem ingest .
```

### 4. Retrieving Context
Before asking your LLM a question, fetch the relevant context.

```bash
# Compact index (Layer 1)
ai-mem search "How do I set up the dependencies for Omega?"

# Timeline context (Layer 2)
ai-mem timeline --query "dependencies for Omega"

# Full details (Layer 3)
ai-mem get <observation_id>
```

### 5. Watch Mode (CLI)
Capture command output or a growing log file.

```bash
ai-mem watch --command "pytest -q"
ai-mem watch --file /var/log/system.log
```

### 6. Export / Import
Move observations between machines or back up your memory store.

```bash
ai-mem export memories.json
ai-mem import memories.json
```

### 7. Cleanup
Remove single observations or entire projects.

```bash
ai-mem delete <observation_id>
ai-mem delete-project /path/to/project
```

### 4. The Web Viewer
Launch the local server to browse your memories visually.

```bash
ai-mem server
```
> Open `http://localhost:8000` in your browser to see the Memory Stream.

Viewer tips:
- Query input supports `Esc` to clear, plus a clear button.
- Toggle "Use global query" to share the same query across projects.
- Timeline mode shows an anchor badge and an Exit button in the header/sidebar.
- Auto-refresh settings (on/off, interval, mode) are stored per project.
- "Use global auto-refresh" lets you share auto-refresh settings across projects.
- The sidebar auto-refresh label shows mode/interval/scope (tooltip) and highlights global scope.
- The results header auto-refresh badge shows a dot when scope is global.

## 🛠️ Architecture

**ai-mem** follows a layered retrieval pattern:

1.  **SQLite + FTS5:** Fast keyword search for observations, summaries, and tags.
2.  **ChromaDB:** Semantic retrieval via local embeddings.
3.  **Progressive Disclosure:** `search` → `timeline` → `get` keeps token usage low.

## 🤖 Automation & Integration

### Using with Scripts
You can use `ai-mem` in your CI/CD pipelines or local scripts to inject context automatically.

```bash
# Example: Injecting memory into a CLI LLM tool
CONTEXT=$(ai-mem get "Fixing the login bug" --format json)
llm-tool --system "$CONTEXT" "Help me fix the login bug in auth.py"
```

### REST API
When you run `ai-mem server`, a full REST API is available at `http://localhost:8000/docs`.

*   `POST /api/memories`: Add a new memory.
*   `GET /api/search?query=...`: Search the memory index.
*   `GET /api/timeline?query=...`: Timeline context around a query or anchor (supports `project`, `obs_type`, `date_start`, `date_end`, `depth_before`, `depth_after`).
*   `POST /api/observations`: Fetch full observation details.
*   `GET /api/observations`: List observations (export).
*   `DELETE /api/observations/{id}`: Delete observation.
*   `GET /api/projects`: List all tracked projects.
*   `GET /api/stats`: Summary counts by type/project/tags/days (supports `project`, `obs_type`, `date_start`, `date_end`, `tag_limit`, `day_limit`, `type_tag_limit`; returns `recent_total`, `previous_total`, `trend_delta`, `trend_pct`).
*   `POST /api/projects/delete`: Delete all observations for a project.
*   `GET /api/export`: Export observations.
*   `POST /api/import`: Import observations.
*   `GET /api/health`: Health check.

## 🔐 API Token (Optional)

Set `AI_MEM_API_TOKEN` to require a bearer token on all API routes (including the UI). The viewer includes a token field that stores the value in localStorage.

```bash
AI_MEM_API_TOKEN=your-token ./scripts/run.sh
```

Then set the same token in the viewer sidebar.

## 🔌 MCP Tools (Optional)

Start the MCP stdio server:

```bash
ai-mem mcp
```

Or use the helper script:

```bash
./scripts/run-mcp.sh
```

Available tools: `search`, `timeline`, `get_observations` plus `__IMPORTANT` workflow guidance.

## 🗺️ Roadmap

- [x] Core CLI (Config, Add, Search, Timeline, Get)
- [x] Gemini Provider Support
- [x] Local embeddings (fastembed)
- [x] OpenAI-compatible embeddings auto-detect
- [ ] VS Code Extension
- [x] "Watch Mode" (Auto-ingest terminal output)

## 🙌 Attribution

This project is inspired by and references architecture from [claude-mem](https://github.com/thedotmack/claude-mem). Portions of the design and concepts follow claude-mem's progressive disclosure and storage model. See their repo for the original implementation.

## 📄 License

AGPL-3.0 License.

# Getting Started

## Installation

Requirements:
- Python 3.10+
- pip, venv
- PostgreSQL 15+ with pgvector extension (only needed when using the pgvector provider)

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

## Quick Start details

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

Scripts in `scripts/` start common stacks:

- `./scripts/run.sh`: server + UI only
- `./scripts/run-all.sh`: server + UI + MCP
- `./scripts/run-stack.sh`: server + OpenAI-compatible proxy
- `./scripts/run-full.sh`: server + OpenAI-compatible proxy + MCP
- `./scripts/run-gemini-stack.sh`: server + Gemini proxy
- `./scripts/run-gemini-full.sh`: server + Gemini proxy + MCP
- `./scripts/run-anthropic-stack.sh`: server + Anthropic proxy
- `./scripts/run-anthropic-full.sh`: server + Anthropic proxy + MCP
- `./scripts/run-azure-stack.sh`: server + Azure OpenAI proxy
- `./scripts/run-azure-full.sh`: server + Azure OpenAI proxy + MCP
- `./scripts/run-bedrock-stack.sh`: server + Bedrock proxy
- `./scripts/run-bedrock-full.sh`: server + Bedrock proxy + MCP
- `./scripts/run-dual-stack.sh`: server + OpenAI-compatible proxy + Gemini proxy
- `./scripts/run-dual-full.sh`: server + OpenAI-compatible proxy + Gemini proxy + MCP
- `./scripts/run-proxy.sh`: OpenAI-compatible proxy only
- `./scripts/run-gemini-proxy.sh`: Gemini proxy only
- `./scripts/run-anthropic-proxy.sh`: Anthropic proxy only
- `./scripts/run-azure-proxy.sh`: Azure OpenAI proxy only
- `./scripts/run-bedrock-proxy.sh`: Bedrock proxy only
- `./scripts/run-mcp.sh`: MCP server only
- `./scripts/install-hooks.sh`: install hook scripts to `~/.config/ai-mem/hooks`
- `./scripts/install-mcp-claude-desktop.sh`: add ai-mem MCP entry to Claude Desktop config
- `./scripts/install-mcp-cursor.sh`: add ai-mem MCP entry to Cursor config
- `./scripts/install-vscode-tasks.sh`: add ai-mem tasks to VS Code
- `./scripts/install-jetbrains-tools.sh`: add ai-mem External Tools to JetBrains IDEs

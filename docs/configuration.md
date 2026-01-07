# Configuration

Config is stored in:
- `~/.config/ai-mem/config.json`
- Override with `AI_MEM_CONFIG=/path/to/config.json`

```mermaid
graph TD
    A[Defaults] --> B(Merge)
    C[Config File<br/>~/.config/ai-mem/config.json] --> B
    D[Env Vars<br/>AI_MEM_*] --> B
    B --> E[Run-time Config]
```

Show current config:

```bash
ai-mem config --show
```

## LLM and Embeddings

Configure LLM and embeddings:

```bash
# Gemini + local embeddings
ai-mem config --llm-provider gemini --llm-model gemini-1.5-flash --llm-api-key YOUR_KEY
ai-mem config --embeddings-provider fastembed

# Anthropic (Claude) summarization
ai-mem config --llm-provider anthropic --llm-model claude-3-haiku-20240307 --llm-api-key YOUR_KEY

# Azure OpenAI summarization
ai-mem config --llm-provider azure-openai --llm-model YOUR_DEPLOYMENT --llm-api-key YOUR_KEY --llm-base-url https://YOUR_RESOURCE.openai.azure.com --llm-api-version 2024-02-01

# AWS Bedrock summarization
ai-mem config --llm-provider bedrock --llm-model anthropic.claude-3-haiku-20240307-v1:0

# OpenAI-compatible model (vLLM, LM Studio, Ollama)
ai-mem config --llm-provider openai-compatible --llm-base-url http://localhost:8000/v1 --llm-model YOUR_MODEL

# If your OpenAI-compatible server exposes embeddings
ai-mem config --embeddings-provider openai-compatible --embeddings-base-url http://localhost:8000/v1 --embeddings-model YOUR_EMBED_MODEL

# Azure OpenAI embeddings
ai-mem config --embeddings-provider azure-openai --embeddings-model YOUR_EMBED_DEPLOYMENT --embeddings-api-key YOUR_KEY --embeddings-base-url https://YOUR_RESOURCE.openai.azure.com --embeddings-api-version 2024-02-01

# AWS Bedrock embeddings
ai-mem config --embeddings-provider bedrock --embeddings-model amazon.titan-embed-text-v1

# Auto embeddings: use OpenAI-compatible if configured, else fastembed
ai-mem config --embeddings-provider auto
```

## Context

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

## Vector Store

Configure the vector backend you want to use for semantic search:

```bash
ai-mem config --vector-provider chroma --vector-chroma-collection observations

ai-mem config \
  --vector-provider pgvector \
  --pgvector-dsn "postgresql://user:pass@localhost:5432/ai_mem" \
  --pgvector-table ai_mem_vectors \
  --pgvector-dimension 1536 \
  --pgvector-index-type ivfflat
```

Environment overrides:

```bash
export AI_MEM_VECTOR_PROVIDER=pgvector
export AI_MEM_VECTOR_CHROMA_COLLECTION=observations
export AI_MEM_PGVECTOR_DSN="postgresql://user:pass@localhost:5432/ai_mem"
export AI_MEM_PGVECTOR_TABLE=ai_mem_vectors
export AI_MEM_PGVECTOR_DIMENSION=1536
export AI_MEM_PGVECTOR_INDEX_TYPE=ivfflat
export AI_MEM_PGVECTOR_LISTS=200
```

The pgvector provider requires Postgres with the `pgvector` extension enabled and a connection DSN. With pgvector you benefit from Postgres tooling (migrations, backups, monitoring) and can join vector hits with other relational metadata or reuse existing transaction semantics for ingestion.

Date filters accept epoch, ISO-8601, or relative durations like 7d, 24h, 30m.

## Search Caching

Avoid re-running identical queries by tuning the search cache.

```mermaid
graph LR
    A[Query] --> B{In Cache?}
    B -- Yes --> C[Return Cached Result<br/>(Hit)]
    B -- No --> D[Perform Hybrid Search]
    D --> E[Store in Cache]
    E --> F[Return Result<br/>(Miss)]
```

Set `search.cache_ttl_seconds` (default 30s) and `search.cache_max_entries` (default 256) in `~/.config/ai-mem/config.json` or via the environment:

```bash
export AI_MEM_SEARCH_CACHE_TTL=20
export AI_MEM_SEARCH_CACHE_ENTRIES=200
```

Adjust how the hybrid search ranks results:

```bash
export AI_MEM_SEARCH_FTS_WEIGHT=0.6
export AI_MEM_SEARCH_VECTOR_WEIGHT=0.4
export AI_MEM_SEARCH_RECENCY_HALFLIFE_HOURS=24
export AI_MEM_SEARCH_RECENCY_WEIGHT=0.2
```

Lower `fts_weight` in favor of `vector_weight` to boost embeddings, or raise `recency_weight`/lower the half-life to favor newer observations. Setting both weights to `0` falls back to taking the larger of FTS or vector score.

Cache hits are surfaced in the web viewer and by the `X-AI-MEM-Search-Cache` header (`hit`/`miss`). Setting TTL to `0` disables the cache.

The viewer’s stats panel also reports cache hits/misses, TTL, and number of cached entries so you can watch reuse patterns while adjusting the settings.

## Ingestion Filtering

Control what tool executions get stored in memory. This reduces noise and protects sensitive data.

```mermaid
graph LR
    A[Tool Event] --> B{Skip List?}
    B -- Yes --> C[Discard]
    B -- No --> D{Failed?}
    D -- Yes & ignore_failed --> C
    D -- No --> E{Min Output?}
    E -- Too Short --> C
    E -- OK --> F[Redact Secrets]
    F --> G[Truncate]
    G --> H[Store]
```

### Skip List (filter noise)

Tools in the skip list are never stored:

```bash
# Default skip list
export AI_MEM_SKIP_TOOL_NAMES="SlashCommand,Skill,TodoWrite,TodoRead,AskFollowupQuestion,AttemptCompletion"

# Skip by prefix (e.g., all MCP tools)
export AI_MEM_SKIP_TOOL_PREFIXES="mcp__,_internal"

# Skip by category (if host provides category info)
export AI_MEM_SKIP_TOOL_CATEGORIES="meta,admin"
```

### Truncation (control context size)

Limit how much of tool input/output gets stored:

```bash
# Max characters to store (default 50000 output, 10000 input)
export AI_MEM_MAX_OUTPUT_CHARS=50000
export AI_MEM_MAX_INPUT_CHARS=10000

# Skip tools with very short output (noise filter)
export AI_MEM_MIN_OUTPUT_CHARS=50
```

### Failed Tools

Optionally skip tools that failed:

```bash
export AI_MEM_IGNORE_FAILED_TOOLS=true
```

### Redaction Patterns

Sensitive data matching these patterns is replaced with `[REDACTED]`:

- API keys (`api_key=...`, `apikey:...`)
- Authorization headers (`Authorization: Bearer ...`)
- OpenAI keys (`sk-...`)
- Google API keys (`AIza...`)

Add custom patterns in `config.json`:

```json
{
  "ingestion": {
    "redaction_patterns": [
      "(?i)api[_-]?key\\s*[=:]\\s*['\"]?[\\w\\-\\.]+",
      "sk-[a-zA-Z0-9]{20,}",
      "your-custom-pattern-here"
    ]
  }
}
```

### Default Tags

Tags added to all auto-ingested tool observations:

```bash
export AI_MEM_INGESTION_DEFAULT_TAGS="tool,auto-ingested"
```

### Host Identifier (LLM-agnostic)

Tag observations with the host/agent that generated them:

```bash
# Set in your hook environment
export AI_MEM_HOST=claude-code   # or gemini, cursor, custom
```

## API Token (Optional)

Set `AI_MEM_API_TOKEN` to require a bearer token on all API routes (including the UI).
The viewer includes a token field that stores the value in localStorage.

```bash
AI_MEM_API_TOKEN=your-token ./scripts/run.sh
```

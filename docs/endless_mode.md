# Endless Mode

Endless Mode is a powerful feature in **ai-mem** that allows for continuous, adaptive context injection into your LLM workflows. Instead of manually querying memory for every interaction, Endless Mode runs in the background (or in a separate terminal), constantly refreshing the "active context" based on a persistent query or your current activity.

## How it works

When you run `ai-mem endless`, the system:
1.  **Polls** the vector store and FTS (Full-Text Search) index at a set interval.
2.  **Retrieves** the most relevant memories based on your query.
3.  **Generates** a formatted context block (XML-wrapped or similar).
4.  **Prunes** older or less relevant items to stay strictly within a defined **token budget**.
5.  **Displays** a live "Scoreboard" showing why certain memories were picked (recency vs. relevance).

This is particularly useful for:
- **Long-running sessions**: Keeping your agent aware of shifting goals without overloading the context window.
- **Background monitoring**: Having a terminal window that always shows "what is relevant right now."
- **Cross-model synchronization**: Feeding the same live stream to multiple agents (Claude, Gemini, etc.) via MCP.

## Usage

The basic command structure is:

```bash
ai-mem endless [OPTIONS]
```

### Common Examples

**1. Track a specific topic:**
Keep the context focused on "database migration" and refresh every 30 seconds.

```bash
ai-mem endless --query "database migration schemas" --interval 30
```

**2. Strict token budgeting:**
Ensure the memory context never exceeds 1000 tokens, preserving space for your main prompt.

```bash
ai-mem endless --query "current project" --token-limit 1000
```

**3. Integration mode (Silent/Pipe-friendly):**
Run in a way that can be easily piped or read by other tools (if supported by your specific version).

```bash
ai-mem endless --query "refactoring" --json
```

## The Scoreboard

Each refresh cycle prints a "Scoreboard" to the console. This gives you transparency into the retrieval logic:

| Metric | Description |
| :--- | :--- |
| **FTS Score** | Relevance based on exact keyword matching (BM25/Sqlite FTS5). |
| **Vector Score** | Semantic similarity relevance from the vector store (Chroma). |
| **Recency** | Boost applied to newer memories (time-decay factor). |
| **Tokens** | The token count of the specific memory item. |

This visibility helps you debug *why* the model knows what it knows (or why it forgot something).

## Integration with MCP

When running the **MCP Server** (`ai-mem mcp`), "Endless Mode" concepts are applied automatically if the client requests continuous updates. However, running `ai-mem endless` manually in a terminal is the best way to *visualize* the shifting context that all your connected MCP clients can potentially access if they query the "current context" tool.

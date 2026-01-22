# Endless Mode

Endless Mode transforms O(N²) context scaling to O(N) by compressing tool outputs in real-time while preserving full outputs in archive memory.

## Overview

When working with AI agents, tool outputs can quickly consume the context window. A typical session might involve:
- Reading files (1000+ tokens each)
- Running commands (variable output)
- Multiple search results

Without compression, context grows exponentially. Endless Mode solves this by:
1. **Compressing tool outputs** to ~500 tokens each
2. **Archiving full outputs** to disk for later retrieval
3. **Maintaining semantic meaning** through AI or heuristic compression

## Benefits

- **~20x more tool uses** before context exhaustion (50 → 1000)
- **~95% token reduction** on large outputs
- **Full transcript preservation** in archive memory
- **O(N) scaling** instead of O(N²)

## Configuration

### Environment Variables

```bash
# Enable Endless Mode
AI_MEM_ENDLESS_ENABLED=true

# Target tokens per compressed observation
AI_MEM_ENDLESS_TARGET_TOKENS=500

# Compression ratio target
AI_MEM_ENDLESS_COMPRESSION_RATIO=4.0

# Enable archive memory (disk storage)
AI_MEM_ENDLESS_ENABLE_ARCHIVE=true

# Archive directory (relative to data dir)
AI_MEM_ENDLESS_ARCHIVE_DIR=archive

# Compression method: "ai" (uses LLM) or "heuristic" (fast, no LLM)
AI_MEM_ENDLESS_COMPRESSION_METHOD=heuristic

# Minimum output size before compression kicks in
AI_MEM_ENDLESS_MIN_OUTPUT_FOR_COMPRESSION=500

# Show compression stats in output
AI_MEM_ENDLESS_SHOW_COMPRESSION_STATS=true

# Auto-enable when observation count exceeds threshold
AI_MEM_ENDLESS_AUTO_ENABLE_THRESHOLD=50
```

### Programmatic Configuration

```python
from ai_mem.config import EndlessModeConfig

config = EndlessModeConfig(
    enabled=True,
    target_observation_tokens=500,
    compression_ratio=4.0,
    enable_archive=True,
    compression_method="heuristic",
)
```

## CLI Commands

```bash
# Enable Endless Mode
ai-mem endless --enable

# Disable Endless Mode
ai-mem endless --disable

# Show compression statistics
ai-mem endless --stats

# Retrieve full (uncompressed) output from archive
ai-mem endless-archive <observation-id> --session-id <session-id>

# Retrieve as JSON
ai-mem endless-archive <observation-id> --session-id <session-id> -f json
```

## API Endpoints

```bash
# Get Endless Mode stats
GET /api/endless/stats

# Enable Endless Mode
POST /api/endless/enable

# Disable Endless Mode
POST /api/endless/disable

# Get archived output
GET /api/endless/archive/{observation_id}?session_id=<session>
```

## How It Works

### Two-Tier Memory

1. **Working Memory** (Context)
   - Compressed observations (~500 tokens each)
   - Fits in LLM context window
   - Used for active reasoning

2. **Archive Memory** (Disk)
   - Full uncompressed outputs
   - NDJSON format for easy parsing
   - Retrievable on demand

### Compression Process

```
Tool Output (5000 tokens)
         ↓
   [Compression]
         ↓
Summary (500 tokens) → Working Memory
         +
Full Output → Archive Memory (disk)
```

### Heuristic Compression

When using `compression_method=heuristic`:
- Extracts key information (errors, results, paths)
- Removes redundant whitespace
- Truncates to target length
- Fast, no LLM API calls required

### AI Compression

When using `compression_method=ai`:
- Uses configured LLM to summarize
- Preserves semantic meaning
- More accurate but slower
- Requires LLM API access

## Archive Format

Archives are stored as NDJSON files:

```
~/.ai-mem/archive/<project>/<session>/archive.ndjson
```

Each line is a JSON object:
```json
{
  "observation_id": "abc-123",
  "session_id": "sess-456",
  "project": "/path/to/project",
  "tool_name": "Read",
  "tool_input": {"path": "/file.py"},
  "tool_output": "full file contents...",
  "compressed_output": "File contains Python class...",
  "compression_ratio": 4.5,
  "original_tokens": 2250,
  "compressed_tokens": 500,
  "created_at": 1700000000.0
}
```

## Best Practices

1. **Start with heuristic compression** for speed
2. **Use AI compression** for complex outputs that need semantic understanding
3. **Monitor compression stats** to tune target tokens
4. **Keep archive enabled** to never lose information
5. **Use archive retrieval** when you need full context

## Integration with Modes

Work modes affect Endless Mode behavior:

| Mode | Endless | Compression Ratio |
|------|---------|-------------------|
| code | Enabled | 4.0 |
| email | Enabled | 2.0 |
| research | Enabled | 3.0 |
| debug | Enabled | 2.0 |
| chill | Disabled | 8.0 |

## Troubleshooting

### High compression loss

If compressed outputs lose too much information:
- Increase `target_observation_tokens`
- Use AI compression instead of heuristic
- Adjust `compression_ratio` target

### Archive growing too large

- Implement periodic cleanup
- Set `max_observations_before_compress` to limit stored entries
- Use `ai-mem cleanup-stale` to remove old entries

### Compression too slow

- Switch to `heuristic` compression method
- Increase `compression_timeout_s`
- Use a faster LLM model

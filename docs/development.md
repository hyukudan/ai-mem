# Development

## Development and Testing

Run tests:

```bash
.venv/bin/python -m pytest tests/test_db.py tests/test_summary.py tests/test_privacy.py tests/test_chunking.py tests/test_dates.py
```

## Roadmap

Planned or proposed improvements:

- Provider adapters: Anthropic, Azure OpenAI, and additional local runtimes.
- Pluggable vector stores (pgvector, Qdrant) and faster hybrid search.
- Richer observations: attachments, file diffs, structured metadata.
- Incremental sync/export formats (JSONL, snapshots, merge tools).
- UI enhancements: session analytics, tag management, saved filters.
- Hook presets for more clients and IDEs.

## Credits and Attribution

ai-mem is inspired by [claude-mem](https://github.com/thedotmack/claude-mem).
We adopt similar memory concepts and extend them with multi-provider support and local-first defaults.

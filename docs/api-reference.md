# API Reference

## REST API

When you run `ai-mem server`, see docs at:
- http://localhost:8000/docs

Key endpoints:
- `POST /api/memories`: add a memory (project, session_id, obs_type, tags, metadata, title, summarize)
- `GET /api/search`: search (project, session_id, obs_type, tags, date_start, date_end, since, show_tokens)
- `GET /api/timeline`: timeline (project, session_id, obs_type, tags, date_start, date_end, since, depth_before, depth_after, show_tokens)
- `GET /api/observations`: list observations (project, session_id, obs_type, tags, date_start, date_end, since, limit)
- `GET /api/observations/{id}`: single observation
- `GET /api/observation/{id}`: alias
- `PATCH /api/observations/{id}`: update tags (tags)
- `DELETE /api/observations/{id}`
- `GET /api/projects`
- `GET /api/sessions` (project, active_only, goal, date_start, date_end, limit)
- `GET /api/sessions/{id}`
- `GET /api/sessions/{id}/observations`
- `POST /api/sessions/start` (project, goal, session_id)
- `POST /api/sessions/end` (session_id or latest)
- `GET /api/stats` (project, session_id, obs_type, tags, date_start, date_end, since, tag_limit, day_limit, type_tag_limit)
- `GET /api/tags` (project, session_id, obs_type, tags, date_start, date_end, limit)
- `POST /api/tags/add` (tag, project, session_id, obs_type, tags, date_start, date_end)
- `POST /api/tags/rename` (old_tag, new_tag, project, session_id, obs_type, tags, date_start, date_end)
- `POST /api/tags/delete` (tag, project, session_id, obs_type, tags, date_start, date_end)
- `GET /api/context/preview` (project, session_id, query, obs_type, obs_types, tags, total, full, full_field, show_tokens, wrap)
- `GET /api/context/inject` (same as preview)
- `GET /api/context` (alias)
- `GET /api/context/config`
- `POST /api/context/preview`
- `POST /api/context/inject`
- `GET /api/stream` (project, session_id, obs_type, tags, query, token)
- `GET /api/export` (project, session_id, obs_type, tags, date_start, date_end, since, limit, format=json|jsonl|csv)
- `POST /api/import` (preserves session_id unless project override changes scope)
- `POST /api/summarize` (project, session_id, count, obs_type, store, tags)
- `GET /api/health`
- `GET /api/readiness`
- `GET /api/version`

**Search cache header:** `/api/search` responses now include `X-AI-MEM-Search-Cache` with `hit`/`miss`, matching the viewer badge so automation can tell when a cached query was reused.

## Storage Layout

Default data directory: `~/.ai-mem`

- SQLite: `~/.ai-mem/ai-mem.sqlite`
- Vector DB: `~/.ai-mem/vector-db`

You can override these in config:

```bash
ai-mem config --data-dir /path/to/data
ai-mem config --sqlite-path /path/to/ai-mem.sqlite
ai-mem config --vector-dir /path/to/vector-db
```

## Privacy and Redaction

Use `<private>...</private>` to prevent sensitive content from being stored.
The private segments are removed before hashing, indexing, and storage.

`<ai-mem-context>` is reserved and stripped to prevent recursive ingestion.

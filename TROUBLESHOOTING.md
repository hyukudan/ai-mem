# ai-mem Troubleshooting Guide

This guide helps diagnose and resolve common issues with ai-mem.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Database Issues](#database-issues)
3. [API Server Issues](#api-server-issues)
4. [Chat Provider Issues](#chat-provider-issues)
5. [Event Ingestion Issues](#event-ingestion-issues)
6. [Performance Issues](#performance-issues)
7. [Configuration Issues](#configuration-issues)
8. [Common Error Messages](#common-error-messages)

---

## Installation Issues

### ModuleNotFoundError: No module named 'ai_mem'

**Cause:** Package not installed or not in Python path.

**Solution:**
```bash
# Install in development mode
pip install -e .

# Or install from wheel
pip install ai-mem
```

### ImportError: cannot import name 'AsyncOpenAI'

**Cause:** Missing or outdated OpenAI package.

**Solution:**
```bash
pip install --upgrade openai>=1.0.0
```

### Google GenerativeAI import error

**Cause:** Missing google-generativeai package.

**Solution:**
```bash
pip install google-generativeai
```

---

## Database Issues

### DatabaseConnectionError: Failed to connect to database

**Cause:** Database file path is invalid or permissions issue.

**Diagnosis:**
```bash
# Check if path exists and is writable
ls -la ~/.ai_mem/

# Check disk space
df -h ~/.ai_mem/
```

**Solutions:**
1. Ensure parent directory exists and is writable
2. Check if another process has locked the database
3. Try with a different database path:
   ```bash
   export AI_MEM_DB_PATH=/tmp/test_ai_mem.db
   ```

### DatabaseIntegrityError: constraint violation

**Cause:** Duplicate entries or foreign key violations.

**Solutions:**
1. Check for duplicate observation IDs
2. Ensure session exists before adding observations to it
3. Use `dedupe=True` when adding observations to skip duplicates

### "database is locked" errors

**Cause:** SQLite concurrent access issue.

**Solutions:**
1. Ensure only one writer process at a time
2. Increase busy timeout (default is 5000ms):
   ```python
   # In db.py, this is already set
   await self.conn.execute("PRAGMA busy_timeout = 5000")
   ```
3. Use WAL mode (already enabled by default)

---

## API Server Issues

### Server won't start: Address already in use

**Cause:** Another process is using the port.

**Solutions:**
```bash
# Find process using the port
lsof -i :8000

# Kill the process or use different port
ai-mem server --port 8001
```

### 401 Unauthorized on all requests

**Cause:** API token mismatch.

**Solutions:**
1. Check your token:
   ```bash
   echo $AI_MEM_API_TOKEN
   ```
2. Ensure client sends correct header:
   ```bash
   curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/api/health
   ```
3. Disable auth for testing (not recommended for production):
   ```bash
   unset AI_MEM_API_TOKEN
   ```

### CORS errors in browser

**Cause:** Origin not allowed.

**Solution:**
```bash
# Allow specific origins
export AI_MEM_ALLOWED_ORIGINS="http://localhost:3000,https://myapp.com"

# Or allow all (not recommended for production)
export AI_MEM_ALLOWED_ORIGINS="*"
```

### 500 Internal Server Error

**Diagnosis:**
1. Check server logs for detailed error
2. Enable debug logging:
   ```bash
   export AI_MEM_LOG_LEVEL=DEBUG
   ```

---

## Chat Provider Issues

### APIError: Anthropic API error

**Causes:**
- Invalid API key
- Rate limiting
- Model not available

**Solutions:**
1. Verify API key:
   ```bash
   echo $AI_MEM_ANTHROPIC_KEY
   ```
2. Check rate limits in Anthropic console
3. Try a different model:
   ```bash
   export AI_MEM_CHAT_MODEL=claude-3-haiku-20240307
   ```

### APIError: OpenAI API error

**Solutions:**
1. Verify API key:
   ```bash
   echo $AI_MEM_OPENAI_KEY
   # or
   echo $OPENAI_API_KEY
   ```
2. For Azure OpenAI, verify all required env vars:
   ```bash
   echo $AI_MEM_AZURE_OPENAI_ENDPOINT
   echo $AI_MEM_AZURE_OPENAI_KEY
   echo $AI_MEM_AZURE_OPENAI_DEPLOYMENT
   ```

### NetworkError: Connection failed

**Causes:**
- Network connectivity issues
- Firewall blocking requests
- Invalid endpoint URL

**Solutions:**
1. Test connectivity:
   ```bash
   curl -v https://api.anthropic.com
   ```
2. Check firewall rules
3. Verify base URL configuration

### Timeouts during summarization

**Cause:** Large content or slow API response.

**Solutions:**
1. Increase timeout:
   ```python
   # In provider initialization
   timeout_s=120.0
   ```
2. Reduce content size with max_length setting
3. Use faster model for summarization

---

## Event Ingestion Issues

### PayloadParseError: Could not parse payload

**Cause:** Event format doesn't match adapter expectations.

**Diagnosis:**
```python
# Check what adapter is being used
from ai_mem.adapters import get_adapter
adapter = get_adapter("your-host")
print(adapter.host_name)
```

**Solutions:**
1. Check payload format matches adapter expectations
2. Use `generic` host for custom formats:
   ```json
   {
     "host": "generic",
     "payload": {
       "tool_name": "MyTool",
       "tool_input": {...},
       "tool_output": {...}
     }
   }
   ```

### Events being skipped

**Causes:**
- Privacy filters blocking content
- Duplicate event (idempotency)
- Empty content

**Diagnosis:**
```bash
# Check response for skip reason
curl -X POST http://localhost:8000/api/events \
  -H "Content-Type: application/json" \
  -d '{"host": "generic", "payload": {...}}'

# Response will include:
# {"status": "skipped", "reason": "private|duplicate|empty_content"}
```

### Tool name not recognized

**Solution:** Add custom tool mappings in adapter configuration or use generic category.

---

## Performance Issues

### Slow searches

**Causes:**
- No FTS indexes
- Large result sets
- Missing database indexes

**Solutions:**
1. Verify FTS is working:
   ```sql
   SELECT * FROM observations_fts LIMIT 1;
   ```
2. Add query filters to reduce result set
3. Use pagination with `limit` parameter
4. Check index health:
   ```sql
   PRAGMA index_list(observations);
   ```

### High memory usage

**Causes:**
- Large search cache
- Many observations loaded at once

**Solutions:**
1. Reduce cache size:
   ```python
   # In config
   search_cache_size: 100
   ```
2. Use pagination for large exports
3. Stream results instead of loading all

### Slow observation ingestion

**Solutions:**
1. Disable summarization for bulk imports:
   ```json
   {"summarize": false}
   ```
2. Use batch imports via `/api/import`
3. Index optimization:
   ```sql
   PRAGMA optimize;
   VACUUM;
   ```

---

## Configuration Issues

### Config file not loading

**Diagnosis:**
```bash
# Check config location
ls -la ~/.ai_mem/config.yaml

# Check environment override
echo $AI_MEM_CONFIG
```

**Solutions:**
1. Create config file:
   ```bash
   mkdir -p ~/.ai_mem
   ai-mem config show > ~/.ai_mem/config.yaml
   ```
2. Use explicit path:
   ```bash
   export AI_MEM_CONFIG=/path/to/config.yaml
   ```

### Environment variables not working

**Common issues:**
- Spaces around equals sign (wrong: `KEY = value`)
- Missing export (just `KEY=value` doesn't persist)
- Shell not reloaded after .env changes

**Solutions:**
```bash
# Correct format
export AI_MEM_API_TOKEN="your-token"

# Verify it's set
env | grep AI_MEM
```

---

## Common Error Messages

### "Invalid UUID format"

**Cause:** Observation/session ID not a valid UUID.

**Solution:** Use proper UUIDs (e.g., from `uuid.uuid4()`) or let ai-mem generate them.

### "Observation not found"

**Causes:**
- ID doesn't exist
- ID is for wrong resource type
- Observation was deleted

**Diagnosis:**
```bash
curl http://localhost:8000/api/observations?limit=10
```

### "Session not found"

**Cause:** Session ID doesn't exist or was never started.

**Solution:** Start a session first:
```bash
ai-mem session start --goal "My task"
```

### "Rate limit exceeded"

**Cause:** Too many API requests.

**Solutions:**
1. Implement client-side rate limiting
2. Increase rate limit in server config:
   ```bash
   export AI_MEM_RATE_LIMIT="100/minute"
   ```
3. Add delay between requests

---

## Getting Help

### Enable Debug Logging

```bash
export AI_MEM_LOG_LEVEL=DEBUG
ai-mem server
```

### Check Version

```bash
ai-mem --version
python -c "import ai_mem; print(ai_mem.__version__)"
```

### Report Issues

When reporting issues, include:

1. ai-mem version
2. Python version
3. Operating system
4. Full error message and stack trace
5. Minimal reproduction steps
6. Relevant configuration (without secrets)

File issues at: https://github.com/anthropics/ai-mem/issues

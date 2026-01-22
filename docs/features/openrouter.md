# OpenRouter Provider

Access 200+ LLM models including free options through a single API.

## Overview

OpenRouter provides unified access to models from:
- OpenAI (GPT-4, GPT-4o)
- Anthropic (Claude 3, Claude 3.5)
- Google (Gemini Pro, Gemini Flash)
- Meta (Llama 3.1, Llama 3.2)
- Mistral (Mistral Large, Mixtral)
- And many more...

## Free Models

These models work with the free tier (no API key required for some):

| Model | ID |
|-------|-----|
| Llama 3.2 3B | `meta-llama/llama-3.2-3b-instruct:free` |
| Llama 3.2 1B | `meta-llama/llama-3.2-1b-instruct:free` |
| Gemma 2 9B | `google/gemma-2-9b-it:free` |
| Qwen 2 7B | `qwen/qwen-2-7b-instruct:free` |
| Mistral 7B | `mistralai/mistral-7b-instruct:free` |
| Zephyr 7B | `huggingfaceh4/zephyr-7b-beta:free` |
| OpenChat 7B | `openchat/openchat-7b:free` |

## Configuration

### Environment Variables

```bash
# API key (get from openrouter.ai)
OPENROUTER_API_KEY=sk-or-v1-...

# Base URL (default)
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Default model
OPENROUTER_DEFAULT_MODEL=meta-llama/llama-3.2-3b-instruct:free

# For rankings/leaderboard
OPENROUTER_SITE_URL=https://your-site.com
OPENROUTER_APP_NAME=ai-mem

# Request settings
OPENROUTER_MAX_TOKENS=4096
OPENROUTER_TEMPERATURE=0.7
OPENROUTER_TIMEOUT=60
OPENROUTER_MAX_CONTEXT_MESSAGES=20
```

## Usage

### Quick Chat

```python
from ai_mem.providers.openrouter import quick_chat

# Simple one-shot completion (uses free model by default)
response = await quick_chat("What is Python?")
print(response)
```

### With Provider Class

```python
from ai_mem.providers import OpenRouterProvider, OpenRouterConfig

# Create provider
provider = OpenRouterProvider()

# Simple chat
response = await provider.chat(
    prompt="Explain async/await in Python",
    system_prompt="You are a Python expert.",
)
print(response.content)
print(f"Tokens used: {response.usage}")
```

### With Conversation History

```python
from ai_mem.providers.openrouter import OpenRouterProvider, ChatMessage

provider = OpenRouterProvider()

messages = [
    ChatMessage("system", "You are a helpful coding assistant."),
    ChatMessage("user", "How do I read a file in Python?"),
    ChatMessage("assistant", "You can use open() with a context manager..."),
    ChatMessage("user", "What about async file reading?"),
]

response = await provider.chat_messages(messages)
print(response.content)
```

### Using Different Models

```python
# Use model alias
response = await provider.chat(
    prompt="Hello!",
    model="gpt-4o",  # Resolves to openai/gpt-4o
)

# Use full model ID
response = await provider.chat(
    prompt="Hello!",
    model="anthropic/claude-3.5-sonnet",
)
```

## Model Aliases

Popular models have short aliases:

| Alias | Full ID |
|-------|---------|
| `gpt-4o` | `openai/gpt-4o` |
| `gpt-4o-mini` | `openai/gpt-4o-mini` |
| `gpt-4-turbo` | `openai/gpt-4-turbo` |
| `claude-3.5-sonnet` | `anthropic/claude-3.5-sonnet` |
| `claude-3-opus` | `anthropic/claude-3-opus` |
| `claude-3-haiku` | `anthropic/claude-3-haiku` |
| `gemini-pro` | `google/gemini-pro` |
| `gemini-1.5-pro` | `google/gemini-pro-1.5` |
| `gemini-1.5-flash` | `google/gemini-flash-1.5` |
| `llama-3.1-70b` | `meta-llama/llama-3.1-70b-instruct` |
| `llama-3.1-8b` | `meta-llama/llama-3.1-8b-instruct` |
| `mistral-large` | `mistralai/mistral-large` |
| `mixtral-8x7b` | `mistralai/mixtral-8x7b-instruct` |

## Listing Models

```python
# List all models
models = await provider.list_models()
for m in models:
    print(f"{m.id}: {m.name} (ctx: {m.context_length})")
    if m.is_free:
        print("  FREE!")

# List only free models
free_models = await provider.list_free_models()

# Get specific model info
model = await provider.get_model_info("openai/gpt-4o")
print(f"Context: {model.context_length}")
print(f"Price: ${model.pricing['prompt']}/1K tokens")
```

## Response Object

```python
response = await provider.chat("Hello!")

# Content
print(response.content)  # "Hello! How can I help?"

# Model used
print(response.model)  # "meta-llama/llama-3.2-3b-instruct:free"

# Token usage
print(response.usage)
# {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}

# Finish reason
print(response.finish_reason)  # "stop"

# Raw API response
print(response.raw_response)
```

## Error Handling

```python
import httpx

try:
    response = await provider.chat("Hello!")
except httpx.HTTPStatusError as e:
    if e.response.status_code == 401:
        print("Invalid API key")
    elif e.response.status_code == 429:
        print("Rate limited")
    else:
        print(f"API error: {e.response.status_code}")
except Exception as e:
    print(f"Request failed: {e}")
```

## Integration with ai-mem

OpenRouter can be used for:

### AI Compression in Endless Mode

```python
from ai_mem.providers.openrouter import get_openrouter_provider
from ai_mem.compression import CompressionService

provider = get_openrouter_provider()
service = CompressionService(provider=provider)

result = await service.compress(
    text="Long tool output...",
    target_ratio=4.0,
)
```

### Semantic Search Reranking

```python
# Use OpenRouter model for cross-encoder reranking
provider = get_openrouter_provider()
# ... use for reranking
```

## Best Practices

1. **Start with free models** for development
2. **Use appropriate models** for each task (small for simple, large for complex)
3. **Monitor token usage** through response.usage
4. **Handle rate limits** with exponential backoff
5. **Close provider** when done: `await provider.close()`

## Pricing

Check current pricing at [openrouter.ai/models](https://openrouter.ai/models).

Free tier has:
- Rate limits
- Some model restrictions
- Sufficient for development/testing

Paid tier offers:
- Higher rate limits
- Access to all models
- Priority routing

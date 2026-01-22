"""OpenRouter Provider - Access to 200+ models including free options.

OpenRouter provides unified access to models from OpenAI, Anthropic, Google,
Meta, Mistral, and many others through a single API.

Configuration:
    OPENROUTER_API_KEY=your-api-key
    OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
    OPENROUTER_DEFAULT_MODEL=meta-llama/llama-3.2-3b-instruct:free
    OPENROUTER_SITE_URL=https://your-site.com  # For rankings
    OPENROUTER_APP_NAME=ai-mem

Free models (no API key required for some):
    - meta-llama/llama-3.2-3b-instruct:free
    - google/gemma-2-9b-it:free
    - qwen/qwen-2-7b-instruct:free
    - mistralai/mistral-7b-instruct:free
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from ..logging_config import get_logger

logger = get_logger("openrouter")

# OpenRouter API configuration
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1"
)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_DEFAULT_MODEL = os.environ.get(
    "OPENROUTER_DEFAULT_MODEL",
    "meta-llama/llama-3.2-3b-instruct:free"
)
OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "")
OPENROUTER_APP_NAME = os.environ.get("OPENROUTER_APP_NAME", "ai-mem")

# Free models that work without API key or with free tier
FREE_MODELS = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.2-1b-instruct:free",
    "google/gemma-2-9b-it:free",
    "qwen/qwen-2-7b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "huggingfaceh4/zephyr-7b-beta:free",
    "openchat/openchat-7b:free",
    "nousresearch/nous-capybara-7b:free",
]

# Popular models
POPULAR_MODELS = {
    # OpenAI
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4-turbo": "openai/gpt-4-turbo",
    # Anthropic
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3-opus": "anthropic/claude-3-opus",
    "claude-3-haiku": "anthropic/claude-3-haiku",
    # Google
    "gemini-pro": "google/gemini-pro",
    "gemini-1.5-pro": "google/gemini-pro-1.5",
    "gemini-1.5-flash": "google/gemini-flash-1.5",
    # Meta
    "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    # Mistral
    "mistral-large": "mistralai/mistral-large",
    "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct",
}


@dataclass
class OpenRouterConfig:
    """Configuration for OpenRouter provider."""

    api_key: str = ""
    base_url: str = OPENROUTER_BASE_URL
    default_model: str = OPENROUTER_DEFAULT_MODEL
    site_url: str = OPENROUTER_SITE_URL
    app_name: str = OPENROUTER_APP_NAME
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 60.0
    max_context_messages: int = 20

    @classmethod
    def from_env(cls) -> "OpenRouterConfig":
        """Create config from environment variables."""
        return cls(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            default_model=OPENROUTER_DEFAULT_MODEL,
            site_url=OPENROUTER_SITE_URL,
            app_name=OPENROUTER_APP_NAME,
            max_tokens=int(os.environ.get("OPENROUTER_MAX_TOKENS", "4096")),
            temperature=float(os.environ.get("OPENROUTER_TEMPERATURE", "0.7")),
            timeout=float(os.environ.get("OPENROUTER_TIMEOUT", "60")),
            max_context_messages=int(os.environ.get("OPENROUTER_MAX_CONTEXT_MESSAGES", "20")),
        )


@dataclass
class ChatMessage:
    """A chat message."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatResponse:
    """Response from chat completion."""

    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class ModelInfo:
    """Information about a model."""

    id: str
    name: str
    description: str
    context_length: int
    pricing: Dict[str, float]  # {"prompt": 0.0, "completion": 0.0}
    is_free: bool


class OpenRouterProvider:
    """Provider for OpenRouter API.

    Usage:
        provider = OpenRouterProvider()

        # Simple completion
        response = await provider.chat("What is Python?")

        # With conversation history
        messages = [
            ChatMessage("system", "You are helpful."),
            ChatMessage("user", "Hello!"),
        ]
        response = await provider.chat_messages(messages)

        # List available models
        models = await provider.list_models()
    """

    def __init__(self, config: Optional[OpenRouterConfig] = None):
        """Initialize provider.

        Args:
            config: Optional configuration (defaults to env vars)
        """
        self.config = config or OpenRouterConfig.from_env()
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {
                "Content-Type": "application/json",
            }

            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            if self.config.site_url:
                headers["HTTP-Referer"] = self.config.site_url

            if self.config.app_name:
                headers["X-Title"] = self.config.app_name

            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=self.config.timeout,
            )

        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        """Simple chat completion.

        Args:
            prompt: User prompt
            model: Model to use (default: config default)
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            ChatResponse with completion
        """
        messages = []

        if system_prompt:
            messages.append(ChatMessage("system", system_prompt))

        messages.append(ChatMessage("user", prompt))

        return await self.chat_messages(messages, model=model, **kwargs)

    async def chat_messages(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ChatResponse:
        """Chat completion with message history.

        Args:
            messages: List of chat messages
            model: Model to use
            max_tokens: Max tokens in response
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            ChatResponse with completion
        """
        model = model or self.config.default_model

        # Resolve model aliases
        if model in POPULAR_MODELS:
            model = POPULAR_MODELS[model]

        payload = {
            "model": model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages[-self.config.max_context_messages:]
            ],
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            **kwargs,
        }

        try:
            response = await self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})

            return ChatResponse(
                content=message.get("content", ""),
                model=data.get("model", model),
                usage=data.get("usage", {}),
                finish_reason=choice.get("finish_reason", "stop"),
                raw_response=data,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter API error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"OpenRouter request failed: {e}")
            raise

    async def list_models(self, include_paid: bool = True) -> List[ModelInfo]:
        """List available models.

        Args:
            include_paid: Include paid models

        Returns:
            List of ModelInfo
        """
        try:
            response = await self.client.get("/models")
            response.raise_for_status()
            data = response.json()

            models = []
            for m in data.get("data", []):
                pricing = m.get("pricing", {})
                is_free = (
                    float(pricing.get("prompt", 1)) == 0 and
                    float(pricing.get("completion", 1)) == 0
                )

                if not include_paid and not is_free:
                    continue

                models.append(ModelInfo(
                    id=m.get("id", ""),
                    name=m.get("name", ""),
                    description=m.get("description", ""),
                    context_length=m.get("context_length", 4096),
                    pricing={
                        "prompt": float(pricing.get("prompt", 0)),
                        "completion": float(pricing.get("completion", 0)),
                    },
                    is_free=is_free,
                ))

            return models

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    async def list_free_models(self) -> List[ModelInfo]:
        """List only free models.

        Returns:
            List of free ModelInfo
        """
        return await self.list_models(include_paid=False)

    async def get_model_info(self, model: str) -> Optional[ModelInfo]:
        """Get information about a specific model.

        Args:
            model: Model ID

        Returns:
            ModelInfo or None
        """
        models = await self.list_models()
        for m in models:
            if m.id == model:
                return m
        return None


# Singleton instance
_provider: Optional[OpenRouterProvider] = None


def get_openrouter_provider() -> OpenRouterProvider:
    """Get or create the OpenRouter provider singleton.

    Returns:
        OpenRouterProvider instance
    """
    global _provider

    if _provider is None:
        _provider = OpenRouterProvider()

    return _provider


async def quick_chat(
    prompt: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Quick one-shot chat completion.

    Args:
        prompt: User prompt
        model: Model to use (default: free model)
        system_prompt: Optional system prompt

    Returns:
        Completion text
    """
    provider = get_openrouter_provider()
    response = await provider.chat(
        prompt=prompt,
        model=model or FREE_MODELS[0],
        system_prompt=system_prompt,
    )
    return response.content

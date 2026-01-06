import httpx
from typing import List, Optional

from .base import ChatProvider, ChatMessage

DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


def _extract_text(payload: dict) -> str:
    content = payload.get("content") or []
    chunks = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks).strip()


class AnthropicProvider(ChatProvider):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str = "https://api.anthropic.com",
        timeout_s: float = 60.0,
        anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
        max_tokens: int = 1024,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.anthropic_version = anthropic_version
        self.max_tokens = max_tokens
        self.client = httpx.Client(timeout=timeout_s)

    def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        payload = {
            "model": model or self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
        }
        response = self.client.post(
            f"{self.base_url}/v1/messages",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        return _extract_text(response.json())

    def get_name(self) -> str:
        return "anthropic"

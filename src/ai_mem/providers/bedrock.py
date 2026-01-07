import asyncio
import json
from typing import Any, Dict, List, Optional

import boto3

from .base import ChatProvider, ChatMessage

DEFAULT_ANTHROPIC_VERSION = "bedrock-2023-05-31"


def _messages_to_prompt(messages: List[ChatMessage]) -> str:
    lines = []
    for msg in messages:
        role = msg.role.upper()
        lines.append(f"{role}: {msg.content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def _extract_anthropic_text(payload: Dict[str, Any]) -> str:
    content = payload.get("content") or []
    chunks: List[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks).strip()


class BedrockProvider(ChatProvider):
    def __init__(
        self,
        model_id: str,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        profile: Optional[str] = None,
        max_tokens: int = 1024,
        anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
    ):
        self.model_id = model_id
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        self.client = session.client(
            "bedrock-runtime",
            region_name=region,
            endpoint_url=endpoint_url,
        )
        self.max_tokens = max_tokens
        self.anthropic_version = anthropic_version

    def _invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(payload),
            accept="application/json",
            contentType="application/json",
        )
        body = response.get("body")
        if hasattr(body, "read"):
            raw = body.read()
        else:
            raw = body
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        return json.loads(raw or "{}")

    def _sync_chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Synchronous chat implementation."""
        model_id = model or self.model_id
        self.model_id = model_id
        if model_id.startswith("anthropic."):
            payload = {
                "anthropic_version": self.anthropic_version,
                "max_tokens": self.max_tokens,
                "temperature": temperature,
                "messages": [
                    {
                        "role": msg.role,
                        "content": [{"type": "text", "text": msg.content}],
                    }
                    for msg in messages
                ],
            }
            data = self._invoke(payload)
            return _extract_anthropic_text(data)

        prompt = _messages_to_prompt(messages)
        if model_id.startswith("amazon."):
            payload = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": self.max_tokens,
                    "temperature": temperature,
                },
            }
            data = self._invoke(payload)
            results = data.get("results") or []
            if results:
                return (results[0].get("outputText") or "").strip()
            return ""

        if model_id.startswith("cohere."):
            payload = {
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": temperature,
            }
            data = self._invoke(payload)
            generations = data.get("generations") or []
            if generations:
                return (generations[0].get("text") or "").strip()
            return ""

        if model_id.startswith("meta."):
            payload = {
                "prompt": prompt,
                "max_gen_len": self.max_tokens,
                "temperature": temperature,
            }
            data = self._invoke(payload)
            return (data.get("generation") or "").strip()

        payload = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
        }
        data = self._invoke(payload)
        for key in ("generation", "output", "text", "completion"):
            value = data.get(key)
            if isinstance(value, str):
                return value.strip()
        return ""

    async def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Async chat using thread pool for blocking API call."""
        return await asyncio.to_thread(
            self._sync_chat, messages, model, temperature
        )

    def get_name(self) -> str:
        return "bedrock"

from typing import List, Optional

from openai import AsyncOpenAI

from .base import ChatProvider, ChatMessage


class OpenAICompatibleProvider(ChatProvider):
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str],
        model_name: str,
        timeout_s: float = 60.0,
    ):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "local",
            timeout=timeout_s,
        )
        self.model_name = model_name

    async def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=model or self.model_name,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()

    def get_name(self) -> str:
        return "openai-compatible"

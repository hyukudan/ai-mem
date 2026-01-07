from typing import List, Optional

from openai import AsyncOpenAI, APIError as OpenAIAPIError, APIConnectionError

from ..exceptions import APIError, NetworkError
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
        self.base_url = base_url

    async def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=model or self.model_name,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
            )
            return (response.choices[0].message.content or "").strip()
        except OpenAIAPIError as e:
            raise APIError("openai", getattr(e, "status_code", None), str(e)) from None
        except APIConnectionError as e:
            raise NetworkError(f"OpenAI connection failed: {type(e).__name__}") from None

    def get_name(self) -> str:
        return "openai-compatible"

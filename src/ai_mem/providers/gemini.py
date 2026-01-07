import asyncio
import google.generativeai as genai
from typing import List, Optional

from ..exceptions import APIError
from .base import ChatProvider, ChatMessage


class GeminiProvider(ChatProvider):
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def _format_messages(self, messages: List[ChatMessage]) -> str:
        formatted = []
        for msg in messages:
            formatted.append(f"{msg.role.upper()}: {msg.content}")
        formatted.append("ASSISTANT:")
        return "\n".join(formatted)

    def _sync_chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.2,
    ) -> str:
        """Synchronous chat implementation."""
        prompt = self._format_messages(messages)
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": temperature},
            )
            return (response.text or "").strip()
        except Exception as e:
            # Sanitize error to avoid leaking API key
            raise APIError("gemini", None, f"Gemini API error: {type(e).__name__}") from None

    async def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Async chat using thread pool for blocking API call."""
        return await asyncio.to_thread(
            self._sync_chat, messages, temperature
        )

    def get_name(self) -> str:
        return "gemini"

import google.generativeai as genai
from typing import List, Optional
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

    def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        prompt = self._format_messages(messages)
        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": temperature},
        )
        return (response.text or "").strip()

    def get_name(self) -> str:
        return "gemini"

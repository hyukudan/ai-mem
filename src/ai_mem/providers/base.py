from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ChatMessage:
    role: str
    content: str


class ChatProvider(ABC):
    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Generate a chat response from a list of messages."""
        raise NotImplementedError

    def summarize(self, text: str, model: Optional[str] = None) -> str:
        prompt = (
            "Summarize the following interaction into a short, factual observation "
            "for a long-term memory system. Use a neutral tone and be concise.\n\n"
            f"Interaction:\n{text}"
        )
        messages = [ChatMessage(role="user", content=prompt)]
        return self.chat(messages, model=model, temperature=0.2)

    @abstractmethod
    def get_name(self) -> str:
        """Returns the provider name."""
        raise NotImplementedError


class NoOpChatProvider(ChatProvider):
    def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        return ""

    def summarize(self, text: str, model: Optional[str] = None) -> str:
        if len(text) <= 500:
            return text
        return text[:500] + "..."

    def get_name(self) -> str:
        return "none"

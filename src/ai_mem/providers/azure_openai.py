from typing import List, Optional

from openai import AsyncAzureOpenAI

from .base import ChatProvider, ChatMessage


class AzureOpenAIProvider(ChatProvider):
    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
        deployment_name: str,
        timeout_s: float = 60.0,
    ):
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=timeout_s,
        )
        self.deployment_name = deployment_name

    async def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=model or self.deployment_name,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()

    def get_name(self) -> str:
        return "azure-openai"

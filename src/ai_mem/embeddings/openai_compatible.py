from typing import List, Optional

from openai import OpenAI

from .base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str],
        model_name: str,
        timeout_s: float = 60.0,
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "local",
            timeout=timeout_s,
        )
        self.model_name = model_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        data = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in data]

    def get_name(self) -> str:
        return "openai-compatible"

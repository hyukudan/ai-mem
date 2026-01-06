from typing import List

from openai import AzureOpenAI

from .base import EmbeddingProvider


class AzureOpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
        deployment_name: str,
        timeout_s: float = 60.0,
    ):
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=timeout_s,
        )
        self.deployment_name = deployment_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.deployment_name,
            input=texts,
        )
        data = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in data]

    def get_name(self) -> str:
        return "azure-openai"

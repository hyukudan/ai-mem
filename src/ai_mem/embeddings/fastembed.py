from typing import List

from fastembed import TextEmbedding

from .base import EmbeddingProvider


class FastEmbedProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.model = TextEmbedding(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[float(x) for x in vector] for vector in self.model.embed(texts)]

    def get_name(self) -> str:
        return "fastembed"

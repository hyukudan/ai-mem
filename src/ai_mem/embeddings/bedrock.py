import json
import os
from typing import Any, Dict, List, Optional

import boto3

from .base import EmbeddingProvider


class BedrockEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model_id: str,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        profile: Optional[str] = None,
        input_type: Optional[str] = None,
    ):
        self.model_id = model_id
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        self.client = session.client(
            "bedrock-runtime",
            region_name=region,
            endpoint_url=endpoint_url,
        )
        self.input_type = input_type

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

    def embed(self, texts: List[str]) -> List[List[float]]:
        model_id = self.model_id
        if model_id.startswith("amazon.titan-embed-text"):
            embeddings: List[List[float]] = []
            for text in texts:
                payload = {"inputText": text}
                data = self._invoke(payload)
                vector = data.get("embedding") or data.get("embeddings")
                if isinstance(vector, list):
                    embeddings.append([float(x) for x in vector])
                else:
                    embeddings.append([])
            return embeddings

        if model_id.startswith("cohere.embed"):
            input_type = (
                self.input_type
                or os.environ.get("AI_MEM_BEDROCK_EMBED_INPUT_TYPE")
                or "search_document"
            )
            payload = {"texts": texts, "input_type": input_type}
            data = self._invoke(payload)
            vectors = data.get("embeddings") or []
            return [[float(x) for x in vec] for vec in vectors]

        raise ValueError(f"Unsupported Bedrock embedding model: {model_id}")

    def get_name(self) -> str:
        return "bedrock"

from typing import Any, Dict, List, Optional
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except ImportError:
    QdrantClient = None
    models = None

from .base import VectorStoreProvider

class QdrantVectorStore(VectorStoreProvider):
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        collection_name: str = "observations",
        vector_size: int = 1536,
        distance: str = "Cosine"
    ):
        if QdrantClient is None:
            raise ImportError("qdrant-client is required to use QdrantVectorStore. Please install it with `pip install qdrant-client`.")
        
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        
        # Ensure collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if not exists:
            if distance.lower() == "cosine":
                dist = models.Distance.COSINE
            elif distance.lower() == "euclidean":
                dist = models.Distance.EUCLID
            elif distance.lower() == "dot":
                dist = models.Distance.DOT
            else:
                dist = models.Distance.COSINE

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=dist),
            )

    def add(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        if not embeddings:
            return
        
        points = []
        for i in range(len(embeddings)):
            payload = metadatas[i].copy()
            payload["document"] = documents[i]
            points.append(
                models.PointStruct(
                    id=ids[i],
                    vector=embeddings[i],
                    payload=payload
                )
            )
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def query(
        self,
        embedding: List[float],
        n_results: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        
        query_filter = None
        if where:
            conditions = []
            for key, value in where.items():
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            if conditions:
                query_filter = models.Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=n_results,
            query_filter=query_filter
        )
        
        ids = []
        metadatas = []
        distances = []
        
        for hit in results:
            ids.append(str(hit.id))
            payload = hit.payload or {}
            # Separate document from metadata if needed, or keep it
            # The interface expects metadatas list
            metadatas.append(payload)
            distances.append(hit.score)
            
        return {
            "ids": [ids],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def delete_ids(self, ids: List[str]) -> None:
        if not ids:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=ids),
        )

    def delete_where(self, where: Dict[str, Any]) -> None:
        if not where:
            return
            
        conditions = []
        for key, value in where.items():
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                )
            )
            
        if conditions:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=conditions)
                )
            )

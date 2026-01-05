from typing import Any, Dict, List, Optional

import chromadb


class VectorStore:
    def __init__(self, path: str, collection_name: str = "observations"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
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
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def query(
        self,
        embedding: List[float],
        n_results: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where,
        )

    def delete_ids(self, ids: List[str]) -> None:
        if not ids:
            return
        self.collection.delete(ids=ids)

    def delete_where(self, where: Dict[str, Any]) -> None:
        if not where:
            return
        self.collection.delete(where=where)

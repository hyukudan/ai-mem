from typing import Any, Dict, List, Optional

class VectorStoreProvider:
    def add(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        raise NotImplementedError()

    def query(
        self,
        embedding: List[float],
        n_results: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Returns a dictionary with keys: 'ids', 'metadatas', 'distances'
        """
        raise NotImplementedError()

    def delete_ids(self, ids: List[str]) -> None:
        raise NotImplementedError()

    def delete_where(self, where: Dict[str, Any]) -> None:
        raise NotImplementedError()

"""Search Service - Orchestrate search operations.

This service coordinates hybrid search (FTS + vector),
timeline queries, and two-stage search with reranking.
"""

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..logging_config import get_logger

if TYPE_CHECKING:
    from ..config import AppConfig
    from ..db import DatabaseManager
    from .cache import SearchCacheService
    from .similarity import SimilarityService
    from .reranking import RerankingService

logger = get_logger("services.search")


class SearchService:
    """Orchestrates search operations.

    Combines FTS (BM25), vector search, and reranking
    for optimal search quality.

    Usage:
        service = SearchService(db, vector_store, embedding_provider, ...)
        results = await service.search("my query", limit=10)
        results = await service.search_with_rerank("my query", limit=10)
    """

    def __init__(
        self,
        db: "DatabaseManager",
        vector_store: Any,
        embedding_provider: Any,
        config: "AppConfig",
        cache_service: "SearchCacheService",
        similarity_service: "SimilarityService",
        reranking_service: "RerankingService",
    ):
        """Initialize search service.

        Args:
            db: Database manager
            vector_store: Vector store for semantic search
            embedding_provider: Provider for query embeddings
            config: Application configuration
            cache_service: Search cache service
            similarity_service: Similarity computation service
            reranking_service: Result reranking service
        """
        self.db = db
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.config = config
        self.cache = cache_service
        self.similarity = similarity_service
        self.reranking = reranking_service

    async def search(
        self,
        query: str,
        limit: int = 10,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        since: Optional[float] = None,
        tag_filters: Optional[List[str]] = None,
        include_scores: bool = True,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining FTS and vector search.

        Args:
            query: Search query
            limit: Maximum results
            project: Filter by project
            obs_type: Filter by observation type
            session_id: Filter by session
            date_start: Filter by date range start
            date_end: Filter by date range end
            since: Filter to observations after this timestamp
            tag_filters: Filter by tags
            include_scores: Include score breakdown

        Returns:
            List of search results with scores
        """
        start_time = time.perf_counter()

        # Normalize date filters
        if since and not date_start:
            date_start = since

        # Check cache
        cache_key = self.cache.build_key(
            query, limit, project, obs_type, session_id, date_start, date_end, tag_filters
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for query: {query[:50]}")
            return cached

        # FTS search
        fts_results = await self.db.search_observations_fts(
            query=query,
            limit=limit * 2,  # Get extra for merging
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=date_start,
            date_end=date_end,
            tag_filters=tag_filters,
        )

        # Vector search
        vector_results = await self._vector_search(
            query=query,
            project=project,
            session_id=session_id,
            date_start=date_start,
            date_end=date_end,
            limit=limit * 2,
            tag_filters=tag_filters,
        )

        # Merge and rank results
        results = self._merge_results(
            fts_results=fts_results,
            vector_results=vector_results,
            limit=limit,
            include_scores=include_scores,
        )

        # Cache results
        self.cache.set(cache_key, results)

        elapsed = time.perf_counter() - start_time
        logger.debug(f"Search completed in {elapsed*1000:.1f}ms: {len(results)} results")

        return results

    async def _vector_search(
        self,
        query: str,
        project: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        limit: int = 20,
        tag_filters: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search.

        Args:
            query: Search query
            project: Filter by project
            session_id: Filter by session
            date_start: Date range start
            date_end: Date range end
            limit: Maximum results
            tag_filters: Tag filters

        Returns:
            Vector search results with similarity scores
        """
        if not self.embedding_provider or not self.vector_store:
            return []

        try:
            # Embed query
            query_embedding = await self.embedding_provider.embed(query)

            # Build metadata filter
            filters = {}
            if project:
                filters["project"] = project
            if session_id:
                filters["session_id"] = session_id

            # Search vector store
            raw_results = await self.vector_store.search(
                embedding=query_embedding,
                limit=limit,
                filters=filters if filters else None,
            )

            # Convert to standard format
            results = []
            for r in raw_results:
                obs_id = r.get("observation_id") or r.get("id", "").split(":")[0]
                results.append({
                    "id": obs_id,
                    "vector_score": r.get("score", r.get("similarity", 0)),
                    "text": r.get("text", ""),
                    "metadata": r.get("metadata", {}),
                })

            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _merge_results(
        self,
        fts_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        limit: int,
        include_scores: bool,
    ) -> List[Dict[str, Any]]:
        """Merge FTS and vector search results.

        Args:
            fts_results: Full-text search results
            vector_results: Vector search results
            limit: Maximum results
            include_scores: Include score breakdown

        Returns:
            Merged and ranked results
        """
        # Index by observation ID
        by_id: Dict[str, Dict[str, Any]] = {}

        for r in fts_results:
            obs_id = r.get("id")
            if obs_id:
                by_id[obs_id] = {
                    **r,
                    "fts_score": r.get("score", r.get("fts_score", 0)),
                    "vector_score": 0,
                }

        for r in vector_results:
            obs_id = r.get("id")
            if obs_id:
                if obs_id in by_id:
                    by_id[obs_id]["vector_score"] = r.get("vector_score", 0)
                else:
                    by_id[obs_id] = {
                        **r,
                        "fts_score": 0,
                        "vector_score": r.get("vector_score", 0),
                    }

        # Compute combined scores
        results = []
        for obs_id, data in by_id.items():
            fts = data.get("fts_score", 0)
            vector = data.get("vector_score", 0)
            created_at = data.get("created_at", 0)

            combined = self.similarity.combine_scores(fts, vector)
            recency = self.similarity.compute_recency_factor(created_at)
            final_score = self.similarity.apply_recency_boost(combined, created_at)

            result = {**data, "score": final_score}

            if include_scores:
                result["scoreboard"] = {
                    "fts_score": round(fts, 4),
                    "vector_score": round(vector, 4),
                    "combined_score": round(combined, 4),
                    "recency_factor": round(recency, 4),
                    "final_score": round(final_score, 4),
                }

            results.append(result)

        # Sort by final score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    async def search_with_rerank(
        self,
        query: str,
        limit: int = 10,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        since: Optional[float] = None,
        tag_filters: Optional[List[str]] = None,
        stage1_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Two-stage search with reranking.

        Stage 1: Retrieve candidates with hybrid search
        Stage 2: Rerank candidates for precision

        Args:
            query: Search query
            limit: Final result limit
            project: Filter by project
            obs_type: Filter by observation type
            session_id: Filter by session
            date_start: Filter by date range start
            date_end: Filter by date range end
            since: Filter to observations after this timestamp
            tag_filters: Filter by tags
            stage1_limit: Number of candidates for stage 1

        Returns:
            Reranked search results
        """
        candidates_limit = stage1_limit or self.config.search.rerank_top_k or limit * 3

        # Stage 1: Get candidates
        candidates = await self.search(
            query=query,
            limit=candidates_limit,
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=date_start,
            date_end=date_end,
            since=since,
            tag_filters=tag_filters,
            include_scores=True,
        )

        if not candidates:
            return []

        # Stage 2: Rerank
        reranked = await self.reranking.rerank(
            query=query,
            results=candidates,
            top_k=limit,
            embedding_provider=self.embedding_provider,
        )

        return reranked

    async def timeline(
        self,
        anchor_id: str,
        query: Optional[str] = None,
        depth_before: int = 5,
        depth_after: int = 5,
        project: Optional[str] = None,
        obs_type: Optional[str] = None,
        session_id: Optional[str] = None,
        date_start: Optional[float] = None,
        date_end: Optional[float] = None,
        since: Optional[float] = None,
        tag_filters: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get timeline context around an anchor observation.

        Args:
            anchor_id: Central observation ID
            query: Optional query to boost relevant observations
            depth_before: Observations before anchor
            depth_after: Observations after anchor
            project: Filter by project
            obs_type: Filter by observation type
            session_id: Filter by session
            date_start: Filter by date range start
            date_end: Filter by date range end
            since: Filter to observations after this timestamp
            tag_filters: Filter by tags

        Returns:
            Timeline of observations around anchor
        """
        return await self.db.get_timeline(
            anchor_id=anchor_id,
            depth_before=depth_before,
            depth_after=depth_after,
            project=project,
            obs_type=obs_type,
            session_id=session_id,
            date_start=date_start,
            date_end=date_end,
            tag_filters=tag_filters,
        )

    def get_cache_summary(self) -> Dict[str, Any]:
        """Get search cache statistics.

        Returns:
            Cache summary
        """
        return self.cache.get_summary()

    def clear_cache(self) -> int:
        """Clear search cache.

        Returns:
            Number of entries cleared
        """
        return self.cache.clear()

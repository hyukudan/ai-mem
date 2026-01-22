"""Reranking Service - Rerank search results using multiple strategies.

This service provides various reranking algorithms to improve
search result quality after initial retrieval.
"""

import math
from collections import Counter
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..logging_config import get_logger

if TYPE_CHECKING:
    from ..config import AppConfig
    from .similarity import SimilarityService

logger = get_logger("services.reranking")


class RerankingService:
    """Reranks search results using multiple strategies.

    Supports bi-encoder, TF-IDF, and cross-encoder reranking
    to improve search precision.

    Usage:
        service = RerankingService(config, similarity_service)
        reranked = await service.rerank(
            query="search query",
            results=initial_results,
            top_k=10,
            reranker_type="biencoder"
        )
    """

    def __init__(
        self,
        config: "AppConfig",
        similarity_service: "SimilarityService",
    ):
        """Initialize reranking service.

        Args:
            config: Application configuration
            similarity_service: Similarity computation service
        """
        self.config = config
        self.similarity = similarity_service
        self._crossencoder_model = None
        self._crossencoder_model_name: Optional[str] = None

    async def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 10,
        reranker_type: Optional[str] = None,
        embedding_provider: Optional[Any] = None,
        rerank_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank search results.

        Args:
            query: Original search query
            results: Initial search results
            top_k: Number of results to return
            reranker_type: Type of reranker (biencoder, tfidf, crossencoder)
            embedding_provider: Provider for embedding computation
            rerank_weight: Weight for rerank score vs original score

        Returns:
            Reranked results
        """
        if not results:
            return results

        reranker = reranker_type or self.config.search.reranker_type
        weight = rerank_weight if rerank_weight is not None else self.config.search.rerank_weight

        if reranker == "biencoder":
            return await self._rerank_biencoder(
                query, results, top_k, weight, embedding_provider
            )
        elif reranker == "tfidf":
            return self._rerank_tfidf(query, results, top_k, weight)
        elif reranker == "crossencoder":
            return await self._rerank_crossencoder(query, results, top_k, weight)
        else:
            logger.warning(f"Unknown reranker type: {reranker}, returning original")
            return results[:top_k]

    async def _rerank_biencoder(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
        rerank_weight: float,
        embedding_provider: Optional[Any],
    ) -> List[Dict[str, Any]]:
        """Rerank using bi-encoder embeddings.

        Computes query embedding and compares with result embeddings
        using cosine similarity.
        """
        if not embedding_provider:
            logger.warning("No embedding provider for biencoder reranking")
            return results[:top_k]

        try:
            # Get query embedding
            query_embedding = await embedding_provider.embed(query)

            # Score each result
            scored = []
            for result in results:
                # Get or compute result embedding
                result_embedding = result.get("embedding")
                if not result_embedding:
                    text = result.get("content") or result.get("summary", "")
                    if text:
                        result_embedding = await embedding_provider.embed(text)
                    else:
                        result_embedding = None

                if result_embedding:
                    sim = self.similarity.compute_embedding_similarity(
                        query_embedding, result_embedding
                    )
                else:
                    sim = 0.0

                # Combine with original score
                original_score = result.get("score", 0.5)
                combined = (1 - rerank_weight) * original_score + rerank_weight * sim

                scored.append({
                    **result,
                    "rerank_score": sim,
                    "score": combined,
                })

            # Sort by combined score
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]

        except Exception as e:
            logger.error(f"Biencoder reranking failed: {e}")
            return results[:top_k]

    def _rerank_tfidf(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
        rerank_weight: float,
    ) -> List[Dict[str, Any]]:
        """Rerank using TF-IDF similarity.

        Computes term frequency and inverse document frequency
        for query-document matching.
        """
        if not results:
            return results

        # Tokenize query
        query_terms = query.lower().split()
        if not query_terms:
            return results[:top_k]

        # Compute IDF for query terms
        doc_freq: Counter[str] = Counter()
        for result in results:
            text = (result.get("content") or result.get("summary", "")).lower()
            terms = set(text.split())
            for term in query_terms:
                if term in terms:
                    doc_freq[term] += 1

        n_docs = len(results)
        idf = {}
        for term in query_terms:
            df = doc_freq.get(term, 0)
            idf[term] = math.log((n_docs + 1) / (df + 1)) + 1

        # Score each result
        scored = []
        for result in results:
            text = (result.get("content") or result.get("summary", "")).lower()
            words = text.split()
            word_count = len(words)
            term_freq = Counter(words)

            # TF-IDF score
            tfidf_score = 0.0
            for term in query_terms:
                tf = term_freq.get(term, 0) / word_count if word_count > 0 else 0
                tfidf_score += tf * idf.get(term, 0)

            # Normalize
            max_tfidf = sum(idf.values())
            normalized_tfidf = tfidf_score / max_tfidf if max_tfidf > 0 else 0

            # Combine with original score
            original_score = result.get("score", 0.5)
            combined = (1 - rerank_weight) * original_score + rerank_weight * normalized_tfidf

            scored.append({
                **result,
                "rerank_score": normalized_tfidf,
                "score": combined,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    async def _rerank_crossencoder(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
        rerank_weight: float,
    ) -> List[Dict[str, Any]]:
        """Rerank using cross-encoder model.

        Uses a cross-encoder to directly score query-document pairs.
        Falls back to TF-IDF if cross-encoder is unavailable.
        """
        try:
            # Try to load cross-encoder
            model = self._get_crossencoder()
            if model is None:
                logger.info("Cross-encoder not available, falling back to TF-IDF")
                return self._rerank_tfidf(query, results, top_k, rerank_weight)

            # Prepare pairs
            pairs = []
            for result in results:
                text = result.get("content") or result.get("summary", "")
                pairs.append([query, text])

            # Score pairs
            scores = model.predict(pairs)

            # Combine scores
            scored = []
            for i, result in enumerate(results):
                crossencoder_score = float(scores[i])
                # Normalize to 0-1
                normalized = 1 / (1 + math.exp(-crossencoder_score))

                original_score = result.get("score", 0.5)
                combined = (1 - rerank_weight) * original_score + rerank_weight * normalized

                scored.append({
                    **result,
                    "rerank_score": normalized,
                    "score": combined,
                })

            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]

        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return self._rerank_tfidf(query, results, top_k, rerank_weight)

    def _get_crossencoder(self) -> Optional[Any]:
        """Get or load cross-encoder model.

        Returns:
            Cross-encoder model or None if unavailable
        """
        model_name = self.config.search.crossencoder_model

        if self._crossencoder_model is not None:
            if self._crossencoder_model_name == model_name:
                return self._crossencoder_model

        try:
            from sentence_transformers import CrossEncoder
            self._crossencoder_model = CrossEncoder(model_name)
            self._crossencoder_model_name = model_name
            logger.info(f"Loaded cross-encoder: {model_name}")
            return self._crossencoder_model
        except ImportError:
            logger.debug("sentence-transformers not installed, cross-encoder unavailable")
            return None
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder: {e}")
            return None

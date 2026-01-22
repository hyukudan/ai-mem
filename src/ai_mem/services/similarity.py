"""Similarity Service - Compute text and embedding similarities.

This service provides various similarity computation methods
for comparing observations and search results.
"""

import math
import time
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from ..logging_config import get_logger

if TYPE_CHECKING:
    from ..config import AppConfig

logger = get_logger("services.similarity")


class SimilarityService:
    """Computes various similarity metrics.

    Provides text and embedding similarity calculations
    used for search ranking and deduplication.

    Usage:
        service = SimilarityService(config)
        text_sim = service.compute_text_similarity(text1, text2)
        emb_sim = service.compute_embedding_similarity(emb1, emb2)
        combined = service.combine_scores(fts_score, vector_score)
    """

    def __init__(self, config: "AppConfig"):
        """Initialize similarity service.

        Args:
            config: Application configuration
        """
        self.config = config

    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0

        # Tokenize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def compute_embedding_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity between -1 and 1
        """
        if not embedding1 or not embedding2:
            return 0.0

        if len(embedding1) != len(embedding2):
            logger.warning(f"Embedding dimension mismatch: {len(embedding1)} vs {len(embedding2)}")
            return 0.0

        # Compute dot product and norms
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = math.sqrt(sum(a * a for a in embedding1))
        norm2 = math.sqrt(sum(b * b for b in embedding2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def combine_scores(
        self,
        fts_score: float,
        vector_score: float,
        fts_weight: Optional[float] = None,
    ) -> float:
        """Combine FTS and vector scores with weighting.

        Args:
            fts_score: Full-text search BM25 score
            vector_score: Vector similarity score
            fts_weight: Weight for FTS (0-1), remainder goes to vector

        Returns:
            Combined weighted score
        """
        weight = fts_weight if fts_weight is not None else self.config.search.fts_weight

        # Normalize FTS score (BM25 can be > 1)
        normalized_fts = min(fts_score / 10.0, 1.0) if fts_score > 0 else 0.0

        # Vector score is already 0-1
        normalized_vector = max(0.0, min(1.0, vector_score))

        return (weight * normalized_fts) + ((1 - weight) * normalized_vector)

    def compute_recency_factor(
        self,
        created_at: float,
        half_life_days: float = 30.0,
    ) -> float:
        """Compute time-decay factor for recency boosting.

        Args:
            created_at: Unix timestamp of creation
            half_life_days: Days until score decays to 50%

        Returns:
            Decay factor between 0 and 1
        """
        if not created_at:
            return 0.5

        age_seconds = time.time() - created_at
        age_days = age_seconds / 86400.0

        if age_days <= 0:
            return 1.0

        # Exponential decay
        decay = math.exp(-0.693 * age_days / half_life_days)
        return max(0.0, min(1.0, decay))

    def apply_recency_boost(
        self,
        base_score: float,
        created_at: float,
        recency_weight: Optional[float] = None,
    ) -> float:
        """Apply recency boost to a score.

        Args:
            base_score: Original score
            created_at: Unix timestamp
            recency_weight: How much to weight recency (0-1)

        Returns:
            Boosted score
        """
        weight = recency_weight if recency_weight is not None else self.config.search.recency_weight

        if weight <= 0:
            return base_score

        recency = self.compute_recency_factor(created_at)
        return base_score * (1 - weight) + recency * weight

    def find_duplicates(
        self,
        texts: List[str],
        threshold: float = 0.8,
    ) -> List[tuple]:
        """Find duplicate text pairs above similarity threshold.

        Args:
            texts: List of texts to compare
            threshold: Minimum similarity to consider duplicate

        Returns:
            List of (index1, index2, similarity) tuples
        """
        duplicates = []
        n = len(texts)

        for i in range(n):
            for j in range(i + 1, n):
                sim = self.compute_text_similarity(texts[i], texts[j])
                if sim >= threshold:
                    duplicates.append((i, j, sim))

        return sorted(duplicates, key=lambda x: x[2], reverse=True)

    def deduplicate_results(
        self,
        results: List[Dict[str, Any]],
        text_field: str = "content",
        threshold: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """Remove near-duplicate results.

        Args:
            results: Search results to deduplicate
            text_field: Field containing text to compare
            threshold: Similarity threshold for deduplication

        Returns:
            Deduplicated results
        """
        if not results:
            return results

        seen_texts: List[str] = []
        deduplicated: List[Dict[str, Any]] = []

        for result in results:
            text = result.get(text_field, "") or result.get("summary", "")
            if not text:
                deduplicated.append(result)
                continue

            # Check against seen texts
            is_duplicate = False
            for seen in seen_texts:
                if self.compute_text_similarity(text, seen) >= threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen_texts.append(text)
                deduplicated.append(result)

        logger.debug(f"Deduplicated {len(results)} -> {len(deduplicated)} results")
        return deduplicated

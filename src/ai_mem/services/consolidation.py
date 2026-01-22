"""Consolidation Service - Find and consolidate similar observations.

This service handles memory consolidation by identifying
and managing duplicate or similar observations.
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..logging_config import get_logger

if TYPE_CHECKING:
    from ..config import AppConfig
    from ..db import DatabaseManager
    from .similarity import SimilarityService

logger = get_logger("services.consolidation")


class ConsolidationService:
    """Manages memory consolidation.

    Finds similar observations and consolidates them
    to reduce redundancy and improve retrieval quality.

    Usage:
        service = ConsolidationService(db, embedding_provider, similarity, config)
        similar = await service.find_similar(project="my-project", threshold=0.8)
        result = await service.consolidate(project="my-project")
    """

    def __init__(
        self,
        db: "DatabaseManager",
        embedding_provider: Any,
        similarity_service: "SimilarityService",
        config: "AppConfig",
    ):
        """Initialize consolidation service.

        Args:
            db: Database manager
            embedding_provider: Provider for embedding computation
            similarity_service: Similarity computation service
            config: Application configuration
        """
        self.db = db
        self.embedding_provider = embedding_provider
        self.similarity = similarity_service
        self.config = config

    async def find_similar_observations(
        self,
        project: Optional[str] = None,
        similarity_threshold: float = 0.8,
        use_embeddings: bool = True,
        obs_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Find pairs of similar observations.

        Args:
            project: Filter by project
            similarity_threshold: Minimum similarity to consider
            use_embeddings: Use embedding similarity (vs text)
            obs_type: Filter by observation type
            limit: Maximum observations to compare

        Returns:
            List of similar pairs with similarity scores
        """
        # Get observations to compare
        observations = await self.db.export_observations(
            project=project,
            obs_type=obs_type,
            limit=limit,
        )

        if len(observations) < 2:
            return []

        similar_pairs = []

        # Compute embeddings if using semantic similarity
        embeddings = {}
        if use_embeddings and self.embedding_provider:
            for obs in observations:
                try:
                    text = obs.get("content") or obs.get("summary", "")
                    if text:
                        embeddings[obs["id"]] = await self.embedding_provider.embed(text)
                except Exception as e:
                    logger.debug(f"Failed to embed observation {obs['id']}: {e}")

        # Compare pairs
        n = len(observations)
        for i in range(n):
            for j in range(i + 1, n):
                obs1 = observations[i]
                obs2 = observations[j]

                # Compute similarity
                if use_embeddings and obs1["id"] in embeddings and obs2["id"] in embeddings:
                    sim = self.similarity.compute_embedding_similarity(
                        embeddings[obs1["id"]],
                        embeddings[obs2["id"]],
                    )
                else:
                    text1 = obs1.get("content") or obs1.get("summary", "")
                    text2 = obs2.get("content") or obs2.get("summary", "")
                    sim = self.similarity.compute_text_similarity(text1, text2)

                if sim >= similarity_threshold:
                    similar_pairs.append({
                        "observation_1": {
                            "id": obs1["id"],
                            "summary": obs1.get("summary", "")[:100],
                            "created_at": obs1.get("created_at"),
                            "type": obs1.get("type"),
                        },
                        "observation_2": {
                            "id": obs2["id"],
                            "summary": obs2.get("summary", "")[:100],
                            "created_at": obs2.get("created_at"),
                            "type": obs2.get("type"),
                        },
                        "similarity": round(sim, 4),
                        "similarity_type": "embedding" if use_embeddings else "text",
                    })

        # Sort by similarity
        similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_pairs

    async def consolidate_memories(
        self,
        project: Optional[str] = None,
        similarity_threshold: float = 0.85,
        keep_strategy: str = "newest",
        obs_type: Optional[str] = None,
        dry_run: bool = False,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Consolidate similar memories.

        Args:
            project: Filter by project
            similarity_threshold: Minimum similarity to consolidate
            keep_strategy: Which to keep: "newest", "oldest", "importance"
            obs_type: Filter by observation type
            dry_run: If True, don't actually modify
            limit: Maximum observations to process

        Returns:
            Consolidation results
        """
        similar_pairs = await self.find_similar_observations(
            project=project,
            similarity_threshold=similarity_threshold,
            obs_type=obs_type,
            limit=limit,
        )

        if not similar_pairs:
            return {
                "consolidated": 0,
                "pairs_found": 0,
                "dry_run": dry_run,
            }

        # Track which observations to mark as superseded
        superseded = []
        kept = []

        # Process each pair
        processed_ids = set()
        for pair in similar_pairs:
            obs1_id = pair["observation_1"]["id"]
            obs2_id = pair["observation_2"]["id"]

            # Skip if already processed
            if obs1_id in processed_ids or obs2_id in processed_ids:
                continue

            # Decide which to keep
            keeper_id, superseded_id = self._select_keeper(
                pair["observation_1"],
                pair["observation_2"],
                keep_strategy,
            )

            superseded.append({
                "id": superseded_id,
                "superseded_by": keeper_id,
                "similarity": pair["similarity"],
            })
            kept.append(keeper_id)

            processed_ids.add(obs1_id)
            processed_ids.add(obs2_id)

        # Apply changes if not dry run
        if not dry_run and superseded:
            for item in superseded:
                await self.db.mark_superseded(
                    obs_id=item["id"],
                    superseded_by=item["superseded_by"],
                )

        return {
            "consolidated": len(superseded),
            "pairs_found": len(similar_pairs),
            "kept": kept,
            "superseded": superseded,
            "dry_run": dry_run,
            "strategy": keep_strategy,
        }

    def _select_keeper(
        self,
        obs1: Dict[str, Any],
        obs2: Dict[str, Any],
        strategy: str,
    ) -> Tuple[str, str]:
        """Select which observation to keep.

        Args:
            obs1: First observation
            obs2: Second observation
            strategy: Selection strategy

        Returns:
            Tuple of (keeper_id, superseded_id)
        """
        if strategy == "newest":
            # Keep the newer one
            if (obs1.get("created_at") or 0) >= (obs2.get("created_at") or 0):
                return obs1["id"], obs2["id"]
            return obs2["id"], obs1["id"]

        elif strategy == "oldest":
            # Keep the older one
            if (obs1.get("created_at") or 0) <= (obs2.get("created_at") or 0):
                return obs1["id"], obs2["id"]
            return obs2["id"], obs1["id"]

        elif strategy == "importance":
            # Keep the more important one
            if (obs1.get("importance_score") or 0.5) >= (obs2.get("importance_score") or 0.5):
                return obs1["id"], obs2["id"]
            return obs2["id"], obs1["id"]

        else:
            # Default to newest
            return self._select_keeper(obs1, obs2, "newest")

    async def cleanup_stale_memories(
        self,
        project: Optional[str] = None,
        max_age_days: int = 90,
        min_access_count: int = 0,
        dry_run: bool = False,
        limit: int = 500,
    ) -> Dict[str, Any]:
        """Clean up old, unused memories.

        Args:
            project: Filter by project
            max_age_days: Maximum age in days
            min_access_count: Minimum access count to keep
            dry_run: If True, don't actually delete
            limit: Maximum to process

        Returns:
            Cleanup results
        """
        import time

        cutoff_time = time.time() - (max_age_days * 86400)

        # Find stale observations
        stale = await self.db.find_stale_observations(
            project=project,
            older_than=cutoff_time,
            min_access_count=min_access_count,
            limit=limit,
        )

        if not stale:
            return {
                "deleted": 0,
                "candidates": 0,
                "dry_run": dry_run,
            }

        deleted = 0
        if not dry_run:
            for obs in stale:
                try:
                    await self.db.delete_observation(obs["id"])
                    deleted += 1
                except Exception as e:
                    logger.error(f"Failed to delete observation {obs['id']}: {e}")

        return {
            "deleted": deleted,
            "candidates": len(stale),
            "dry_run": dry_run,
            "max_age_days": max_age_days,
        }

    async def get_consolidation_stats(
        self,
        project: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get statistics about potential consolidation.

        Args:
            project: Filter by project

        Returns:
            Statistics about duplicate potential
        """
        # Sample observations for analysis
        similar_80 = await self.find_similar_observations(
            project=project,
            similarity_threshold=0.8,
            limit=50,
        )

        similar_90 = await self.find_similar_observations(
            project=project,
            similarity_threshold=0.9,
            limit=50,
        )

        return {
            "project": project,
            "pairs_above_80_percent": len(similar_80),
            "pairs_above_90_percent": len(similar_90),
            "potential_savings": len(similar_90),  # Could remove these
            "recommendation": (
                "High duplicate potential - consider consolidation"
                if len(similar_90) > 10
                else "Low duplicate potential"
            ),
        }

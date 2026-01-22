"""Indexing Service - Handle vector indexing of observations.

This service manages the indexing of observations into
the vector store for semantic search.
"""

import hashlib
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..logging_config import get_logger

if TYPE_CHECKING:
    from ..config import AppConfig
    from ..models import Observation

logger = get_logger("services.indexing")


class IndexingService:
    """Handles vector indexing of observations.

    Manages chunking, embedding generation, and vector store
    insertion for semantic search capabilities.

    Usage:
        service = IndexingService(config)
        await service.index_observation(obs, embedding_provider, vector_store)
        await service.index_batch(observations, embedding_provider, vector_store)
    """

    def __init__(self, config: "AppConfig"):
        """Initialize indexing service.

        Args:
            config: Application configuration
        """
        self.config = config
        self._chunk_size = getattr(config, 'chunk_size', 512)
        self._chunk_overlap = getattr(config, 'chunk_overlap', 50)

    async def index_observation(
        self,
        observation: "Observation",
        embedding_provider: Any,
        vector_store: Any,
    ) -> bool:
        """Index a single observation.

        Args:
            observation: Observation to index
            embedding_provider: Provider for generating embeddings
            vector_store: Vector store for storage

        Returns:
            True if indexing succeeded
        """
        if not embedding_provider or not vector_store:
            logger.debug("Skipping indexing: no embedding provider or vector store")
            return False

        try:
            # Get text to embed
            text = self._get_indexable_text(observation)
            if not text:
                logger.debug(f"No text to index for observation: {observation.id}")
                return False

            # Chunk if needed
            chunks = self._chunk_text(text)

            # Generate embeddings
            for i, chunk in enumerate(chunks):
                embedding = await embedding_provider.embed(chunk)

                # Create chunk ID
                chunk_id = f"{observation.id}:{i}" if len(chunks) > 1 else observation.id

                # Store in vector store
                metadata = {
                    "observation_id": observation.id,
                    "project": observation.project,
                    "session_id": observation.session_id,
                    "type": observation.type,
                    "created_at": observation.created_at,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }

                await vector_store.add(
                    id=chunk_id,
                    embedding=embedding,
                    metadata=metadata,
                    text=chunk,
                )

            logger.debug(f"Indexed observation {observation.id} ({len(chunks)} chunks)")
            return True

        except Exception as e:
            logger.error(f"Failed to index observation {observation.id}: {e}")
            return False

    async def index_batch(
        self,
        observations: List["Observation"],
        embedding_provider: Any,
        vector_store: Any,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Index multiple observations in batch.

        Args:
            observations: Observations to index
            embedding_provider: Provider for generating embeddings
            vector_store: Vector store for storage
            batch_size: Number of embeddings to generate at once

        Returns:
            Dictionary with indexing statistics
        """
        if not observations:
            return {"indexed": 0, "failed": 0, "skipped": 0}

        indexed = 0
        failed = 0
        skipped = 0

        # Prepare all texts and metadata
        items = []
        for obs in observations:
            text = self._get_indexable_text(obs)
            if not text:
                skipped += 1
                continue

            chunks = self._chunk_text(text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{obs.id}:{i}" if len(chunks) > 1 else obs.id
                items.append({
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": {
                        "observation_id": obs.id,
                        "project": obs.project,
                        "session_id": obs.session_id,
                        "type": obs.type,
                        "created_at": obs.created_at,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                })

        # Process in batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            texts = [item["text"] for item in batch]

            try:
                # Batch embed
                if hasattr(embedding_provider, 'embed_batch'):
                    embeddings = await embedding_provider.embed_batch(texts)
                else:
                    embeddings = [await embedding_provider.embed(t) for t in texts]

                # Store in vector store
                for item, embedding in zip(batch, embeddings):
                    await vector_store.add(
                        id=item["id"],
                        embedding=embedding,
                        metadata=item["metadata"],
                        text=item["text"],
                    )
                    indexed += 1

            except Exception as e:
                logger.error(f"Batch indexing failed: {e}")
                failed += len(batch)

        logger.info(f"Batch indexing complete: {indexed} indexed, {failed} failed, {skipped} skipped")
        return {"indexed": indexed, "failed": failed, "skipped": skipped}

    async def delete_from_index(
        self,
        observation_id: str,
        vector_store: Any,
    ) -> bool:
        """Remove an observation from the index.

        Args:
            observation_id: ID of observation to remove
            vector_store: Vector store

        Returns:
            True if deletion succeeded
        """
        try:
            # Delete main ID and any chunks
            await vector_store.delete(observation_id)

            # Also try to delete chunks (0-99 should cover all cases)
            for i in range(100):
                chunk_id = f"{observation_id}:{i}"
                try:
                    await vector_store.delete(chunk_id)
                except Exception:
                    break  # No more chunks

            logger.debug(f"Deleted observation from index: {observation_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete from index: {e}")
            return False

    async def rebuild_index(
        self,
        observations: List["Observation"],
        embedding_provider: Any,
        vector_store: Any,
        project: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Rebuild the vector index for observations.

        Args:
            observations: Observations to reindex
            embedding_provider: Provider for generating embeddings
            vector_store: Vector store
            project: If specified, clear only this project's vectors

        Returns:
            Indexing statistics
        """
        # Clear existing vectors
        if project:
            await vector_store.delete_by_metadata({"project": project})
        else:
            await vector_store.clear()

        # Reindex all
        return await self.index_batch(observations, embedding_provider, vector_store)

    def _get_indexable_text(self, observation: "Observation") -> str:
        """Get text to index from an observation.

        Args:
            observation: Observation

        Returns:
            Text to embed
        """
        parts = []

        if observation.title:
            parts.append(observation.title)

        if observation.content:
            parts.append(observation.content)
        elif observation.summary:
            parts.append(observation.summary)

        if observation.tags:
            parts.append(" ".join(observation.tags))

        return " ".join(parts)

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for embedding.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if len(text) <= self._chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self._chunk_size

            # Try to break at word boundary
            if end < len(text):
                # Look for last space in chunk
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            chunks.append(text[start:end].strip())
            start = end - self._chunk_overlap

        return [c for c in chunks if c]

    def compute_content_hash(self, content: str) -> str:
        """Compute hash for deduplication.

        Args:
            content: Content to hash

        Returns:
            SHA-256 hex digest
        """
        return hashlib.sha256(content.encode()).hexdigest()

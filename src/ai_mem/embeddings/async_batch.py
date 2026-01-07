"""Async Batch Embedding Support.

This module provides async batching capabilities for embedding providers,
enabling parallel processing of embedding requests for improved performance.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Tuple
from dataclasses import dataclass, field
import time

from .base import EmbeddingProvider


@dataclass
class BatchConfig:
    """Configuration for batch embedding."""
    max_batch_size: int = 32  # Max texts per batch
    max_concurrent_batches: int = 4  # Max parallel batches
    timeout_seconds: float = 60.0  # Timeout per batch
    retry_count: int = 2  # Retries on failure
    retry_delay_seconds: float = 1.0  # Delay between retries


@dataclass
class BatchResult:
    """Result of a batch embedding operation."""
    embeddings: List[List[float]]
    batch_count: int
    total_texts: int
    elapsed_seconds: float
    errors: List[str] = field(default_factory=list)


class AsyncBatchEmbedder:
    """Async wrapper for batch embedding with parallelism.

    Wraps a synchronous EmbeddingProvider and provides:
    - Automatic batching of large text lists
    - Parallel execution of batches
    - Error handling and retries
    - Performance metrics

    Example:
        provider = FastEmbedProvider()
        batch_embedder = AsyncBatchEmbedder(provider)

        # Embed many texts in parallel batches
        texts = ["text1", "text2", ..., "text1000"]
        embeddings = await batch_embedder.embed_batch(texts)
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        config: Optional[BatchConfig] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        """Initialize async batch embedder.

        Args:
            provider: Base embedding provider (sync)
            config: Batch configuration
            executor: Thread pool executor (created if not provided)
        """
        self.provider = provider
        self.config = config or BatchConfig()
        self._executor = executor
        self._owns_executor = executor is None

    @property
    def executor(self) -> ThreadPoolExecutor:
        """Get or create thread pool executor."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.max_concurrent_batches
            )
        return self._executor

    def _split_into_batches(self, texts: List[str]) -> List[List[str]]:
        """Split texts into batches of max_batch_size."""
        batches = []
        for i in range(0, len(texts), self.config.max_batch_size):
            batch = texts[i:i + self.config.max_batch_size]
            batches.append(batch)
        return batches

    def _embed_batch_sync(self, batch: List[str]) -> Tuple[List[List[float]], Optional[str]]:
        """Embed a single batch synchronously with retry logic."""
        last_error = None

        for attempt in range(self.config.retry_count + 1):
            try:
                embeddings = self.provider.embed(batch)
                return embeddings, None
            except Exception as e:
                last_error = str(e)
                if attempt < self.config.retry_count:
                    time.sleep(self.config.retry_delay_seconds)

        return [], last_error

    async def _embed_batch_async(
        self,
        batch: List[str],
        batch_idx: int,
    ) -> Tuple[int, List[List[float]], Optional[str]]:
        """Embed a batch asynchronously using thread executor."""
        loop = asyncio.get_event_loop()

        try:
            embeddings, error = await asyncio.wait_for(
                loop.run_in_executor(
                    self.executor,
                    self._embed_batch_sync,
                    batch,
                ),
                timeout=self.config.timeout_seconds,
            )
            return batch_idx, embeddings, error
        except asyncio.TimeoutError:
            return batch_idx, [], f"Timeout after {self.config.timeout_seconds}s"
        except Exception as e:
            return batch_idx, [], str(e)

    async def embed_batch(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult:
        """Embed texts in parallel batches.

        Args:
            texts: List of texts to embed
            progress_callback: Optional callback(completed, total) for progress

        Returns:
            BatchResult with embeddings and metrics
        """
        if not texts:
            return BatchResult(
                embeddings=[],
                batch_count=0,
                total_texts=0,
                elapsed_seconds=0.0,
            )

        start_time = time.perf_counter()
        batches = self._split_into_batches(texts)
        batch_count = len(batches)

        # Process batches with concurrency limit
        semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)

        async def process_with_semaphore(batch: List[str], idx: int):
            async with semaphore:
                result = await self._embed_batch_async(batch, idx)
                if progress_callback:
                    progress_callback(idx + 1, batch_count)
                return result

        # Launch all batch tasks
        tasks = [
            process_with_semaphore(batch, idx)
            for idx, batch in enumerate(batches)
        ]

        # Gather results
        results = await asyncio.gather(*tasks)

        # Reconstruct embeddings in order
        ordered_results: List[Tuple[int, List[List[float]], Optional[str]]] = sorted(
            results, key=lambda x: x[0]
        )

        all_embeddings: List[List[float]] = []
        errors: List[str] = []

        for batch_idx, embeddings, error in ordered_results:
            if error:
                errors.append(f"Batch {batch_idx}: {error}")
                # Fill with empty embeddings for failed batch
                batch_size = len(batches[batch_idx])
                all_embeddings.extend([[]] * batch_size)
            else:
                all_embeddings.extend(embeddings)

        elapsed = time.perf_counter() - start_time

        return BatchResult(
            embeddings=all_embeddings,
            batch_count=batch_count,
            total_texts=len(texts),
            elapsed_seconds=elapsed,
            errors=errors,
        )

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous batch embedding (blocks until complete).

        Convenience method for sync contexts. Runs async embed_batch
        in a new event loop.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        async def _run():
            return await self.embed_batch(texts)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, need to run in thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _run())
                    result = future.result()
            else:
                result = loop.run_until_complete(_run())
        except RuntimeError:
            # No event loop, create one
            result = asyncio.run(_run())

        return result.embeddings

    def close(self):
        """Close the executor if we own it."""
        if self._owns_executor and self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class CachedBatchEmbedder(AsyncBatchEmbedder):
    """Batch embedder with caching to avoid re-computing embeddings.

    Useful when processing observations that may have been indexed before.
    Uses content hash as cache key.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        config: Optional[BatchConfig] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        max_cache_size: int = 10000,
    ):
        super().__init__(provider, config, executor)
        self._cache: dict[str, List[float]] = {}
        self._max_cache_size = max_cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    def _hash_text(self, text: str) -> str:
        """Create hash key for text."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()

    async def embed_batch(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult:
        """Embed with caching - only compute uncached embeddings."""
        if not texts:
            return BatchResult(
                embeddings=[],
                batch_count=0,
                total_texts=0,
                elapsed_seconds=0.0,
            )

        start_time = time.perf_counter()

        # Check cache
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []
        cached_embeddings: dict[int, List[float]] = {}

        for i, text in enumerate(texts):
            key = self._hash_text(text)
            if key in self._cache:
                cached_embeddings[i] = self._cache[key]
                self._cache_hits += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                self._cache_misses += 1

        # Embed uncached texts
        if uncached_texts:
            result = await super().embed_batch(uncached_texts, progress_callback)

            # Update cache
            for idx, (text, embedding) in enumerate(zip(uncached_texts, result.embeddings)):
                key = self._hash_text(text)
                if len(self._cache) < self._max_cache_size:
                    self._cache[key] = embedding
                original_idx = uncached_indices[idx]
                cached_embeddings[original_idx] = embedding

        # Reconstruct in order
        all_embeddings = [cached_embeddings.get(i, []) for i in range(len(texts))]

        elapsed = time.perf_counter() - start_time

        return BatchResult(
            embeddings=all_embeddings,
            batch_count=len(self._split_into_batches(uncached_texts)) if uncached_texts else 0,
            total_texts=len(texts),
            elapsed_seconds=elapsed,
        )

    @property
    def cache_stats(self) -> dict:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
            "max_cache_size": self._max_cache_size,
        }

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

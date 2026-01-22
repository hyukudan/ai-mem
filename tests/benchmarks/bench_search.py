"""Search Benchmarks - Measure search operation performance.

Benchmarks for:
- FTS (Full-Text Search) latency
- Vector search latency
- Hybrid search latency
- Search cache effectiveness
- Reranking overhead
"""

import asyncio
import statistics
import time
from typing import Any, Dict, List, Optional

import pytest


class SearchBenchmarks:
    """Benchmarks for search operations."""

    def __init__(self):
        self.results: Dict[str, List[float]] = {}

    def record(self, operation: str, duration_ms: float) -> None:
        """Record a benchmark result."""
        if operation not in self.results:
            self.results[operation] = []
        self.results[operation].append(duration_ms)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get benchmark summary statistics."""
        summary = {}
        for op, times in self.results.items():
            if times:
                summary[op] = {
                    "count": len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "mean_ms": statistics.mean(times),
                    "median_ms": statistics.median(times),
                    "p95_ms": sorted(times)[int(len(times) * 0.95)] if len(times) >= 20 else max(times),
                    "p99_ms": sorted(times)[int(len(times) * 0.99)] if len(times) >= 100 else max(times),
                }
        return summary


@pytest.fixture
def benchmarks():
    """Create benchmark tracker."""
    return SearchBenchmarks()


class TestFTSBenchmarks:
    """Full-Text Search benchmarks."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_fts_single_word_query(self, benchmarks, populated_db):
        """Benchmark single word FTS query."""
        db = populated_db
        queries = ["python", "javascript", "database", "memory", "search"]

        for query in queries:
            for _ in range(10):
                start = time.perf_counter()
                await db.search_observations_fts(query=query, limit=10)
                duration_ms = (time.perf_counter() - start) * 1000
                benchmarks.record("fts_single_word", duration_ms)

        summary = benchmarks.summary()
        assert summary["fts_single_word"]["mean_ms"] < 50, "FTS should be < 50ms"

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_fts_phrase_query(self, benchmarks, populated_db):
        """Benchmark phrase FTS query."""
        db = populated_db
        queries = [
            "python programming",
            "memory management",
            "search optimization",
            "database indexing",
        ]

        for query in queries:
            for _ in range(10):
                start = time.perf_counter()
                await db.search_observations_fts(query=query, limit=10)
                duration_ms = (time.perf_counter() - start) * 1000
                benchmarks.record("fts_phrase", duration_ms)

        summary = benchmarks.summary()
        assert summary["fts_phrase"]["mean_ms"] < 100, "FTS phrase should be < 100ms"

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_fts_with_filters(self, benchmarks, populated_db):
        """Benchmark FTS with project and type filters."""
        db = populated_db

        for _ in range(20):
            start = time.perf_counter()
            await db.search_observations_fts(
                query="search",
                limit=10,
                project="test-project",
                obs_type="note",
            )
            duration_ms = (time.perf_counter() - start) * 1000
            benchmarks.record("fts_filtered", duration_ms)

        summary = benchmarks.summary()
        assert summary["fts_filtered"]["mean_ms"] < 100, "Filtered FTS should be < 100ms"


class TestHybridSearchBenchmarks:
    """Hybrid search (FTS + Vector) benchmarks."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_hybrid_search_latency(self, benchmarks, search_service):
        """Benchmark hybrid search end-to-end latency."""
        queries = [
            "python async patterns",
            "memory optimization techniques",
            "search ranking algorithms",
        ]

        for query in queries:
            for _ in range(5):
                start = time.perf_counter()
                await search_service.search(query=query, limit=10)
                duration_ms = (time.perf_counter() - start) * 1000
                benchmarks.record("hybrid_search", duration_ms)

        summary = benchmarks.summary()
        # Hybrid search should be reasonable even without vector store
        assert summary["hybrid_search"]["mean_ms"] < 200, "Hybrid search should be < 200ms"

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_search_cache_effectiveness(self, benchmarks, search_service):
        """Benchmark cache hit vs cache miss."""
        query = "python programming patterns"

        # First search (cache miss)
        start = time.perf_counter()
        await search_service.search(query=query, limit=10)
        miss_ms = (time.perf_counter() - start) * 1000
        benchmarks.record("cache_miss", miss_ms)

        # Subsequent searches (cache hits)
        for _ in range(10):
            start = time.perf_counter()
            await search_service.search(query=query, limit=10)
            hit_ms = (time.perf_counter() - start) * 1000
            benchmarks.record("cache_hit", hit_ms)

        summary = benchmarks.summary()
        # Cache hits should be significantly faster
        if summary["cache_miss"]["mean_ms"] > 0:
            speedup = summary["cache_miss"]["mean_ms"] / summary["cache_hit"]["mean_ms"]
            assert speedup > 2, "Cache should provide at least 2x speedup"


class TestRerankingBenchmarks:
    """Reranking operation benchmarks."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_tfidf_reranking(self, benchmarks, reranking_service):
        """Benchmark TF-IDF reranking."""
        # Mock results
        results = [
            {"id": f"obs-{i}", "content": f"Sample content about topic {i}", "score": 0.5}
            for i in range(50)
        ]

        for _ in range(10):
            start = time.perf_counter()
            reranking_service._rerank_tfidf(
                query="sample topic content",
                results=results,
                top_k=10,
                rerank_weight=0.5,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            benchmarks.record("tfidf_rerank", duration_ms)

        summary = benchmarks.summary()
        assert summary["tfidf_rerank"]["mean_ms"] < 50, "TF-IDF rerank should be < 50ms"


class TestScalabilityBenchmarks:
    """Benchmarks for scalability testing."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_search_scaling_with_result_count(self, benchmarks, populated_db):
        """Benchmark search time vs result limit."""
        query = "test"
        limits = [10, 50, 100, 500]

        for limit in limits:
            for _ in range(5):
                start = time.perf_counter()
                await populated_db.search_observations_fts(query=query, limit=limit)
                duration_ms = (time.perf_counter() - start) * 1000
                benchmarks.record(f"fts_limit_{limit}", duration_ms)

        summary = benchmarks.summary()

        # Verify reasonable scaling
        for limit in limits:
            key = f"fts_limit_{limit}"
            if key in summary:
                # Allow some overhead but should scale sub-linearly
                assert summary[key]["mean_ms"] < limit * 2, f"Limit {limit} took too long"


# =============================================================================
# Fixtures for benchmarks
# =============================================================================

@pytest.fixture
async def populated_db(temp_db_path):
    """Create a database populated with test data."""
    from ai_mem.db import DatabaseManager
    import random

    db = DatabaseManager(temp_db_path)
    await db.initialize()
    await db.create_tables()

    # Populate with test data
    types = ["note", "bug", "feature", "discovery"]
    projects = ["project-a", "project-b", "project-c"]
    words = [
        "python", "javascript", "rust", "database", "memory",
        "search", "optimization", "algorithm", "async", "pattern",
        "architecture", "testing", "deployment", "monitoring", "logging",
    ]

    for i in range(1000):
        content = " ".join(random.sample(words, 5)) + f" observation {i}"
        await db.add_observation(
            obs_id=f"obs-{i}",
            session_id=f"session-{i % 10}",
            project=random.choice(projects),
            obs_type=random.choice(types),
            content=content,
            summary=content[:100],
            created_at=time.time() - random.randint(0, 86400 * 30),
            tags=[random.choice(words), random.choice(words)],
        )

    yield db
    await db.close()


@pytest.fixture
def search_service(populated_db):
    """Create search service with populated DB."""
    from ai_mem.services.cache import SearchCacheService
    from ai_mem.services.similarity import SimilarityService
    from ai_mem.services.reranking import RerankingService
    from ai_mem.services.search import SearchService
    from ai_mem.config import AppConfig

    config = AppConfig()
    cache_service = SearchCacheService(config)
    similarity_service = SimilarityService(config)
    reranking_service = RerankingService(config, similarity_service)

    return SearchService(
        populated_db, None, None, config,
        cache_service, similarity_service, reranking_service
    )


@pytest.fixture
def reranking_service():
    """Create reranking service."""
    from ai_mem.services.similarity import SimilarityService
    from ai_mem.services.reranking import RerankingService
    from ai_mem.config import AppConfig

    config = AppConfig()
    similarity_service = SimilarityService(config)
    return RerankingService(config, similarity_service)


@pytest.fixture
def temp_db_path():
    """Create temporary database path."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    try:
        os.unlink(f.name)
    except OSError:
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "benchmark"])

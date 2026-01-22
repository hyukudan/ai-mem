"""Database Benchmarks - Measure database operation performance.

Benchmarks for:
- Observation insertion (single and batch)
- Observation retrieval
- Tag operations
- Statistics computation
"""

import asyncio
import statistics
import time
from typing import Any, Dict, List
import uuid

import pytest


class DatabaseBenchmarks:
    """Benchmarks for database operations."""

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
                    "min_ms": round(min(times), 2),
                    "max_ms": round(max(times), 2),
                    "mean_ms": round(statistics.mean(times), 2),
                    "median_ms": round(statistics.median(times), 2),
                    "total_ms": round(sum(times), 2),
                }
        return summary

    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("DATABASE BENCHMARK RESULTS")
        print("=" * 60)
        for op, stats in self.summary().items():
            print(f"\n{op}:")
            for metric, value in stats.items():
                print(f"  {metric}: {value}")


@pytest.fixture
def benchmarks():
    """Create benchmark tracker."""
    return DatabaseBenchmarks()


@pytest.fixture
async def db_manager(temp_db_path):
    """Create database manager."""
    from ai_mem.db import DatabaseManager

    db = DatabaseManager(temp_db_path)
    await db.initialize()
    await db.create_tables()
    yield db
    await db.close()


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


class TestInsertionBenchmarks:
    """Benchmarks for observation insertion."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_single_insert_latency(self, benchmarks, db_manager):
        """Benchmark single observation insert."""
        for i in range(100):
            obs_id = str(uuid.uuid4())
            session_id = f"session-{i % 10}"

            start = time.perf_counter()
            await db_manager.add_observation(
                obs_id=obs_id,
                session_id=session_id,
                project="bench-project",
                obs_type="note",
                content=f"Benchmark observation {i} with some content",
                summary=f"Summary {i}",
                created_at=time.time(),
                tags=["benchmark", "test"],
            )
            duration_ms = (time.perf_counter() - start) * 1000
            benchmarks.record("single_insert", duration_ms)

        summary = benchmarks.summary()
        print(f"\nSingle insert: {summary['single_insert']}")
        assert summary["single_insert"]["mean_ms"] < 10, "Single insert should be < 10ms"

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_batch_insert_throughput(self, benchmarks, db_manager):
        """Benchmark batch observation insert."""
        batch_sizes = [10, 50, 100]

        for batch_size in batch_sizes:
            observations = []
            for i in range(batch_size):
                observations.append({
                    "obs_id": str(uuid.uuid4()),
                    "session_id": f"session-{i % 5}",
                    "project": "bench-project",
                    "obs_type": "note",
                    "content": f"Batch observation {i}",
                    "summary": f"Summary {i}",
                    "created_at": time.time(),
                    "tags": ["batch", "benchmark"],
                })

            start = time.perf_counter()
            for obs in observations:
                await db_manager.add_observation(**obs)
            duration_ms = (time.perf_counter() - start) * 1000

            benchmarks.record(f"batch_insert_{batch_size}", duration_ms)
            per_item_ms = duration_ms / batch_size
            benchmarks.record("batch_insert_per_item", per_item_ms)

        summary = benchmarks.summary()
        print(f"\nBatch insert per item: {summary.get('batch_insert_per_item', {})}")


class TestRetrievalBenchmarks:
    """Benchmarks for observation retrieval."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_single_retrieval(self, benchmarks, populated_db):
        """Benchmark single observation retrieval."""
        # Get some IDs to retrieve
        obs_ids = [f"obs-{i}" for i in range(100)]

        for obs_id in obs_ids:
            start = time.perf_counter()
            await populated_db.get_observation(obs_id)
            duration_ms = (time.perf_counter() - start) * 1000
            benchmarks.record("single_retrieval", duration_ms)

        summary = benchmarks.summary()
        print(f"\nSingle retrieval: {summary['single_retrieval']}")
        assert summary["single_retrieval"]["mean_ms"] < 5, "Single retrieval should be < 5ms"

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_batch_retrieval(self, benchmarks, populated_db):
        """Benchmark batch observation retrieval."""
        batch_sizes = [10, 50, 100]

        for batch_size in batch_sizes:
            obs_ids = [f"obs-{i}" for i in range(batch_size)]

            start = time.perf_counter()
            await populated_db.get_observations(obs_ids)
            duration_ms = (time.perf_counter() - start) * 1000

            benchmarks.record(f"batch_retrieval_{batch_size}", duration_ms)

        summary = benchmarks.summary()
        for key in summary:
            if key.startswith("batch_retrieval"):
                print(f"\n{key}: {summary[key]}")


class TestQueryBenchmarks:
    """Benchmarks for query operations."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_list_observations(self, benchmarks, populated_db):
        """Benchmark listing observations with filters."""
        for _ in range(20):
            start = time.perf_counter()
            await populated_db.list_observations(
                project="project-a",
                limit=100,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            benchmarks.record("list_with_project", duration_ms)

        for _ in range(20):
            start = time.perf_counter()
            await populated_db.list_observations(
                obs_type="note",
                limit=100,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            benchmarks.record("list_with_type", duration_ms)

        summary = benchmarks.summary()
        print(f"\nList with project: {summary['list_with_project']}")
        print(f"List with type: {summary['list_with_type']}")

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_stats_computation(self, benchmarks, populated_db):
        """Benchmark statistics computation."""
        for _ in range(10):
            start = time.perf_counter()
            await populated_db.get_stats(
                project="project-a",
                tag_limit=20,
                day_limit=30,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            benchmarks.record("get_stats", duration_ms)

        summary = benchmarks.summary()
        print(f"\nStats computation: {summary['get_stats']}")
        assert summary["get_stats"]["mean_ms"] < 500, "Stats should be < 500ms"


class TestTagBenchmarks:
    """Benchmarks for tag operations."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_tag_counts(self, benchmarks, populated_db):
        """Benchmark tag count retrieval."""
        for _ in range(20):
            start = time.perf_counter()
            await populated_db.get_tag_counts(limit=50)
            duration_ms = (time.perf_counter() - start) * 1000
            benchmarks.record("tag_counts", duration_ms)

        summary = benchmarks.summary()
        print(f"\nTag counts: {summary['tag_counts']}")

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_tag_update(self, benchmarks, populated_db):
        """Benchmark tag update operations."""
        for i in range(10):
            start = time.perf_counter()
            await populated_db.add_tag(
                tag=f"new-tag-{i}",
                project="project-a",
            )
            duration_ms = (time.perf_counter() - start) * 1000
            benchmarks.record("tag_add", duration_ms)

        summary = benchmarks.summary()
        print(f"\nTag add: {summary['tag_add']}")


@pytest.fixture
async def populated_db(temp_db_path):
    """Create populated database."""
    from ai_mem.db import DatabaseManager
    import random

    db = DatabaseManager(temp_db_path)
    await db.initialize()
    await db.create_tables()

    # Add sessions
    for i in range(10):
        await db.add_session(
            session_id=f"session-{i}",
            project=f"project-{chr(97 + i % 3)}",
            start_time=time.time() - 86400 * (10 - i),
        )

    # Add observations
    types = ["note", "bug", "feature", "discovery"]
    projects = ["project-a", "project-b", "project-c"]
    words = ["python", "javascript", "database", "memory", "search", "test"]

    for i in range(1000):
        content = " ".join(random.sample(words, 3)) + f" content {i}"
        await db.add_observation(
            obs_id=f"obs-{i}",
            session_id=f"session-{i % 10}",
            project=random.choice(projects),
            obs_type=random.choice(types),
            content=content,
            summary=content[:50],
            created_at=time.time() - random.randint(0, 86400 * 30),
            tags=[random.choice(words)],
        )

    yield db
    await db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "benchmark", "-s"])

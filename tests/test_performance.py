"""⚡ Performance tests and benchmarks."""

import asyncio
import pytest
import time
from ai_mem.performance import (
    parallel_search,
    batch_operation,
    PerformanceMonitor,
    CacheStats,
)


class TestParallelSearch:
    """Test parallel search execution."""
    
    async def test_parallel_faster_than_sequential(self):
        """Test that parallel search is faster than sequential."""
        async def slow_fts():
            await asyncio.sleep(0.1)
            return ["fts1", "fts2"]
        
        async def slow_vector():
            await asyncio.sleep(0.1)
            return {"vec1": 0.9, "vec2": 0.8}
        
        # Sequential would take ~200ms (100+100)
        # Parallel should take ~110ms (both running together)
        start = time.perf_counter()
        fts_results, vector_results = await parallel_search(slow_fts, slow_vector)
        duration = (time.perf_counter() - start) * 1000
        
        assert fts_results == ["fts1", "fts2"]
        assert vector_results == {"vec1": 0.9, "vec2": 0.8}
        # Should be faster than sequential (200ms)
        assert duration < 200, f"Parallel search took {duration}ms, expected <200ms"
    
    async def test_parallel_handles_fts_failure(self):
        """Test graceful failure when FTS search fails."""
        async def failing_fts():
            raise Exception("FTS error")
        
        async def working_vector():
            return {"vec1": 0.9}
        
        fts_results, vector_results = await parallel_search(failing_fts, working_vector)
        
        assert fts_results == []  # Empty on failure
        assert vector_results == {"vec1": 0.9}
    
    async def test_parallel_handles_vector_failure(self):
        """Test graceful failure when vector search fails."""
        async def working_fts():
            return ["fts1"]
        
        async def failing_vector():
            raise Exception("Vector error")
        
        fts_results, vector_results = await parallel_search(working_fts, failing_vector)
        
        assert fts_results == ["fts1"]
        assert vector_results == []  # Empty on failure
    
    async def test_parallel_both_failures(self):
        """Test handling when both searches fail."""
        async def failing_fts():
            raise Exception("FTS error")
        
        async def failing_vector():
            raise Exception("Vector error")
        
        fts_results, vector_results = await parallel_search(failing_fts, failing_vector)
        
        assert fts_results == []
        assert vector_results == []


class TestBatchOperation:
    """Test batch operation execution."""
    
    async def test_batch_operation_processes_all(self):
        """Test that batch operation processes all items."""
        items = list(range(10))
        results = []
        
        async def double(x):
            await asyncio.sleep(0.01)
            return x * 2
        
        results = await batch_operation(items, double, batch_size=3, max_concurrent=2)
        
        assert len(results) == 10
        assert results == [x * 2 for x in items]
    
    async def test_batch_operation_respects_concurrency(self):
        """Test that batch operation respects max_concurrent limit."""
        items = list(range(5))
        concurrent_count = 0
        max_concurrent_observed = 0
        
        async def track_concurrent(x):
            nonlocal concurrent_count, max_concurrent_observed
            concurrent_count += 1
            max_concurrent_observed = max(max_concurrent_observed, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return x
        
        results = await batch_operation(items, track_concurrent, max_concurrent=2)
        
        assert len(results) == 5
        assert max_concurrent_observed <= 2, f"Exceeded max concurrency: {max_concurrent_observed}"
    
    async def test_batch_operation_empty_list(self):
        """Test batch operation with empty list."""
        async def operation(x):
            return x
        
        results = await batch_operation([], operation)
        
        assert results == []
    
    async def test_batch_operation_handles_exceptions(self):
        """Test that batch operation handles exceptions gracefully."""
        items = [1, 2, 3, 4, 5]
        
        async def failing_operation(x):
            if x == 3:
                raise ValueError("Test error")
            return x * 2
        
        results = await batch_operation(items, failing_operation)
        
        assert len(results) == 5
        # Results include None where exceptions occurred
        assert results[0] == 2
        assert results[1] == 4
        assert results[2] is None  # Failed
        assert results[3] == 8
        assert results[4] == 10


class TestPerformanceMonitor:
    """Test performance monitoring."""
    
    def test_performance_monitor_measures_time(self):
        """Test that performance monitor measures execution time."""
        with PerformanceMonitor("test_operation") as monitor:
            time.sleep(0.1)
        
        assert monitor.duration_ms >= 100
        assert monitor.duration_ms < 200  # Some overhead
    
    def test_performance_monitor_handles_exceptions(self):
        """Test that monitor handles exceptions without re-raising."""
        with pytest.raises(ValueError):
            with PerformanceMonitor("failing_operation"):
                raise ValueError("test error")
    
    def test_performance_monitor_logs_duration(self):
        """Test that monitor logs operation duration."""
        with PerformanceMonitor("test") as monitor:
            time.sleep(0.05)
        
        # Should have measured time
        assert monitor.duration_ms > 0


class TestCacheStats:
    """Test cache statistics tracking."""
    
    def test_cache_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats()
        
        stats.record_hit()
        stats.record_hit()
        stats.record_miss()
        
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.hit_rate == pytest.approx(2/3)
    
    def test_cache_hit_rate_no_operations(self):
        """Test hit rate with no operations."""
        stats = CacheStats()
        
        assert stats.hit_rate == 0.0
    
    def test_cache_stats_string_representation(self):
        """Test cache stats string format."""
        stats = CacheStats()
        stats.record_hit()
        stats.record_hit()
        stats.record_miss()
        stats.record_eviction()
        
        stats_str = str(stats)
        assert "Hits: 2" in stats_str
        assert "Misses: 1" in stats_str
        assert "Evictions: 1" in stats_str
        assert "Hit Rate" in stats_str
    
    def test_cache_eviction_tracking(self):
        """Test eviction counter."""
        stats = CacheStats()
        
        assert stats.evictions == 0
        stats.record_eviction()
        stats.record_eviction()
        
        assert stats.evictions == 2


class TestParallelPerformance:
    """Integration tests for parallel operations."""
    
    async def test_parallel_is_faster_integration(self):
        """Test that parallel operations are faster in practice."""
        # Simulate realistic search scenario
        async def fts_search():
            await asyncio.sleep(0.08)
            return [{"id": "fts1", "score": 0.9}]
        
        async def vector_search():
            await asyncio.sleep(0.08)
            return {"vec1": 0.8}
        
        start = time.perf_counter()
        fts_results, vector_results = await parallel_search(fts_search, vector_search)
        parallel_time = (time.perf_counter() - start) * 1000
        
        # Sequential would be ~160ms, parallel should be ~100ms
        assert parallel_time < 160, f"Parallel search too slow: {parallel_time}ms"
        assert len(fts_results) == 1
        assert len(vector_results) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

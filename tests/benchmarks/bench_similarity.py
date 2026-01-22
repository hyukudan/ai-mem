"""Similarity Benchmarks - Measure similarity computation performance.

Benchmarks for:
- Text similarity (Jaccard)
- Embedding similarity (Cosine)
- Score combination
- Batch similarity computation
"""

import random
import statistics
import time
from typing import Dict, List

import pytest


class SimilarityBenchmarks:
    """Benchmarks for similarity computations."""

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
                    "min_ms": round(min(times), 4),
                    "max_ms": round(max(times), 4),
                    "mean_ms": round(statistics.mean(times), 4),
                    "ops_per_sec": round(len(times) / (sum(times) / 1000), 2) if sum(times) > 0 else 0,
                }
        return summary


@pytest.fixture
def benchmarks():
    """Create benchmark tracker."""
    return SimilarityBenchmarks()


@pytest.fixture
def similarity_service():
    """Create similarity service."""
    from ai_mem.services.similarity import SimilarityService
    from ai_mem.config import AppConfig

    config = AppConfig()
    return SimilarityService(config)


@pytest.fixture
def sample_texts():
    """Generate sample texts for benchmarking."""
    words = [
        "python", "programming", "language", "code", "development",
        "software", "engineering", "algorithm", "data", "structure",
        "function", "class", "method", "variable", "loop",
        "condition", "exception", "module", "package", "library",
    ]

    texts = []
    for _ in range(100):
        length = random.randint(10, 50)
        text = " ".join(random.choices(words, k=length))
        texts.append(text)

    return texts


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for benchmarking."""
    embeddings = []
    for _ in range(100):
        # 384-dimensional embeddings (typical for sentence-transformers)
        embedding = [random.uniform(-1, 1) for _ in range(384)]
        embeddings.append(embedding)
    return embeddings


class TestTextSimilarityBenchmarks:
    """Benchmarks for text similarity."""

    @pytest.mark.benchmark
    def test_jaccard_similarity(self, benchmarks, similarity_service, sample_texts):
        """Benchmark Jaccard similarity computation."""
        for i in range(len(sample_texts) - 1):
            text1 = sample_texts[i]
            text2 = sample_texts[i + 1]

            start = time.perf_counter()
            similarity_service.compute_text_similarity(text1, text2)
            duration_ms = (time.perf_counter() - start) * 1000
            benchmarks.record("jaccard_similarity", duration_ms)

        summary = benchmarks.summary()
        print(f"\nJaccard similarity: {summary['jaccard_similarity']}")
        # Should be very fast
        assert summary["jaccard_similarity"]["mean_ms"] < 1, "Jaccard should be < 1ms"

    @pytest.mark.benchmark
    def test_batch_text_similarity(self, benchmarks, similarity_service, sample_texts):
        """Benchmark batch text similarity (n² comparisons)."""
        n = 20  # Compare first 20 texts

        start = time.perf_counter()
        for i in range(n):
            for j in range(i + 1, n):
                similarity_service.compute_text_similarity(
                    sample_texts[i],
                    sample_texts[j]
                )
        duration_ms = (time.perf_counter() - start) * 1000

        comparisons = n * (n - 1) // 2
        per_comparison_ms = duration_ms / comparisons

        benchmarks.record("batch_text_total", duration_ms)
        benchmarks.record("batch_text_per_pair", per_comparison_ms)

        print(f"\nBatch text similarity ({comparisons} pairs): {duration_ms:.2f}ms total, {per_comparison_ms:.4f}ms/pair")


class TestEmbeddingSimilarityBenchmarks:
    """Benchmarks for embedding similarity."""

    @pytest.mark.benchmark
    def test_cosine_similarity(self, benchmarks, similarity_service, sample_embeddings):
        """Benchmark cosine similarity computation."""
        for i in range(len(sample_embeddings) - 1):
            emb1 = sample_embeddings[i]
            emb2 = sample_embeddings[i + 1]

            start = time.perf_counter()
            similarity_service.compute_embedding_similarity(emb1, emb2)
            duration_ms = (time.perf_counter() - start) * 1000
            benchmarks.record("cosine_similarity", duration_ms)

        summary = benchmarks.summary()
        print(f"\nCosine similarity: {summary['cosine_similarity']}")
        # Should be fast but slightly slower than Jaccard due to math
        assert summary["cosine_similarity"]["mean_ms"] < 1, "Cosine should be < 1ms"

    @pytest.mark.benchmark
    def test_batch_embedding_similarity(self, benchmarks, similarity_service, sample_embeddings):
        """Benchmark batch embedding similarity."""
        n = 20

        start = time.perf_counter()
        for i in range(n):
            for j in range(i + 1, n):
                similarity_service.compute_embedding_similarity(
                    sample_embeddings[i],
                    sample_embeddings[j]
                )
        duration_ms = (time.perf_counter() - start) * 1000

        comparisons = n * (n - 1) // 2
        per_comparison_ms = duration_ms / comparisons

        benchmarks.record("batch_embedding_total", duration_ms)
        benchmarks.record("batch_embedding_per_pair", per_comparison_ms)

        print(f"\nBatch embedding similarity ({comparisons} pairs): {duration_ms:.2f}ms total, {per_comparison_ms:.4f}ms/pair")


class TestScoreCombiningBenchmarks:
    """Benchmarks for score combining operations."""

    @pytest.mark.benchmark
    def test_score_combining(self, benchmarks, similarity_service):
        """Benchmark score combination."""
        for _ in range(1000):
            fts_score = random.uniform(0, 10)
            vector_score = random.uniform(0, 1)

            start = time.perf_counter()
            similarity_service.combine_scores(fts_score, vector_score)
            duration_ms = (time.perf_counter() - start) * 1000
            benchmarks.record("combine_scores", duration_ms)

        summary = benchmarks.summary()
        print(f"\nScore combining: {summary['combine_scores']}")

    @pytest.mark.benchmark
    def test_recency_factor(self, benchmarks, similarity_service):
        """Benchmark recency factor computation."""
        timestamps = [time.time() - random.randint(0, 86400 * 30) for _ in range(1000)]

        for ts in timestamps:
            start = time.perf_counter()
            similarity_service.compute_recency_factor(ts)
            duration_ms = (time.perf_counter() - start) * 1000
            benchmarks.record("recency_factor", duration_ms)

        summary = benchmarks.summary()
        print(f"\nRecency factor: {summary['recency_factor']}")


class TestDeduplicationBenchmarks:
    """Benchmarks for deduplication operations."""

    @pytest.mark.benchmark
    def test_find_duplicates(self, benchmarks, similarity_service, sample_texts):
        """Benchmark duplicate finding."""
        n = 50

        start = time.perf_counter()
        duplicates = similarity_service.find_duplicates(
            sample_texts[:n],
            threshold=0.5,
        )
        duration_ms = (time.perf_counter() - start) * 1000

        benchmarks.record("find_duplicates", duration_ms)
        print(f"\nFind duplicates ({n} texts): {duration_ms:.2f}ms, found {len(duplicates)} pairs")

    @pytest.mark.benchmark
    def test_deduplicate_results(self, benchmarks, similarity_service, sample_texts):
        """Benchmark result deduplication."""
        results = [
            {"id": f"obs-{i}", "content": text}
            for i, text in enumerate(sample_texts[:50])
        ]

        start = time.perf_counter()
        deduped = similarity_service.deduplicate_results(
            results,
            threshold=0.9,
        )
        duration_ms = (time.perf_counter() - start) * 1000

        benchmarks.record("deduplicate_results", duration_ms)
        print(f"\nDeduplicate results: {len(results)} -> {len(deduped)} in {duration_ms:.2f}ms")


class TestScalabilityBenchmarks:
    """Benchmarks for scaling behavior."""

    @pytest.mark.benchmark
    def test_similarity_scaling(self, benchmarks, similarity_service):
        """Measure how similarity computation scales with input size."""
        sizes = [10, 50, 100, 200, 500]

        for size in sizes:
            # Generate texts of different sizes
            text1 = " ".join(["word"] * size)
            text2 = " ".join(["word"] * size)

            start = time.perf_counter()
            for _ in range(100):
                similarity_service.compute_text_similarity(text1, text2)
            duration_ms = (time.perf_counter() - start) * 1000

            benchmarks.record(f"text_sim_size_{size}", duration_ms / 100)

        summary = benchmarks.summary()
        print("\nText similarity scaling:")
        for size in sizes:
            key = f"text_sim_size_{size}"
            if key in summary:
                print(f"  {size} words: {summary[key]['mean_ms']:.4f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "benchmark", "-s"])

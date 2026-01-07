"""Tests for vector store implementations."""

import os
import shutil
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from ai_mem.vector_stores.base import VectorStoreProvider
from ai_mem.vector_stores.chroma import ChromaVectorStore


class TestVectorStoreProviderBase:
    """Test the base class interface."""

    def test_add_not_implemented(self):
        provider = VectorStoreProvider()
        with pytest.raises(NotImplementedError):
            provider.add([], [], [], [])

    def test_query_not_implemented(self):
        provider = VectorStoreProvider()
        with pytest.raises(NotImplementedError):
            provider.query([], 10)

    def test_delete_ids_not_implemented(self):
        provider = VectorStoreProvider()
        with pytest.raises(NotImplementedError):
            provider.delete_ids([])

    def test_delete_where_not_implemented(self):
        provider = VectorStoreProvider()
        with pytest.raises(NotImplementedError):
            provider.delete_where({})


class TestChromaVectorStore:
    """Tests for the ChromaDB vector store implementation."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for Chroma storage."""
        dir_path = tempfile.mkdtemp(prefix="ai_mem_test_chroma_")
        yield dir_path
        shutil.rmtree(dir_path, ignore_errors=True)

    @pytest.fixture
    def store(self, temp_dir):
        """Create a ChromaVectorStore instance."""
        return ChromaVectorStore(path=temp_dir, collection_name="test_collection")

    def test_init_creates_collection(self, temp_dir):
        """Test that initialization creates or gets the collection."""
        store = ChromaVectorStore(path=temp_dir, collection_name="test_init")
        assert store.collection is not None
        assert store.client is not None

    def test_add_empty_list(self, store):
        """Test adding empty lists does nothing."""
        # Should not raise
        store.add(
            embeddings=[],
            documents=[],
            metadatas=[],
            ids=[],
        )

    def test_add_single_document(self, store):
        """Test adding a single document."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        store.add(
            embeddings=[embedding],
            documents=["Test document"],
            metadatas=[{"project": "test"}],
            ids=["doc-1"],
        )
        # Verify by querying
        result = store.query(embedding=embedding, n_results=1)
        assert len(result["ids"][0]) == 1
        assert result["ids"][0][0] == "doc-1"

    def test_add_multiple_documents(self, store):
        """Test adding multiple documents."""
        embeddings = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.4, 0.5, 0.6, 0.7],
        ]
        store.add(
            embeddings=embeddings,
            documents=["Doc 1", "Doc 2", "Doc 3"],
            metadatas=[
                {"type": "note"},
                {"type": "code"},
                {"type": "note"},
            ],
            ids=["doc-1", "doc-2", "doc-3"],
        )
        # Query and verify
        result = store.query(embedding=embeddings[0], n_results=3)
        assert len(result["ids"][0]) == 3

    def test_query_returns_nearest(self, store):
        """Test that query returns documents in order of similarity."""
        # Add documents with distinct embeddings
        store.add(
            embeddings=[
                [1.0, 0.0, 0.0, 0.0, 0.0],  # doc-1
                [0.0, 1.0, 0.0, 0.0, 0.0],  # doc-2
                [0.0, 0.0, 1.0, 0.0, 0.0],  # doc-3
            ],
            documents=["Doc 1", "Doc 2", "Doc 3"],
            metadatas=[{"idx": "1"}, {"idx": "2"}, {"idx": "3"}],
            ids=["doc-1", "doc-2", "doc-3"],
        )
        # Query with embedding similar to doc-1
        result = store.query(embedding=[0.9, 0.1, 0.0, 0.0, 0.0], n_results=3)
        # First result should be doc-1 (most similar)
        assert result["ids"][0][0] == "doc-1"

    def test_query_with_n_results_limit(self, store):
        """Test that query respects n_results limit."""
        store.add(
            embeddings=[
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.2, 0.3, 0.4, 0.5, 0.6],
                [0.3, 0.4, 0.5, 0.6, 0.7],
            ],
            documents=["Doc 1", "Doc 2", "Doc 3"],
            metadatas=[{"idx": "1"}, {"idx": "2"}, {"idx": "3"}],
            ids=["doc-1", "doc-2", "doc-3"],
        )
        result = store.query(embedding=[0.1, 0.2, 0.3, 0.4, 0.5], n_results=2)
        assert len(result["ids"][0]) == 2

    def test_query_with_where_filter(self, store):
        """Test that query respects where filter."""
        store.add(
            embeddings=[
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.1, 0.2, 0.3, 0.4, 0.5],
            ],
            documents=["Note 1", "Code 1", "Note 2"],
            metadatas=[
                {"type": "note"},
                {"type": "code"},
                {"type": "note"},
            ],
            ids=["doc-1", "doc-2", "doc-3"],
        )
        # Query only notes
        result = store.query(
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            n_results=10,
            where={"type": "note"},
        )
        # Should only get the 2 notes
        assert len(result["ids"][0]) == 2
        for metadata in result["metadatas"][0]:
            assert metadata["type"] == "note"

    def test_delete_ids_empty(self, store):
        """Test deleting empty list does nothing."""
        store.delete_ids([])  # Should not raise

    def test_delete_ids_single(self, store):
        """Test deleting a single document by ID."""
        store.add(
            embeddings=[[0.1, 0.2, 0.3, 0.4, 0.5]],
            documents=["Test doc"],
            metadatas=[{"idx": "1"}],
            ids=["doc-1"],
        )
        store.delete_ids(["doc-1"])
        # Verify deletion
        result = store.query(embedding=[0.1, 0.2, 0.3, 0.4, 0.5], n_results=10)
        assert len(result["ids"][0]) == 0

    def test_delete_ids_multiple(self, store):
        """Test deleting multiple documents by ID."""
        store.add(
            embeddings=[
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.2, 0.3, 0.4, 0.5, 0.6],
                [0.3, 0.4, 0.5, 0.6, 0.7],
            ],
            documents=["Doc 1", "Doc 2", "Doc 3"],
            metadatas=[{"idx": "1"}, {"idx": "2"}, {"idx": "3"}],
            ids=["doc-1", "doc-2", "doc-3"],
        )
        store.delete_ids(["doc-1", "doc-3"])
        # Only doc-2 should remain
        result = store.query(embedding=[0.2, 0.3, 0.4, 0.5, 0.6], n_results=10)
        assert len(result["ids"][0]) == 1
        assert result["ids"][0][0] == "doc-2"

    def test_delete_where_empty(self, store):
        """Test deleting with empty where does nothing."""
        store.delete_where({})  # Should not raise

    def test_delete_where_filter(self, store):
        """Test deleting documents matching a filter."""
        store.add(
            embeddings=[
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.2, 0.3, 0.4, 0.5, 0.6],
                [0.3, 0.4, 0.5, 0.6, 0.7],
            ],
            documents=["Note 1", "Code 1", "Note 2"],
            metadatas=[
                {"type": "note"},
                {"type": "code"},
                {"type": "note"},
            ],
            ids=["doc-1", "doc-2", "doc-3"],
        )
        # Delete all notes
        store.delete_where({"type": "note"})
        # Only code document should remain
        result = store.query(embedding=[0.2, 0.3, 0.4, 0.5, 0.6], n_results=10)
        assert len(result["ids"][0]) == 1
        assert result["metadatas"][0][0]["type"] == "code"

    def test_query_result_structure(self, store):
        """Test that query returns expected structure."""
        store.add(
            embeddings=[[0.1, 0.2, 0.3, 0.4, 0.5]],
            documents=["Test doc"],
            metadatas=[{"key": "value"}],
            ids=["doc-1"],
        )
        result = store.query(embedding=[0.1, 0.2, 0.3, 0.4, 0.5], n_results=1)
        # Check structure
        assert "ids" in result
        assert "metadatas" in result
        assert "distances" in result
        # Check types
        assert isinstance(result["ids"], list)
        assert isinstance(result["ids"][0], list)
        assert isinstance(result["metadatas"][0], list)

    def test_collection_persistence(self, temp_dir):
        """Test that data persists across instances."""
        # Create store and add data
        store1 = ChromaVectorStore(path=temp_dir, collection_name="persist_test")
        store1.add(
            embeddings=[[0.1, 0.2, 0.3, 0.4, 0.5]],
            documents=["Persistent doc"],
            metadatas=[{"persisted": True}],
            ids=["persist-1"],
        )
        # Create new store instance with same path
        store2 = ChromaVectorStore(path=temp_dir, collection_name="persist_test")
        # Data should still be there
        result = store2.query(embedding=[0.1, 0.2, 0.3, 0.4, 0.5], n_results=1)
        assert len(result["ids"][0]) == 1
        assert result["ids"][0][0] == "persist-1"


class TestPGVectorStoreMocked:
    """Tests for PGVector store using mocks (no actual postgres connection)."""

    def test_instantiation_with_mocks(self):
        """Test that PGVectorStore can be instantiated with mocked dependencies."""
        with patch("ai_mem.vector_stores.pgvector.psycopg.connect"), \
             patch("ai_mem.vector_stores.pgvector.register_vector"), \
             patch("ai_mem.vector_stores.pgvector.ConnectionPool"):

            from ai_mem.vector_stores.pgvector import PGVectorStore

            store = PGVectorStore(
                dsn="postgresql://test:test@localhost/test",
                table="test_vectors",
                dimension=3,
            )

            assert store is not None
            assert store._table_name == "test_vectors"
            assert store._dimension == 3

    def test_init_parameters(self):
        """Test that initialization stores correct parameters."""
        with patch("ai_mem.vector_stores.pgvector.psycopg.connect"), \
             patch("ai_mem.vector_stores.pgvector.register_vector"), \
             patch("ai_mem.vector_stores.pgvector.ConnectionPool"):

            from ai_mem.vector_stores.pgvector import PGVectorStore

            store = PGVectorStore(
                dsn="postgresql://user:pass@localhost/mydb",
                table="custom_table",
                dimension=768,
                index_type="hnsw",
                lists=150,
            )

            assert store._table_name == "custom_table"
            assert store._dimension == 768
            assert store._index_type == "hnsw"
            assert store._lists == 150


class TestQdrantStoreMocked:
    """Tests for Qdrant store using mocks (no actual qdrant connection)."""

    def test_add_calls_upsert(self):
        """Test that add calls the qdrant client."""
        with patch("ai_mem.vector_stores.qdrant.QdrantClient") as mock_client, \
             patch("ai_mem.vector_stores.qdrant.models") as mock_models:

            from ai_mem.vector_stores.qdrant import QdrantVectorStore

            # Setup mocks
            mock_models.Distance.COSINE = "Cosine"
            mock_models.VectorParams = MagicMock()
            mock_models.PointStruct = MagicMock(side_effect=lambda id, vector, payload: {
                "id": id, "vector": vector, "payload": payload
            })

            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.get_collections.return_value.collections = []

            store = QdrantVectorStore(
                url="http://localhost:6333",
                collection_name="test_collection",
            )

            # Add data
            store.add(
                embeddings=[[0.1, 0.2, 0.3]],
                documents=["Test"],
                metadatas=[{}],
                ids=["test-1"],
            )

            # Verify upsert was called
            mock_instance.upsert.assert_called_once()

    def test_query_calls_search(self):
        """Test that query calls the qdrant client."""
        with patch("ai_mem.vector_stores.qdrant.QdrantClient") as mock_client, \
             patch("ai_mem.vector_stores.qdrant.models") as mock_models:

            from ai_mem.vector_stores.qdrant import QdrantVectorStore

            # Setup mocks
            mock_models.Distance.COSINE = "Cosine"
            mock_models.VectorParams = MagicMock()

            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.get_collections.return_value.collections = []
            mock_instance.search.return_value = []

            store = QdrantVectorStore(
                url="http://localhost:6333",
                collection_name="test_collection",
            )

            # Query
            result = store.query(embedding=[0.1, 0.2, 0.3], n_results=10)

            # Verify search was called
            mock_instance.search.assert_called_once()
            assert "ids" in result
            assert "metadatas" in result
            assert "distances" in result

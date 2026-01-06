import pytest
from unittest.mock import MagicMock, patch
from ai_mem.config import AppConfig, StorageConfig, VectorConfig
from ai_mem.vector_store import build_vector_store
from ai_mem.vector_stores.chroma import ChromaVectorStore
from ai_mem.vector_stores.pgvector import PGVectorStore
from ai_mem.vector_stores.qdrant import QdrantVectorStore

@pytest.fixture
def mock_storage():
    storage = MagicMock(spec=StorageConfig)
    storage.vector_dir = "/tmp/vector"
    return storage

def test_build_chroma(mock_storage):
    config = AppConfig()
    config.vector.provider = "chroma"
    
    with patch("ai_mem.vector_stores.chroma.chromadb.PersistentClient"):
        store = build_vector_store(config, mock_storage)
        assert isinstance(store, ChromaVectorStore)

def test_build_pgvector(mock_storage):
    config = AppConfig()
    config.vector.provider = "pgvector"
    config.vector.pgvector_dsn = "postgresql://user:pass@localhost:5432/db"

    with patch("ai_mem.vector_stores.pgvector.psycopg.connect"), \
         patch("ai_mem.vector_stores.pgvector.register_vector"), \
         patch("ai_mem.vector_stores.pgvector.ConnectionPool"):
        store = build_vector_store(config, mock_storage)
        assert isinstance(store, PGVectorStore)

def test_build_qdrant(mock_storage):
    config = AppConfig()
    config.vector.provider = "qdrant"
    config.vector.qdrant_url = "http://localhost:6333"

    # Mock QdrantClient to avoid import errors or connection attempts
    # We also need to mock 'models' since it's None if import fails
    with patch("ai_mem.vector_stores.qdrant.QdrantClient") as mock_client, \
         patch("ai_mem.vector_stores.qdrant.models") as mock_models:
        
        # Setup mock models
        mock_models.Distance.COSINE = "Cosine"
        mock_models.VectorParams = MagicMock()
        mock_models.PointStruct = MagicMock()
        
        # Mock get_collections
        mock_client.return_value.get_collections.return_value.collections = []
        
        store = build_vector_store(config, mock_storage)
        assert isinstance(store, QdrantVectorStore)

def test_invalid_provider(mock_storage):
    config = AppConfig()
    config.vector.provider = "invalid_provider"
    
    with pytest.raises(ValueError, match="Unsupported vector provider"):
        build_vector_store(config, mock_storage)


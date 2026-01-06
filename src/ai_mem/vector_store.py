from .config import AppConfig, StorageConfig
from .vector_stores.base import VectorStoreProvider as BaseVectorStore
from .vector_stores.chroma import ChromaVectorStore
from .vector_stores.pgvector import PGVectorStore

# Try to import Qdrant, if available
try:
    from .vector_stores.qdrant import QdrantVectorStore
except ImportError:
    QdrantVectorStore = None


def build_vector_store(config: AppConfig, storage: StorageConfig) -> BaseVectorStore:
    provider = config.vector.provider.lower()
    
    if provider == "chroma":
        collection_name = config.vector.chroma_collection or "observations"
        return ChromaVectorStore(path=storage.vector_dir, collection_name=collection_name)
    
    if provider in {"pgvector", "postgres", "postgresql"}:
        dsn = config.vector.pgvector_dsn
        if not dsn:
            raise ValueError("pgvector provider requires AI_MEM_PGVECTOR_DSN or pgvector_dsn in config.")
        return PGVectorStore(
            dsn=dsn,
            table=config.vector.pgvector_table or "ai_mem_vectors",
            dimension=config.vector.pgvector_dimension,
            index_type=config.vector.pgvector_index_type,
            lists=config.vector.pgvector_lists,
        )

    if provider == "qdrant":
        if QdrantVectorStore is None:
             raise ValueError("Qdrant provider requires 'qdrant-client' to be installed. Please run `pip install qdrant-client`.")
        
        url = config.vector.qdrant_url
        if not url:
            raise ValueError("Qdrant provider requires qdrant_url in config.")
            
        return QdrantVectorStore(
            url=url,
            api_key=config.vector.qdrant_api_key,
            collection_name=config.vector.qdrant_collection or "observations",
            vector_size=config.vector.qdrant_vector_size or 1536, # Default to OpenAI size
        )

    raise ValueError(f"Unsupported vector provider: {config.vector.provider}")

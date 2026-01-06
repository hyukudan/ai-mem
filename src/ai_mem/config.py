import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, ConfigDict


class LLMConfig(BaseModel):
    provider: str = "none"
    model: str = "local-model"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    timeout_s: float = 60.0


class EmbeddingConfig(BaseModel):
    provider: str = "fastembed"
    model: str = "BAAI/bge-small-en-v1.5"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None


class StorageConfig(BaseModel):
    data_dir: str = "~/.ai-mem"
    sqlite_path: Optional[str] = None
    vector_dir: Optional[str] = None


class SearchConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 100
    vector_top_k: int = 20
    fts_top_k: int = 20


class VectorConfig(BaseModel):
    provider: str = "chroma"
    chroma_collection: str = "observations"
    pgvector_dsn: Optional[str] = None
    pgvector_table: str = "ai_mem_vectors"
    pgvector_dimension: int = 1536
    pgvector_index_type: str = "ivfflat"
    pgvector_lists: int = 100


class ContextConfig(BaseModel):
    total_observation_count: int = 12
    full_observation_count: int = 4
    observation_types: list[str] = Field(default_factory=list)
    tag_filters: list[str] = Field(default_factory=list)
    full_observation_field: str = "content"
    show_token_estimates: bool = True
    wrap_context_tag: bool = True


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    vector: VectorConfig = Field(default_factory=VectorConfig)


def _config_path() -> Path:
    env_path = os.environ.get("AI_MEM_CONFIG")
    if env_path:
        return Path(env_path).expanduser()

    config_root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_root / "ai-mem" / "config.json"


def _legacy_config_path() -> Path:
    return Path.home() / ".ai_mem_config.json"


def load_config() -> AppConfig:
    path = _config_path()
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        config = AppConfig.model_validate(data)
        return _apply_env_overrides(config)

    legacy_path = _legacy_config_path()
    if legacy_path.exists():
        with legacy_path.open("r", encoding="utf-8") as handle:
            legacy = json.load(handle)
        migrated = {
            "llm": {
                "provider": legacy.get("provider", "gemini"),
                "api_key": legacy.get("api_key"),
            }
        }
        config = AppConfig.model_validate(migrated)
        return _apply_env_overrides(config)

    return _apply_env_overrides(AppConfig())


def save_config(config: AppConfig) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config.model_dump(), handle, indent=2)


def update_config(patch: Dict[str, Any]) -> AppConfig:
    config = load_config()
    merged = config.model_dump()
    for key, value in patch.items():
        if isinstance(value, dict) and key in merged:
            merged[key].update(value)
        else:
            merged[key] = value
    updated = AppConfig.model_validate(merged)
    save_config(updated)
    return updated


def resolve_storage_paths(config: AppConfig) -> StorageConfig:
    storage = config.storage.model_copy(deep=True)
    data_dir = Path(storage.data_dir).expanduser()
    storage.data_dir = str(data_dir)
    if not storage.sqlite_path:
        storage.sqlite_path = str(data_dir / "ai-mem.sqlite")
    if not storage.vector_dir:
        storage.vector_dir = str(data_dir / "vector-db")
    return storage


def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_bool(name: str) -> Optional[bool]:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _env_list(name: str) -> Optional[list[str]]:
    value = os.environ.get(name)
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _apply_env_overrides(config: AppConfig) -> AppConfig:
    context_total = _env_int("AI_MEM_CONTEXT_TOTAL")
    context_full = _env_int("AI_MEM_CONTEXT_FULL")
    context_types = _env_list("AI_MEM_CONTEXT_TYPES")
    context_tags = _env_list("AI_MEM_CONTEXT_TAGS")
    context_full_field = os.environ.get("AI_MEM_CONTEXT_FULL_FIELD")
    context_show_tokens = _env_bool("AI_MEM_CONTEXT_SHOW_TOKENS")
    context_wrap = _env_bool("AI_MEM_CONTEXT_WRAP")

    if context_total is not None:
        config.context.total_observation_count = context_total
    if context_full is not None:
        config.context.full_observation_count = context_full
    if context_types is not None:
        config.context.observation_types = context_types
    if context_tags is not None:
        config.context.tag_filters = context_tags
    if context_full_field:
        config.context.full_observation_field = context_full_field
    if context_show_tokens is not None:
        config.context.show_token_estimates = context_show_tokens
    if context_wrap is not None:
        config.context.wrap_context_tag = context_wrap

    vector_provider = os.environ.get("AI_MEM_VECTOR_PROVIDER")
    vector_collection = os.environ.get("AI_MEM_VECTOR_CHROMA_COLLECTION")
    pgvector_dsn = os.environ.get("AI_MEM_PGVECTOR_DSN")
    pgvector_table = os.environ.get("AI_MEM_PGVECTOR_TABLE")
    pgvector_dimension = _env_int("AI_MEM_PGVECTOR_DIMENSION")
    pgvector_index_type = os.environ.get("AI_MEM_PGVECTOR_INDEX_TYPE")
    pgvector_lists = _env_int("AI_MEM_PGVECTOR_LISTS")

    if vector_provider:
        config.vector.provider = vector_provider
    if vector_collection:
        config.vector.chroma_collection = vector_collection
    if pgvector_dsn:
        config.vector.pgvector_dsn = pgvector_dsn
    if pgvector_table:
        config.vector.pgvector_table = pgvector_table
    if pgvector_dimension is not None:
        config.vector.pgvector_dimension = pgvector_dimension
    if pgvector_index_type:
        config.vector.pgvector_index_type = pgvector_index_type
    if pgvector_lists is not None:
        config.vector.pgvector_lists = pgvector_lists

    return config

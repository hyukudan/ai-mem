import json
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from pydantic import BaseModel, Field, ConfigDict

from .exceptions import ConfigurationError, InvalidConfigError, MissingConfigError


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
    cache_ttl_seconds: int = 30
    cache_max_entries: int = 256
    fts_weight: float = 0.5
    vector_weight: float = 0.5
    recency_half_life_hours: float = 48.0
    recency_weight: float = 0.1


class VectorConfig(BaseModel):
    provider: str = "chroma"
    chroma_collection: str = "observations"
    pgvector_dsn: Optional[str] = None
    pgvector_table: str = "ai_mem_vectors"
    pgvector_dimension: int = 1536
    pgvector_index_type: str = "ivfflat"
    pgvector_lists: int = 100
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_collection: Optional[str] = "observations"
    qdrant_vector_size: int = 1536


class ContextConfig(BaseModel):
    total_observation_count: int = 12
    full_observation_count: int = 4
    observation_types: list[str] = Field(default_factory=list)
    tag_filters: list[str] = Field(default_factory=list)
    full_observation_field: str = "content"
    show_token_estimates: bool = True
    wrap_context_tag: bool = True


class IngestionConfig(BaseModel):
    """Configuration for event ingestion and filtering."""

    # Tools to skip entirely (exact match)
    skip_tool_names: list[str] = Field(
        default_factory=lambda: [
            "SlashCommand",
            "Skill",
            "TodoWrite",
            "TodoRead",
            "AskFollowupQuestion",
            "AttemptCompletion",
        ]
    )
    # Tool name prefixes to skip (e.g., "mcp__" for MCP tools, "_internal")
    skip_tool_prefixes: list[str] = Field(default_factory=list)
    # Tool categories to skip (if host provides category info)
    skip_tool_categories: list[str] = Field(default_factory=list)
    # Maximum output characters to store (truncate if exceeded)
    max_output_chars: int = 50000
    # Maximum input characters to store
    max_input_chars: int = 10000
    # Skip tools that failed (success=false)
    ignore_failed_tools: bool = False
    # Minimum output length to consider storing (filter noise)
    min_output_chars: int = 0
    # Regex patterns for content that should be fully redacted (replaced with [REDACTED])
    redaction_patterns: list[str] = Field(
        default_factory=lambda: [
            r"(?i)(api[_-]?key|apikey|secret[_-]?key|password|passwd|token|bearer)\s*[=:]\s*['\"]?[\w\-\.]+['\"]?",
            r"(?i)authorization:\s*bearer\s+[\w\-\.]+",
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI API keys
            r"AIza[a-zA-Z0-9_\-]{35}",  # Google API keys
        ]
    )
    # Tags to add to all ingested tool observations
    default_tags: list[str] = Field(default_factory=lambda: ["tool", "auto-ingested"])


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    vector: VectorConfig = Field(default_factory=VectorConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)


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

def _env_float(name: str) -> Optional[float]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return float(value)
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

    qdrant_url = os.environ.get("AI_MEM_QDRANT_URL")
    qdrant_api_key = os.environ.get("AI_MEM_QDRANT_API_KEY")
    qdrant_collection = os.environ.get("AI_MEM_QDRANT_COLLECTION")
    qdrant_vector_size = _env_int("AI_MEM_QDRANT_VECTOR_SIZE")

    if vector_provider:
        config.vector.provider = vector_provider
    cache_ttl = _env_int("AI_MEM_SEARCH_CACHE_TTL")
    cache_entries = _env_int("AI_MEM_SEARCH_CACHE_ENTRIES")
    fts_weight = _env_float("AI_MEM_SEARCH_FTS_WEIGHT")
    vector_weight = _env_float("AI_MEM_SEARCH_VECTOR_WEIGHT")
    recency_half_life = _env_float("AI_MEM_SEARCH_RECENCY_HALFLIFE_HOURS")
    recency_weight = _env_float("AI_MEM_SEARCH_RECENCY_WEIGHT")

    if cache_ttl is not None:
        config.search.cache_ttl_seconds = cache_ttl
    if cache_entries is not None:
        config.search.cache_max_entries = cache_entries
    if fts_weight is not None:
        config.search.fts_weight = fts_weight
    if vector_weight is not None:
        config.search.vector_weight = vector_weight
    if recency_half_life is not None:
        config.search.recency_half_life_hours = recency_half_life
    if recency_weight is not None:
        config.search.recency_weight = recency_weight
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
    if qdrant_url:
        config.vector.qdrant_url = qdrant_url
    if qdrant_api_key:
        config.vector.qdrant_api_key = qdrant_api_key
    if qdrant_collection:
        config.vector.qdrant_collection = qdrant_collection
    if qdrant_vector_size is not None:
        config.vector.qdrant_vector_size = qdrant_vector_size

    # Ingestion config overrides
    skip_tools = _env_list("AI_MEM_SKIP_TOOL_NAMES")
    skip_prefixes = _env_list("AI_MEM_SKIP_TOOL_PREFIXES")
    skip_categories = _env_list("AI_MEM_SKIP_TOOL_CATEGORIES")
    max_output = _env_int("AI_MEM_MAX_OUTPUT_CHARS")
    max_input = _env_int("AI_MEM_MAX_INPUT_CHARS")
    ignore_failed = _env_bool("AI_MEM_IGNORE_FAILED_TOOLS")
    min_output = _env_int("AI_MEM_MIN_OUTPUT_CHARS")
    default_tags = _env_list("AI_MEM_INGESTION_DEFAULT_TAGS")

    if skip_tools is not None:
        config.ingestion.skip_tool_names = skip_tools
    if skip_prefixes is not None:
        config.ingestion.skip_tool_prefixes = skip_prefixes
    if skip_categories is not None:
        config.ingestion.skip_tool_categories = skip_categories
    if max_output is not None:
        config.ingestion.max_output_chars = max_output
    if max_input is not None:
        config.ingestion.max_input_chars = max_input
    if ignore_failed is not None:
        config.ingestion.ignore_failed_tools = ignore_failed
    if min_output is not None:
        config.ingestion.min_output_chars = min_output
    if default_tags is not None:
        config.ingestion.default_tags = default_tags

    return config


# =============================================================================
# Configuration Validation
# =============================================================================

# LLM providers that require API keys
_LLM_API_KEY_REQUIRED = {"openai", "anthropic", "gemini", "azure-openai", "cohere"}

# Embedding providers that require API keys
_EMBEDDING_API_KEY_REQUIRED = {"openai", "cohere", "azure-openai", "voyageai"}

# Vector providers that require external connections
_VECTOR_EXTERNAL_PROVIDERS = {"pgvector", "qdrant"}


def validate_config(config: AppConfig, strict: bool = False) -> Tuple[bool, List[str]]:
    """Validate configuration settings.

    Args:
        config: The configuration to validate
        strict: If True, treat warnings as errors

    Returns:
        Tuple of (is_valid, list of warning/error messages)

    Raises:
        MissingConfigError: If a required configuration value is missing
        InvalidConfigError: If a configuration value is invalid
    """
    warnings: List[str] = []
    errors: List[str] = []

    # Validate LLM configuration
    _validate_llm_config(config.llm, errors, warnings)

    # Validate embedding configuration
    _validate_embedding_config(config.embeddings, errors, warnings)

    # Validate search configuration
    _validate_search_config(config.search, errors, warnings)

    # Validate vector store configuration
    _validate_vector_config(config.vector, errors, warnings)

    # Validate storage configuration
    _validate_storage_config(config.storage, errors, warnings)

    # Validate ingestion configuration
    _validate_ingestion_config(config.ingestion, errors, warnings)

    # Validate context configuration
    _validate_context_config(config.context, errors, warnings)

    # If strict mode, treat warnings as errors
    if strict:
        errors.extend(warnings)
        warnings = []

    if errors:
        raise ConfigurationError(
            f"Configuration validation failed with {len(errors)} error(s): {'; '.join(errors)}",
            {"errors": errors, "warnings": warnings}
        )

    return len(errors) == 0, warnings


def _validate_llm_config(llm: LLMConfig, errors: List[str], warnings: List[str]) -> None:
    """Validate LLM configuration."""
    provider = llm.provider.lower()

    # Check API key requirements
    if provider in _LLM_API_KEY_REQUIRED:
        if not llm.api_key:
            # Check environment variable fallbacks
            env_key = os.environ.get(f"{provider.upper().replace('-', '_')}_API_KEY")
            if not env_key:
                errors.append(f"LLM provider '{provider}' requires an API key")

    # Azure-specific validation
    if provider == "azure-openai":
        if not llm.base_url:
            errors.append("Azure OpenAI requires 'base_url' (azure_endpoint)")
        if not llm.api_version:
            warnings.append("Azure OpenAI api_version not set, using default")

    # Validate timeout
    if llm.timeout_s <= 0:
        errors.append(f"LLM timeout must be positive, got {llm.timeout_s}")
    elif llm.timeout_s > 300:
        warnings.append(f"LLM timeout is very high ({llm.timeout_s}s), may cause long hangs")


def _validate_embedding_config(embeddings: EmbeddingConfig, errors: List[str], warnings: List[str]) -> None:
    """Validate embedding configuration."""
    provider = embeddings.provider.lower()

    # Check API key requirements
    if provider in _EMBEDDING_API_KEY_REQUIRED:
        if not embeddings.api_key:
            env_key = os.environ.get(f"{provider.upper().replace('-', '_')}_API_KEY")
            if not env_key:
                errors.append(f"Embedding provider '{provider}' requires an API key")


def _validate_search_config(search: SearchConfig, errors: List[str], warnings: List[str]) -> None:
    """Validate search configuration."""
    # Validate chunk settings
    if search.chunk_size <= 0:
        errors.append(f"chunk_size must be positive, got {search.chunk_size}")
    if search.chunk_overlap < 0:
        errors.append(f"chunk_overlap must be non-negative, got {search.chunk_overlap}")
    if search.chunk_overlap >= search.chunk_size:
        errors.append(f"chunk_overlap ({search.chunk_overlap}) must be less than chunk_size ({search.chunk_size})")

    # Validate top_k settings
    if search.vector_top_k <= 0:
        errors.append(f"vector_top_k must be positive, got {search.vector_top_k}")
    if search.fts_top_k <= 0:
        errors.append(f"fts_top_k must be positive, got {search.fts_top_k}")

    # Validate cache settings
    if search.cache_ttl_seconds < 0:
        errors.append(f"cache_ttl_seconds must be non-negative, got {search.cache_ttl_seconds}")
    if search.cache_max_entries < 0:
        errors.append(f"cache_max_entries must be non-negative, got {search.cache_max_entries}")

    # Validate weights (should be non-negative and reasonably sum)
    if search.fts_weight < 0:
        errors.append(f"fts_weight must be non-negative, got {search.fts_weight}")
    if search.vector_weight < 0:
        errors.append(f"vector_weight must be non-negative, got {search.vector_weight}")
    if search.recency_weight < 0:
        errors.append(f"recency_weight must be non-negative, got {search.recency_weight}")

    total_weight = search.fts_weight + search.vector_weight + search.recency_weight
    if total_weight == 0:
        errors.append("At least one search weight (fts, vector, recency) must be positive")
    elif total_weight > 10:
        warnings.append(f"Total search weights ({total_weight}) are unusually high")

    # Validate recency half-life
    if search.recency_half_life_hours <= 0:
        errors.append(f"recency_half_life_hours must be positive, got {search.recency_half_life_hours}")


def _validate_vector_config(vector: VectorConfig, errors: List[str], warnings: List[str]) -> None:
    """Validate vector store configuration."""
    provider = vector.provider.lower()
    valid_providers = {"chroma", "pgvector", "qdrant"}

    if provider not in valid_providers:
        warnings.append(f"Unknown vector provider '{provider}', valid options: {valid_providers}")

    # PGVector-specific validation
    if provider == "pgvector":
        if not vector.pgvector_dsn:
            env_dsn = os.environ.get("AI_MEM_PGVECTOR_DSN")
            if not env_dsn:
                errors.append("pgvector provider requires pgvector_dsn configuration")

        if vector.pgvector_dimension <= 0:
            errors.append(f"pgvector_dimension must be positive, got {vector.pgvector_dimension}")

        valid_index_types = {"ivfflat", "hnsw"}
        if vector.pgvector_index_type.lower() not in valid_index_types:
            warnings.append(f"Unknown pgvector index type '{vector.pgvector_index_type}', valid: {valid_index_types}")

        if vector.pgvector_lists <= 0:
            errors.append(f"pgvector_lists must be positive, got {vector.pgvector_lists}")

    # Qdrant-specific validation
    if provider == "qdrant":
        if not vector.qdrant_url:
            env_url = os.environ.get("AI_MEM_QDRANT_URL")
            if not env_url:
                errors.append("qdrant provider requires qdrant_url configuration")

        if vector.qdrant_vector_size <= 0:
            errors.append(f"qdrant_vector_size must be positive, got {vector.qdrant_vector_size}")


def _validate_storage_config(storage: StorageConfig, errors: List[str], warnings: List[str]) -> None:
    """Validate storage configuration."""
    # Check data_dir is a valid path format
    try:
        data_path = Path(storage.data_dir).expanduser()
        # Check if parent directory exists or can be created
        if data_path.exists() and not data_path.is_dir():
            errors.append(f"data_dir '{storage.data_dir}' exists but is not a directory")
    except Exception as e:
        errors.append(f"Invalid data_dir path '{storage.data_dir}': {e}")


def _validate_ingestion_config(ingestion: IngestionConfig, errors: List[str], warnings: List[str]) -> None:
    """Validate ingestion configuration."""
    # Validate character limits
    if ingestion.max_output_chars <= 0:
        errors.append(f"max_output_chars must be positive, got {ingestion.max_output_chars}")
    if ingestion.max_input_chars <= 0:
        errors.append(f"max_input_chars must be positive, got {ingestion.max_input_chars}")
    if ingestion.min_output_chars < 0:
        errors.append(f"min_output_chars must be non-negative, got {ingestion.min_output_chars}")

    # Validate redaction patterns are valid regex
    for i, pattern in enumerate(ingestion.redaction_patterns):
        try:
            re.compile(pattern)
        except re.error as e:
            errors.append(f"Invalid regex in redaction_patterns[{i}]: {e}")


def _validate_context_config(context: ContextConfig, errors: List[str], warnings: List[str]) -> None:
    """Validate context configuration."""
    if context.total_observation_count <= 0:
        errors.append(f"total_observation_count must be positive, got {context.total_observation_count}")
    if context.full_observation_count < 0:
        errors.append(f"full_observation_count must be non-negative, got {context.full_observation_count}")
    if context.full_observation_count > context.total_observation_count:
        errors.append(
            f"full_observation_count ({context.full_observation_count}) "
            f"cannot exceed total_observation_count ({context.total_observation_count})"
        )

    valid_fields = {"content", "structured", "summary"}
    if context.full_observation_field not in valid_fields:
        warnings.append(
            f"Unknown full_observation_field '{context.full_observation_field}', "
            f"valid options: {valid_fields}"
        )


def load_and_validate_config(strict: bool = False) -> Tuple[AppConfig, List[str]]:
    """Load and validate configuration.

    Args:
        strict: If True, treat warnings as errors

    Returns:
        Tuple of (validated config, list of warnings)

    Raises:
        ConfigurationError: If validation fails
    """
    config = load_config()
    is_valid, warnings = validate_config(config, strict=strict)
    return config, warnings

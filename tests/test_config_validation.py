"""Tests for configuration validation."""

import pytest

from ai_mem.config import (
    AppConfig,
    LLMConfig,
    EmbeddingConfig,
    SearchConfig,
    VectorConfig,
    StorageConfig,
    IngestionConfig,
    ContextConfig,
    validate_config,
    load_and_validate_config,
)
from ai_mem.exceptions import ConfigurationError


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_valid_default_config(self):
        """Test that default configuration is valid."""
        config = AppConfig()
        is_valid, warnings = validate_config(config)
        assert is_valid

    def test_invalid_chunk_size(self):
        """Test validation fails for invalid chunk_size."""
        config = AppConfig(
            search=SearchConfig(chunk_size=0)
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "chunk_size must be positive" in str(exc_info.value)

    def test_invalid_chunk_overlap(self):
        """Test validation fails for negative chunk_overlap."""
        config = AppConfig(
            search=SearchConfig(chunk_overlap=-1)
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "chunk_overlap must be non-negative" in str(exc_info.value)

    def test_chunk_overlap_exceeds_size(self):
        """Test validation fails when chunk_overlap >= chunk_size."""
        config = AppConfig(
            search=SearchConfig(chunk_size=100, chunk_overlap=100)
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "chunk_overlap" in str(exc_info.value) and "less than chunk_size" in str(exc_info.value)

    def test_invalid_cache_settings(self):
        """Test validation fails for negative cache settings."""
        config = AppConfig(
            search=SearchConfig(cache_ttl_seconds=-1)
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "cache_ttl_seconds must be non-negative" in str(exc_info.value)

    def test_invalid_search_weights(self):
        """Test validation fails for negative weights."""
        config = AppConfig(
            search=SearchConfig(fts_weight=-0.1)
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "fts_weight must be non-negative" in str(exc_info.value)

    def test_zero_total_weights(self):
        """Test validation fails when all weights are zero."""
        config = AppConfig(
            search=SearchConfig(fts_weight=0, vector_weight=0, recency_weight=0)
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "At least one search weight" in str(exc_info.value)

    def test_invalid_recency_half_life(self):
        """Test validation fails for non-positive recency half-life."""
        config = AppConfig(
            search=SearchConfig(recency_half_life_hours=0)
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "recency_half_life_hours must be positive" in str(exc_info.value)


class TestLLMConfigValidation:
    """Tests for LLM configuration validation."""

    def test_openai_requires_api_key(self):
        """Test OpenAI provider requires API key."""
        config = AppConfig(
            llm=LLMConfig(provider="openai", api_key=None)
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "requires an API key" in str(exc_info.value)

    def test_anthropic_requires_api_key(self):
        """Test Anthropic provider requires API key."""
        config = AppConfig(
            llm=LLMConfig(provider="anthropic", api_key=None)
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "requires an API key" in str(exc_info.value)

    def test_gemini_requires_api_key(self):
        """Test Gemini provider requires API key."""
        config = AppConfig(
            llm=LLMConfig(provider="gemini", api_key=None)
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "requires an API key" in str(exc_info.value)

    def test_none_provider_no_api_key_needed(self):
        """Test 'none' provider doesn't require API key."""
        config = AppConfig(
            llm=LLMConfig(provider="none", api_key=None)
        )
        is_valid, warnings = validate_config(config)
        assert is_valid

    def test_invalid_timeout(self):
        """Test validation fails for non-positive timeout."""
        config = AppConfig(
            llm=LLMConfig(timeout_s=0)
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "timeout must be positive" in str(exc_info.value)

    def test_high_timeout_warning(self):
        """Test warning for very high timeout."""
        config = AppConfig(
            llm=LLMConfig(timeout_s=500)
        )
        is_valid, warnings = validate_config(config)
        assert is_valid
        assert any("timeout is very high" in w for w in warnings)

    def test_azure_requires_base_url(self):
        """Test Azure OpenAI requires base_url."""
        config = AppConfig(
            llm=LLMConfig(provider="azure-openai", api_key="test-key", base_url=None)
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "Azure OpenAI requires 'base_url'" in str(exc_info.value)


class TestVectorConfigValidation:
    """Tests for vector store configuration validation."""

    def test_pgvector_requires_dsn(self):
        """Test pgvector provider requires DSN."""
        config = AppConfig(
            vector=VectorConfig(provider="pgvector", pgvector_dsn=None)
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "pgvector_dsn configuration" in str(exc_info.value)

    def test_pgvector_invalid_dimension(self):
        """Test pgvector validates dimension."""
        config = AppConfig(
            vector=VectorConfig(
                provider="pgvector",
                pgvector_dsn="postgresql://test:test@localhost/test",
                pgvector_dimension=0
            )
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "pgvector_dimension must be positive" in str(exc_info.value)

    def test_qdrant_requires_url(self):
        """Test qdrant provider requires URL."""
        config = AppConfig(
            vector=VectorConfig(provider="qdrant", qdrant_url=None)
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "qdrant_url configuration" in str(exc_info.value)

    def test_unknown_provider_warning(self):
        """Test warning for unknown vector provider."""
        config = AppConfig(
            vector=VectorConfig(provider="unknown_provider")
        )
        is_valid, warnings = validate_config(config)
        assert is_valid
        assert any("Unknown vector provider" in w for w in warnings)

    def test_chroma_valid_without_extra_config(self):
        """Test chroma is valid with default settings."""
        config = AppConfig(
            vector=VectorConfig(provider="chroma")
        )
        is_valid, warnings = validate_config(config)
        assert is_valid


class TestIngestionConfigValidation:
    """Tests for ingestion configuration validation."""

    def test_invalid_max_output_chars(self):
        """Test validation fails for non-positive max_output_chars."""
        config = AppConfig(
            ingestion=IngestionConfig(max_output_chars=0)
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "max_output_chars must be positive" in str(exc_info.value)

    def test_invalid_max_input_chars(self):
        """Test validation fails for non-positive max_input_chars."""
        config = AppConfig(
            ingestion=IngestionConfig(max_input_chars=0)
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "max_input_chars must be positive" in str(exc_info.value)

    def test_invalid_min_output_chars(self):
        """Test validation fails for negative min_output_chars."""
        config = AppConfig(
            ingestion=IngestionConfig(min_output_chars=-1)
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "min_output_chars must be non-negative" in str(exc_info.value)

    def test_invalid_regex_pattern(self):
        """Test validation fails for invalid regex in redaction_patterns."""
        config = AppConfig(
            ingestion=IngestionConfig(redaction_patterns=["[invalid regex("])
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "Invalid regex in redaction_patterns" in str(exc_info.value)


class TestContextConfigValidation:
    """Tests for context configuration validation."""

    def test_invalid_total_observation_count(self):
        """Test validation fails for non-positive total_observation_count."""
        config = AppConfig(
            context=ContextConfig(total_observation_count=0)
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "total_observation_count must be positive" in str(exc_info.value)

    def test_invalid_full_observation_count(self):
        """Test validation fails for negative full_observation_count."""
        config = AppConfig(
            context=ContextConfig(full_observation_count=-1)
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "full_observation_count must be non-negative" in str(exc_info.value)

    def test_full_exceeds_total(self):
        """Test validation fails when full_observation_count > total_observation_count."""
        config = AppConfig(
            context=ContextConfig(total_observation_count=5, full_observation_count=10)
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "cannot exceed total_observation_count" in str(exc_info.value)

    def test_unknown_full_observation_field_warning(self):
        """Test warning for unknown full_observation_field."""
        config = AppConfig(
            context=ContextConfig(full_observation_field="unknown_field")
        )
        is_valid, warnings = validate_config(config)
        assert is_valid
        assert any("Unknown full_observation_field" in w for w in warnings)


class TestStrictMode:
    """Tests for strict validation mode."""

    def test_strict_mode_treats_warnings_as_errors(self):
        """Test that strict mode treats warnings as errors."""
        config = AppConfig(
            llm=LLMConfig(timeout_s=500)  # This generates a warning
        )
        # Non-strict should pass
        is_valid, warnings = validate_config(config, strict=False)
        assert is_valid
        assert len(warnings) > 0

        # Strict should fail
        with pytest.raises(ConfigurationError):
            validate_config(config, strict=True)


class TestLoadAndValidateConfig:
    """Tests for load_and_validate_config function."""

    def test_load_and_validate_returns_tuple(self):
        """Test that load_and_validate_config returns config and warnings."""
        config, warnings = load_and_validate_config(strict=False)
        assert isinstance(config, AppConfig)
        assert isinstance(warnings, list)

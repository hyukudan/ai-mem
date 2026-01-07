"""Tests for Additional Features (Query Expansion, Token Budget, Inline Tags, Context Preview).

Tests for the additional features implemented:
- Query Expansion (synonyms, case variations)
- Token Budget Warnings
- Inline Context Tags (<mem> tag parsing and expansion)
- Context Preview Mode
"""

import pytest
import json
from dataclasses import asdict


# =============================================================================
# Query Expansion Tests
# =============================================================================


class TestQueryExpansion:
    """Tests for query expansion functionality."""

    def test_expand_term_basic(self):
        """Basic term expansion returns original term."""
        from ai_mem.intent import expand_term

        result = expand_term("hello")
        assert "hello" in result

    def test_expand_term_with_synonym(self):
        """Term with known synonyms expands correctly."""
        from ai_mem.intent import expand_term

        result = expand_term("auth")
        assert "auth" in result
        assert "authentication" in result or "login" in result

    def test_expand_term_case_variation_snake(self):
        """Snake_case term generates camelCase variant."""
        from ai_mem.intent import expand_term

        result = expand_term("get_user")
        # Should include original and camelCase
        assert "get_user" in result
        assert "getUser" in result

    def test_expand_term_case_variation_camel(self):
        """CamelCase term generates snake_case variant."""
        from ai_mem.intent import expand_term

        result = expand_term("getUserData")
        assert "getUserData" in result
        # Should generate snake variant
        assert any("_" in term for term in result)

    def test_expand_query_basic(self):
        """Basic query expansion works."""
        from ai_mem.intent import expand_query

        result = expand_query("auth login")
        assert result.original == "auth login"
        assert len(result.all_queries) >= 1
        assert result.all_queries[0] == "auth login"

    def test_expand_query_adds_synonyms(self):
        """Query expansion adds synonyms."""
        from ai_mem.intent import expand_query

        result = expand_query("database query")
        assert len(result.expanded_terms) > 2  # More than original terms

    def test_expand_query_empty(self):
        """Empty query returns empty result."""
        from ai_mem.intent import expand_query

        result = expand_query("")
        assert result.original == ""
        assert result.all_queries == [""]

    def test_expand_query_max_terms(self):
        """Query expansion respects max terms limit."""
        from ai_mem.intent import expand_query

        result = expand_query("auth database api", max_total_terms=5)
        assert len(result.expanded_terms) <= 5

    def test_generate_expanded_queries(self):
        """Generate expanded queries from prompt."""
        from ai_mem.intent import generate_expanded_queries

        result = generate_expanded_queries("Fix the authentication bug")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_expand_query_deduplication(self):
        """Expanded terms are deduplicated."""
        from ai_mem.intent import expand_query

        result = expand_query("auth authentication")  # Overlapping terms
        # Should not have duplicates
        assert len(result.expanded_terms) == len(set(result.expanded_terms))


class TestTechnicalSynonyms:
    """Tests for technical synonym coverage."""

    def test_auth_synonyms(self):
        """Auth has expected synonyms."""
        from ai_mem.intent import TECHNICAL_SYNONYMS

        assert "auth" in TECHNICAL_SYNONYMS
        synonyms = TECHNICAL_SYNONYMS["auth"]
        assert "authentication" in synonyms
        assert "login" in synonyms

    def test_database_synonyms(self):
        """Database has expected synonyms."""
        from ai_mem.intent import TECHNICAL_SYNONYMS

        assert "database" in TECHNICAL_SYNONYMS
        assert "db" in TECHNICAL_SYNONYMS["database"]

    def test_error_synonyms(self):
        """Error has expected synonyms."""
        from ai_mem.intent import TECHNICAL_SYNONYMS

        assert "error" in TECHNICAL_SYNONYMS
        assert "exception" in TECHNICAL_SYNONYMS["error"]
        assert "bug" in TECHNICAL_SYNONYMS["error"]


# =============================================================================
# Token Budget Warning Tests
# =============================================================================


class TestTokenBudgetWarnings:
    """Tests for token budget warning functionality."""

    def test_get_model_context_window_known(self):
        """Known model returns correct context window."""
        from ai_mem.context import get_model_context_window

        assert get_model_context_window("claude-3-opus") == 200000
        assert get_model_context_window("gpt-4") == 8192
        assert get_model_context_window("gemini-pro") == 32000

    def test_get_model_context_window_unknown(self):
        """Unknown model returns default context window."""
        from ai_mem.context import get_model_context_window

        result = get_model_context_window("unknown-model")
        assert result == 16000  # Default

    def test_get_model_context_window_none(self):
        """None model returns default context window."""
        from ai_mem.context import get_model_context_window

        result = get_model_context_window(None)
        assert result == 16000

    def test_get_model_context_window_partial_match(self):
        """Partial model name matches correctly."""
        from ai_mem.context import get_model_context_window

        # Should match via partial matching
        result = get_model_context_window("claude-3-opus-20240229")
        assert result == 200000

    def test_check_token_budget_no_warning(self):
        """Low token usage returns no warning."""
        from ai_mem.context import check_token_budget

        # 100 tokens with 4000 budget (2.5%) - no warning
        result = check_token_budget(100, max_budget=4000)
        assert result is None

    def test_check_token_budget_info(self):
        """Info level warning at 15%+ usage."""
        from ai_mem.context import check_token_budget

        # 700 tokens with 4000 budget (17.5%) - info
        result = check_token_budget(700, max_budget=4000)
        assert result is not None
        assert result.level == "info"

    def test_check_token_budget_warning(self):
        """Warning level at 30%+ usage."""
        from ai_mem.context import check_token_budget

        # 1500 tokens with 4000 budget (37.5%) - warning
        result = check_token_budget(1500, max_budget=4000)
        assert result is not None
        assert result.level == "warning"

    def test_check_token_budget_critical(self):
        """Critical level at 50%+ usage."""
        from ai_mem.context import check_token_budget

        # 2500 tokens with 4000 budget (62.5%) - critical
        result = check_token_budget(2500, max_budget=4000)
        assert result is not None
        assert result.level == "critical"

    def test_check_token_budget_with_model(self):
        """Check budget using model context window."""
        from ai_mem.context import check_token_budget

        # Using model's context window (200000 * 0.25 = 50000 budget)
        # 30000 tokens = 60% of budget = critical
        result = check_token_budget(30000, model="claude-3-opus")
        assert result is not None
        assert result.level == "critical"

    def test_token_budget_warning_has_recommendations(self):
        """Warning includes recommendations."""
        from ai_mem.context import check_token_budget

        result = check_token_budget(2500, max_budget=4000)
        assert result.recommendations
        assert len(result.recommendations) > 0

    def test_format_token_warning(self):
        """Token warning formats correctly."""
        from ai_mem.context import check_token_budget, format_token_warning

        warning = check_token_budget(2500, max_budget=4000)
        formatted = format_token_warning(warning)

        assert "CRITICAL" in formatted
        assert "Recommendations" in formatted


class TestTokenBudgetWarningDataclass:
    """Tests for TokenBudgetWarning dataclass."""

    def test_warning_dataclass_fields(self):
        """Warning dataclass has all expected fields."""
        from ai_mem.context import TokenBudgetWarning

        warning = TokenBudgetWarning(
            level="warning",
            message="Test message",
            tokens_used=1000,
            tokens_budget=4000,
            percentage=25.0,
            recommendations=["Rec 1", "Rec 2"],
        )

        assert warning.level == "warning"
        assert warning.tokens_used == 1000
        assert warning.tokens_budget == 4000
        assert warning.percentage == 25.0
        assert len(warning.recommendations) == 2


# =============================================================================
# Inline Context Tags Tests
# =============================================================================


class TestInlineTagsParsing:
    """Tests for inline <mem> tag parsing."""

    def test_has_mem_tags_true(self):
        """Detects presence of mem tags."""
        from ai_mem.inline_tags import has_mem_tags

        assert has_mem_tags('Query <mem query="auth"/> for context')
        assert has_mem_tags('<mem query="test"/>')

    def test_has_mem_tags_false(self):
        """Returns false when no mem tags."""
        from ai_mem.inline_tags import has_mem_tags

        assert not has_mem_tags("No tags here")
        assert not has_mem_tags("<memory>not a mem tag</memory>")

    def test_parse_prompt_no_tags(self):
        """Parse prompt with no tags."""
        from ai_mem.inline_tags import parse_prompt

        result = parse_prompt("Just a regular prompt")
        assert not result.has_tags
        assert len(result.tags) == 0
        assert result.cleaned == "Just a regular prompt"

    def test_parse_prompt_single_tag(self):
        """Parse prompt with single self-closing tag."""
        from ai_mem.inline_tags import parse_prompt

        result = parse_prompt('Help with <mem query="auth"/> please')
        assert result.has_tags
        assert len(result.tags) == 1
        assert result.tags[0].query == "auth"

    def test_parse_prompt_multiple_tags(self):
        """Parse prompt with multiple tags."""
        from ai_mem.inline_tags import parse_prompt

        result = parse_prompt('<mem query="auth"/> and <mem query="db"/>')
        assert result.has_tags
        assert len(result.tags) == 2
        assert result.tags[0].query == "auth"
        assert result.tags[1].query == "db"

    def test_parse_tag_with_limit(self):
        """Parse tag with limit attribute."""
        from ai_mem.inline_tags import parse_prompt

        result = parse_prompt('<mem query="test" limit="10"/>')
        assert result.tags[0].limit == 10

    def test_parse_tag_with_mode(self):
        """Parse tag with mode attribute."""
        from ai_mem.inline_tags import parse_prompt

        result = parse_prompt('<mem query="test" mode="compact"/>')
        assert result.tags[0].mode == "compact"

    def test_parse_tag_with_type(self):
        """Parse tag with type attribute."""
        from ai_mem.inline_tags import parse_prompt

        result = parse_prompt('<mem query="test" type="decision"/>')
        assert result.tags[0].obs_type == "decision"

    def test_parse_tag_with_tags(self):
        """Parse tag with tags attribute (comma-separated)."""
        from ai_mem.inline_tags import parse_prompt

        result = parse_prompt('<mem query="test" tags="api,backend,auth"/>')
        assert result.tags[0].tags == ["api", "backend", "auth"]

    def test_parse_tag_with_project(self):
        """Parse tag with project attribute."""
        from ai_mem.inline_tags import parse_prompt

        result = parse_prompt('<mem query="test" project="/my/project"/>')
        assert result.tags[0].project == "/my/project"

    def test_parse_cleaned_prompt(self):
        """Cleaned prompt has tags removed."""
        from ai_mem.inline_tags import parse_prompt

        result = parse_prompt('Before <mem query="test"/> after')
        assert "Before" in result.cleaned
        assert "after" in result.cleaned
        assert "<mem" not in result.cleaned

    def test_parse_tag_default_limit(self):
        """Tag without limit uses default."""
        from ai_mem.inline_tags import parse_prompt

        result = parse_prompt('<mem query="test"/>')
        assert result.tags[0].limit == 5  # Default

    def test_parse_tag_default_mode(self):
        """Tag without mode uses standard."""
        from ai_mem.inline_tags import parse_prompt

        result = parse_prompt('<mem query="test"/>')
        assert result.tags[0].mode == "standard"


class TestInlineTagsAttributes:
    """Tests for tag attribute parsing."""

    def test_parse_mem_tag_attrs_basic(self):
        """Parse basic attributes."""
        from ai_mem.inline_tags import parse_mem_tag_attrs

        result = parse_mem_tag_attrs('query="auth" limit="5"')
        assert result["query"] == "auth"
        assert result["limit"] == "5"

    def test_parse_mem_tag_attrs_single_quotes(self):
        """Parse single-quoted attributes."""
        from ai_mem.inline_tags import parse_mem_tag_attrs

        result = parse_mem_tag_attrs("query='auth' limit='5'")
        assert result["query"] == "auth"
        assert result["limit"] == "5"

    def test_parse_mem_tag_attrs_mixed_case(self):
        """Attribute names are case-insensitive."""
        from ai_mem.inline_tags import parse_mem_tag_attrs

        result = parse_mem_tag_attrs('Query="auth" LIMIT="5"')
        assert result["query"] == "auth"
        assert result["limit"] == "5"


class TestMemTagDataclass:
    """Tests for MemTag dataclass."""

    def test_mem_tag_defaults(self):
        """MemTag has correct defaults."""
        from ai_mem.inline_tags import MemTag

        tag = MemTag(raw="<mem/>")
        assert tag.query is None
        assert tag.limit == 5
        assert tag.mode == "standard"
        assert tag.tags == []
        assert tag.project is None


class TestParsedPromptDataclass:
    """Tests for ParsedPrompt dataclass."""

    def test_parsed_prompt_fields(self):
        """ParsedPrompt has all expected fields."""
        from ai_mem.inline_tags import ParsedPrompt

        result = ParsedPrompt(
            original="test",
            tags=[],
            has_tags=False,
            cleaned="test",
        )
        assert result.original == "test"
        assert result.cleaned == "test"


class TestExpansionSummary:
    """Tests for expansion summary formatting."""

    def test_format_expansion_summary_empty(self):
        """Empty expansions returns appropriate message."""
        from ai_mem.inline_tags import format_expansion_summary

        result = format_expansion_summary([])
        assert "No mem tags" in result

    def test_format_expansion_summary_with_expansions(self):
        """Formats expansion summary correctly."""
        from ai_mem.inline_tags import format_expansion_summary

        expansions = [
            {"query": "auth", "results": 5, "tokens": 100, "mode": "standard"},
            {"query": "db", "results": 3, "tokens": 75, "mode": "compact"},
        ]
        result = format_expansion_summary(expansions)

        assert "2 mem tag(s)" in result
        assert "auth" in result
        assert "db" in result


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntentWithExpansion:
    """Integration tests for intent detection with query expansion."""

    def test_intent_generates_expandable_query(self):
        """Intent detection generates query that can be expanded."""
        from ai_mem.intent import detect_intent, expand_query

        intent = detect_intent("Fix the authentication bug")
        expanded = expand_query(intent.query)

        assert expanded.original == intent.query
        assert len(expanded.all_queries) >= 1


class TestContextWithBudgetCheck:
    """Integration tests for context building with budget checking."""

    def test_context_metadata_has_tokens(self):
        """Context metadata includes token info for budget checking."""
        from ai_mem.context import estimate_tokens

        # Basic token estimation
        text = "This is a test string with some words"
        tokens = estimate_tokens(text)
        assert tokens > 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_expand_query_special_characters(self):
        """Query with special characters handles correctly."""
        from ai_mem.intent import expand_query

        result = expand_query("user_id=123")
        assert result.original == "user_id=123"

    def test_parse_tag_empty_query(self):
        """Tag with empty query attribute."""
        from ai_mem.inline_tags import parse_prompt

        result = parse_prompt('<mem query=""/>')
        assert result.tags[0].query == ""

    def test_parse_tag_invalid_limit(self):
        """Tag with invalid limit falls back to default."""
        from ai_mem.inline_tags import parse_prompt

        result = parse_prompt('<mem query="test" limit="invalid"/>')
        assert result.tags[0].limit == 5  # Default

    def test_token_budget_zero_budget(self):
        """Zero budget falls back to model-based calculation."""
        from ai_mem.context import check_token_budget

        # When max_budget=0, it falls back to model-based calculation
        # Using a small token count that won't trigger warning
        result = check_token_budget(10, max_budget=0)
        # With 0 max_budget, falls back to model default, so 10 tokens is tiny
        assert result is None  # 10 tokens is too small to trigger any warning

    def test_expand_query_very_long(self):
        """Very long query handles correctly."""
        from ai_mem.intent import expand_query

        long_query = " ".join(["auth"] * 100)
        result = expand_query(long_query, max_total_terms=10)
        assert len(result.expanded_terms) <= 10

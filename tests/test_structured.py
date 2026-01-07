"""Tests for structured observation extraction."""

import pytest

from ai_mem.structured import (
    StructuredData,
    extract_by_patterns,
    extract_from_tool_output,
    combine_extraction,
)


class TestStructuredData:
    def test_empty(self):
        data = StructuredData()
        assert data.is_empty()

    def test_not_empty_with_facts(self):
        data = StructuredData(facts=["Python 3.11"])
        assert not data.is_empty()

    def test_to_dict_empty(self):
        data = StructuredData()
        assert data.to_dict() == {}

    def test_to_dict_with_data(self):
        data = StructuredData(
            facts=["uses Python 3.11"],
            decisions=["use FastAPI"],
        )
        result = data.to_dict()
        assert result["facts"] == ["uses Python 3.11"]
        assert result["decisions"] == ["use FastAPI"]
        assert "todos" not in result  # Empty lists not included

    def test_from_dict(self):
        data = StructuredData.from_dict({
            "facts": ["fact1"],
            "todos": ["todo1"],
        })
        assert data.facts == ["fact1"]
        assert data.todos == ["todo1"]
        assert data.decisions == []

    def test_merge(self):
        data1 = StructuredData(facts=["fact1"], todos=["todo1"])
        data2 = StructuredData(facts=["fact2", "fact1"], decisions=["decision1"])
        merged = data1.merge(data2)

        assert "fact1" in merged.facts
        assert "fact2" in merged.facts
        assert "todo1" in merged.todos
        assert "decision1" in merged.decisions
        # No duplicates
        assert merged.facts.count("fact1") == 1


class TestExtractByPatterns:
    def test_extract_version(self):
        content = "This project uses Python 3.11 and Node 18.0"
        result = extract_by_patterns(content)
        # Should extract version info
        assert any("3.11" in f or "18.0" in f for f in result.facts)

    def test_extract_todo(self):
        content = "TODO: Fix the authentication bug"
        result = extract_by_patterns(content)
        assert any("authentication" in t.lower() for t in result.todos)

    def test_extract_fixme(self):
        content = "FIXME: Handle edge case"
        result = extract_by_patterns(content)
        assert any("edge case" in t.lower() for t in result.todos)

    def test_extract_question(self):
        content = "How should we handle this error case?"
        result = extract_by_patterns(content)
        assert len(result.questions) > 0
        assert any("error" in q.lower() for q in result.questions)

    def test_extract_error(self):
        content = "Error: Connection refused to database"
        result = extract_by_patterns(content)
        assert any("connection" in e.lower() or "database" in e.lower() for e in result.errors)

    def test_extract_code_ref(self):
        content = "See the fix in src/main.py:42"
        result = extract_by_patterns(content)
        assert len(result.code_refs) > 0
        assert result.code_refs[0]["file"] == "src/main.py"
        assert result.code_refs[0]["line"] == "42"

    def test_extract_decision(self):
        content = "We decided to use PostgreSQL for the database"
        result = extract_by_patterns(content)
        assert any("postgresql" in d.lower() for d in result.decisions)

    def test_extract_port(self):
        content = "PORT=8080 configured for the server"
        result = extract_by_patterns(content)
        assert any("8080" in f for f in result.facts)


class TestExtractFromToolOutput:
    def test_read_tool(self):
        result = extract_from_tool_output(
            tool_name="Read",
            tool_input={"file_path": "/src/main.py", "offset": 10},
            tool_output="file contents here"
        )
        assert len(result.code_refs) > 0
        assert result.code_refs[0]["file"] == "/src/main.py"

    def test_grep_tool(self):
        result = extract_from_tool_output(
            tool_name="Grep",
            tool_input={"pattern": "error"},
            tool_output="src/utils.py:25: handle error\nlib/core.py:100: error case"
        )
        assert len(result.code_refs) >= 2
        files = [r["file"] for r in result.code_refs]
        assert "src/utils.py" in files
        assert "lib/core.py" in files

    def test_bash_tool_error(self):
        result = extract_from_tool_output(
            tool_name="Bash",
            tool_input={"command": "npm test"},
            tool_output="Error: Test failed\nAssertionError: expected 1 to equal 2"
        )
        assert len(result.errors) > 0

    def test_edit_tool(self):
        result = extract_from_tool_output(
            tool_name="Edit",
            tool_input={"file_path": "/src/config.py"},
            tool_output="File edited successfully"
        )
        assert len(result.code_refs) > 0
        assert result.code_refs[0]["file"] == "/src/config.py"
        assert any("config.py" in f for f in result.facts)

    def test_write_tool(self):
        result = extract_from_tool_output(
            tool_name="Write",
            tool_input={"file_path": "/new_file.py"},
            tool_output="File created"
        )
        assert len(result.code_refs) > 0
        assert result.code_refs[0]["file"] == "/new_file.py"


class TestCombineExtraction:
    def test_combines_content_and_tool(self):
        result = combine_extraction(
            content="TODO: Refactor this code. Uses Python 3.11",
            tool_name="Read",
            tool_input={"file_path": "/src/app.py"},
            tool_output="class App: pass"
        )
        # Should have data from both content patterns and tool extraction
        assert len(result.todos) > 0 or len(result.facts) > 0
        assert len(result.code_refs) > 0

    def test_content_only(self):
        result = combine_extraction(
            content="Error occurred: Database connection failed",
        )
        assert len(result.errors) > 0

    def test_tool_only(self):
        result = combine_extraction(
            content="",
            tool_name="Grep",
            tool_input={},
            tool_output="main.py:10: match"
        )
        assert len(result.code_refs) > 0


class TestEdgeCases:
    def test_empty_content(self):
        result = extract_by_patterns("")
        assert result.is_empty()

    def test_no_patterns_found(self):
        result = extract_by_patterns("Just a regular sentence with nothing special.")
        # May or may not find anything, but shouldn't crash
        assert isinstance(result, StructuredData)

    def test_very_long_error_truncated(self):
        long_error = "Error: " + "x" * 500
        result = extract_by_patterns(long_error)
        if result.errors:
            assert len(result.errors[0]) <= 200

    def test_special_characters(self):
        content = "TODO: Fix regex for pattern [a-z]+ and \\d{3}"
        result = extract_by_patterns(content)
        # Should handle special chars without crashing
        assert isinstance(result, StructuredData)

    def test_unicode_content(self):
        content = "Error: 文件不存在 (file not found)"
        result = extract_by_patterns(content)
        assert isinstance(result, StructuredData)

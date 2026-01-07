"""Structured observation extraction.

This module provides tools to extract structured information from
raw observation content, such as:
- Facts: Verified information or data points
- Decisions: Choices made during the session
- Patterns: Recurring themes or approaches
- Todos: Action items or tasks to do
- Questions: Open questions or uncertainties
- Insights: Key learnings or discoveries

The extraction can be done via:
1. Pattern matching (fast, rule-based)
2. LLM-based extraction (more accurate, requires API)
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StructuredData:
    """Container for structured information extracted from observations."""

    facts: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    todos: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    code_refs: List[Dict[str, str]] = field(default_factory=list)  # {file, line, context}
    errors: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Check if no structured data was extracted."""
        return not any([
            self.facts, self.decisions, self.patterns,
            self.todos, self.questions, self.insights,
            self.code_refs, self.errors
        ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {}
        if self.facts:
            result["facts"] = self.facts
        if self.decisions:
            result["decisions"] = self.decisions
        if self.patterns:
            result["patterns"] = self.patterns
        if self.todos:
            result["todos"] = self.todos
        if self.questions:
            result["questions"] = self.questions
        if self.insights:
            result["insights"] = self.insights
        if self.code_refs:
            result["code_refs"] = self.code_refs
        if self.errors:
            result["errors"] = self.errors
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredData":
        """Create from dictionary."""
        return cls(
            facts=data.get("facts", []),
            decisions=data.get("decisions", []),
            patterns=data.get("patterns", []),
            todos=data.get("todos", []),
            questions=data.get("questions", []),
            insights=data.get("insights", []),
            code_refs=data.get("code_refs", []),
            errors=data.get("errors", []),
        )

    def merge(self, other: "StructuredData") -> "StructuredData":
        """Merge with another StructuredData, avoiding duplicates."""
        return StructuredData(
            facts=list(dict.fromkeys(self.facts + other.facts)),
            decisions=list(dict.fromkeys(self.decisions + other.decisions)),
            patterns=list(dict.fromkeys(self.patterns + other.patterns)),
            todos=list(dict.fromkeys(self.todos + other.todos)),
            questions=list(dict.fromkeys(self.questions + other.questions)),
            insights=list(dict.fromkeys(self.insights + other.insights)),
            code_refs=self.code_refs + [r for r in other.code_refs if r not in self.code_refs],
            errors=list(dict.fromkeys(self.errors + other.errors)),
        )


# === Pattern-based extraction ===

# Patterns for detecting different types of structured information
FACT_PATTERNS = [
    r"(?:uses?|using|utilizes?)\s+([A-Z][a-zA-Z0-9_]+(?:\s+\d+(?:\.\d+)*)?)",  # "uses Python 3.11"
    r"(?:version|v)\s*[=:]?\s*(\d+(?:\.\d+)+)",  # version numbers
    r"(?:running|installed|requires?)\s+([A-Za-z][a-zA-Z0-9_-]+)",  # dependencies
    r"(?:configured|set|defined)\s+(?:as|to|=)\s+['\"]?([^'\"]+)['\"]?",  # configuration
    r"(?:port|PORT)\s*[=:]\s*(\d+)",  # ports
    r"(?:path|PATH|directory)\s*[=:]\s*([^\s,]+)",  # paths
]

DECISION_PATTERNS = [
    r"(?:decided|chosen?|selected?|going with|opted for)\s+(.+?)(?:\.|$)",
    r"(?:will|should|must)\s+(?:use|implement|add|create)\s+(.+?)(?:\.|$)",
    r"(?:approach|strategy|solution):\s*(.+?)(?:\.|$)",
]

TODO_PATTERNS = [
    r"(?:TODO|FIXME|HACK|XXX)[:\s]+(.+?)(?:\n|$)",
    r"(?:need to|should|must|have to)\s+(.+?)(?:\.|$)",
    r"(?:remaining|left to do|pending):\s*(.+?)(?:\.|$)",
]

QUESTION_PATTERNS = [
    r"\?[^\?]*$",  # Lines ending with ?
    r"(?:wondering|unsure|unclear|question)(?:ing)?\s+(?:about|if|whether)\s+(.+?)(?:\.|$)",
    r"(?:how|what|why|when|where|which|should)\s+.+\?",
]

ERROR_PATTERNS = [
    r"(?:error|exception|failed|failure)[:\s]+(.+?)(?:\n|$)",
    r"(?:traceback|stack trace):.+",
    r"(?:cannot|could not|unable to)\s+(.+?)(?:\.|$)",
]

CODE_REF_PATTERN = r"([a-zA-Z0-9_/.-]+\.[a-zA-Z]+):(\d+)"  # file.py:123


def extract_by_patterns(content: str) -> StructuredData:
    """Extract structured data using regex patterns.

    This is fast but less accurate than LLM-based extraction.
    Good for real-time extraction during observation creation.
    """
    result = StructuredData()
    content_lower = content.lower()

    # Extract facts
    for pattern in FACT_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            fact = match.group(1).strip() if match.lastindex else match.group(0).strip()
            if fact and len(fact) > 2 and fact not in result.facts:
                result.facts.append(fact)

    # Extract decisions
    for pattern in DECISION_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            decision = match.group(1).strip() if match.lastindex else match.group(0).strip()
            if decision and len(decision) > 5 and decision not in result.decisions:
                result.decisions.append(decision)

    # Extract todos
    for pattern in TODO_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            todo = match.group(1).strip() if match.lastindex else match.group(0).strip()
            if todo and len(todo) > 3 and todo not in result.todos:
                result.todos.append(todo)

    # Extract questions (from lines ending with ?)
    for line in content.split("\n"):
        line = line.strip()
        if line.endswith("?") and len(line) > 10:
            if line not in result.questions:
                result.questions.append(line)

    # Extract errors
    for pattern in ERROR_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            error = match.group(1).strip() if match.lastindex else match.group(0).strip()
            if error and len(error) > 5 and error not in result.errors:
                result.errors.append(error[:200])  # Truncate long errors

    # Extract code references (file:line)
    for match in re.finditer(CODE_REF_PATTERN, content):
        ref = {"file": match.group(1), "line": match.group(2)}
        if ref not in result.code_refs:
            result.code_refs.append(ref)

    return result


def extract_from_tool_output(tool_name: str, tool_input: Any, tool_output: Any) -> StructuredData:
    """Extract structured data from tool execution results.

    Different tools provide different types of structured information:
    - Read: file content, code refs
    - Grep/Glob: file matches
    - Bash: command output, errors
    - Edit/Write: file changes
    """
    result = StructuredData()

    tool_lower = tool_name.lower() if tool_name else ""
    output_str = str(tool_output) if tool_output else ""

    # Handle Read tool - extract file reference
    if tool_lower == "read":
        if isinstance(tool_input, dict) and "file_path" in tool_input:
            result.code_refs.append({
                "file": tool_input["file_path"],
                "line": str(tool_input.get("offset", 1)),
                "context": "read",
            })

    # Handle Grep tool - extract matches
    elif tool_lower == "grep":
        # Extract file references from grep output
        for match in re.finditer(CODE_REF_PATTERN, output_str):
            result.code_refs.append({
                "file": match.group(1),
                "line": match.group(2),
                "context": "grep match",
            })

    # Handle Bash tool - detect errors
    elif tool_lower in ("bash", "bashoutput"):
        # Check for common error patterns
        if any(err in output_str.lower() for err in ["error", "failed", "exception", "traceback"]):
            lines = output_str.split("\n")
            for line in lines[:5]:  # First 5 lines of error
                if any(err in line.lower() for err in ["error", "failed", "exception"]):
                    result.errors.append(line.strip()[:200])

    # Handle Edit/Write - extract file changes
    elif tool_lower in ("edit", "write", "multiedit"):
        if isinstance(tool_input, dict) and "file_path" in tool_input:
            result.code_refs.append({
                "file": tool_input["file_path"],
                "line": "modified",
                "context": "edit",
            })
            result.facts.append(f"Modified {tool_input['file_path']}")

    # Also run pattern extraction on output
    pattern_data = extract_by_patterns(output_str)
    result = result.merge(pattern_data)

    return result


# === LLM-based extraction (optional, more accurate) ===

EXTRACTION_PROMPT = """Analyze the following observation and extract structured information.

Return a JSON object with the following fields (only include non-empty arrays):
- facts: Verified information, data points, versions, configurations
- decisions: Choices or decisions made
- patterns: Recurring themes or approaches observed
- todos: Action items or tasks to complete
- questions: Open questions or uncertainties
- insights: Key learnings or discoveries
- errors: Error messages or failures encountered

Observation:
{content}

JSON response (only the JSON, no markdown):"""


async def extract_with_llm(
    content: str,
    chat_provider: Any = None,
) -> Optional[StructuredData]:
    """Extract structured data using an LLM.

    More accurate than pattern matching but requires API calls.
    Use for batch processing or when accuracy is critical.

    Args:
        content: The observation content to analyze
        chat_provider: An LLM chat provider with a completion method

    Returns:
        StructuredData if extraction succeeds, None otherwise
    """
    if not chat_provider:
        return None

    try:
        import json

        prompt = EXTRACTION_PROMPT.format(content=content[:4000])  # Truncate long content

        # Call LLM (assumes chat_provider has a method for this)
        if hasattr(chat_provider, "complete"):
            response = await chat_provider.complete(prompt)
        elif hasattr(chat_provider, "generate"):
            response = chat_provider.generate(prompt)
        else:
            return None

        # Parse JSON response
        response_text = str(response).strip()

        # Handle markdown code blocks
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        data = json.loads(response_text)
        return StructuredData.from_dict(data)

    except Exception:
        return None


def combine_extraction(
    content: str,
    tool_name: str = None,
    tool_input: Any = None,
    tool_output: Any = None,
) -> StructuredData:
    """Combine multiple extraction methods for best results.

    Uses pattern matching on content plus tool-specific extraction.
    """
    result = StructuredData()

    # Pattern-based extraction from content
    if content:
        pattern_data = extract_by_patterns(content)
        result = result.merge(pattern_data)

    # Tool-specific extraction
    if tool_name:
        tool_data = extract_from_tool_output(tool_name, tool_input, tool_output)
        result = result.merge(tool_data)

    return result

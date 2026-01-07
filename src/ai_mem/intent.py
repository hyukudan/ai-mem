"""Intent Detection for ai-mem.

This module provides LLM-agnostic intent detection and keyword extraction
from user prompts to enable smart context retrieval.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


# Technical patterns to extract
TECHNICAL_PATTERNS = {
    # File paths
    "file_path": re.compile(r'(?:^|[\s"])([a-zA-Z_][\w/\-\.]+\.[a-z]{1,5})(?:[\s"\']|$)'),
    # Function/method names
    "function": re.compile(r'\b(?:function|def|fn|func|method)\s+(\w+)'),
    # Class names
    "class": re.compile(r'\b(?:class|struct|interface|type)\s+(\w+)'),
    # Variable assignments
    "variable": re.compile(r'\b(\w+)\s*[=:]='),
    # Import statements
    "import": re.compile(r'(?:import|from|require|include)\s+["\']?(\w+)'),
    # URLs/endpoints
    "endpoint": re.compile(r'/api/[\w/\-]+'),
    # Error codes/names
    "error": re.compile(r'\b([A-Z][a-z]+Error|Exception|Failure)\b'),
}

# Action verbs that indicate intent
ACTION_VERBS = {
    "create": ["create", "add", "new", "make", "build", "generate", "implement"],
    "fix": ["fix", "repair", "solve", "debug", "resolve", "patch", "correct"],
    "update": ["update", "modify", "change", "edit", "refactor", "improve"],
    "delete": ["delete", "remove", "drop", "clean", "clear"],
    "read": ["read", "show", "display", "get", "fetch", "find", "search", "look"],
    "explain": ["explain", "describe", "what", "how", "why", "tell"],
    "test": ["test", "verify", "check", "validate", "assert"],
    "deploy": ["deploy", "release", "publish", "ship", "push"],
}

# Technical concepts commonly discussed
TECH_CONCEPTS = {
    "auth", "authentication", "authorization", "login", "logout", "session",
    "api", "endpoint", "route", "controller", "middleware",
    "database", "db", "sql", "query", "migration", "schema",
    "cache", "redis", "memcached",
    "config", "configuration", "settings", "env", "environment",
    "test", "spec", "unittest", "pytest", "jest",
    "error", "exception", "bug", "issue", "fix",
    "performance", "optimization", "speed", "memory",
    "security", "encryption", "token", "jwt", "oauth",
    "frontend", "backend", "fullstack", "api",
    "component", "hook", "state", "props", "context",
    "model", "view", "controller", "service", "repository",
}


@dataclass
class Intent:
    """Represents detected intent from a user prompt."""
    action: Optional[str] = None  # Primary action (create, fix, update, etc.)
    entities: List[str] = field(default_factory=list)  # Files, functions, classes
    concepts: List[str] = field(default_factory=list)  # Technical concepts
    keywords: List[str] = field(default_factory=list)  # Important keywords
    query: str = ""  # Generated search query


def extract_technical_entities(text: str) -> List[str]:
    """Extract technical entities (files, functions, classes) from text.

    Args:
        text: User prompt text

    Returns:
        List of extracted entities
    """
    entities = []

    for pattern_name, pattern in TECHNICAL_PATTERNS.items():
        matches = pattern.findall(text)
        entities.extend(matches)

    return list(set(entities))  # Deduplicate


def detect_action(text: str) -> Optional[str]:
    """Detect the primary action from the text.

    Args:
        text: User prompt text

    Returns:
        Action category or None
    """
    text_lower = text.lower()

    for action, verbs in ACTION_VERBS.items():
        for verb in verbs:
            if verb in text_lower:
                return action

    return None


def extract_concepts(text: str) -> List[str]:
    """Extract technical concepts from text.

    Args:
        text: User prompt text

    Returns:
        List of detected concepts
    """
    words = set(text.lower().split())
    return [concept for concept in TECH_CONCEPTS if concept in words]


def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """Extract important keywords from text.

    Uses a simple TF-based approach without external dependencies.

    Args:
        text: User prompt text
        max_keywords: Maximum number of keywords to return

    Returns:
        List of keywords
    """
    # Tokenize and clean
    words = re.findall(r'\b[a-zA-Z_]\w{2,}\b', text.lower())

    # Filter common words
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "can", "need", "want", "like",
        "please", "help", "can", "you", "your", "this", "that",
        "for", "with", "from", "about", "into", "through", "just",
    }

    filtered = [w for w in words if w not in stopwords and len(w) > 2]

    # Count frequency
    freq = {}
    for word in filtered:
        freq[word] = freq.get(word, 0) + 1

    # Sort by frequency
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    return [word for word, _ in sorted_words[:max_keywords]]


def detect_intent(prompt: str) -> Intent:
    """Detect intent from a user prompt.

    Analyzes the prompt to extract:
    - Primary action (create, fix, update, etc.)
    - Technical entities (files, functions, classes)
    - Technical concepts
    - Important keywords
    - Generated search query

    Args:
        prompt: User prompt text

    Returns:
        Intent object with extracted information
    """
    action = detect_action(prompt)
    entities = extract_technical_entities(prompt)
    concepts = extract_concepts(prompt)
    keywords = extract_keywords(prompt)

    # Generate search query by combining entities, concepts, and keywords
    query_parts = []

    # Add entities first (most specific)
    query_parts.extend(entities[:3])

    # Add concepts
    query_parts.extend(concepts[:2])

    # Add keywords
    query_parts.extend(keywords[:3])

    # Deduplicate and join
    query = " ".join(dict.fromkeys(query_parts))

    return Intent(
        action=action,
        entities=entities,
        concepts=concepts,
        keywords=keywords,
        query=query,
    )


def should_inject_context(prompt: str) -> Tuple[bool, str]:
    """Determine if context should be injected for this prompt.

    Args:
        prompt: User prompt text

    Returns:
        Tuple of (should_inject, reason)
    """
    intent = detect_intent(prompt)

    # Always inject if there are technical entities
    if intent.entities:
        return True, f"Found entities: {', '.join(intent.entities[:3])}"

    # Inject if there are relevant concepts
    if intent.concepts:
        return True, f"Found concepts: {', '.join(intent.concepts[:3])}"

    # Inject for specific actions
    inject_actions = {"fix", "update", "explain", "read"}
    if intent.action in inject_actions:
        return True, f"Action '{intent.action}' may benefit from context"

    # Check if prompt is long enough to be substantive
    if len(prompt.split()) > 10:
        return True, "Substantive prompt"

    # Skip for simple greetings/commands
    simple_patterns = ["hi", "hello", "thanks", "bye", "help", "exit", "quit"]
    if prompt.lower().strip() in simple_patterns:
        return False, "Simple greeting/command"

    return True, "Default: inject context"


def generate_context_query(prompt: str, max_length: int = 100) -> str:
    """Generate a search query from a user prompt.

    This is a convenience function that extracts intent and returns
    the generated search query.

    Args:
        prompt: User prompt text
        max_length: Maximum query length

    Returns:
        Search query string
    """
    intent = detect_intent(prompt)
    query = intent.query

    if len(query) > max_length:
        query = query[:max_length].rsplit(" ", 1)[0]

    return query.strip() or prompt[:50]

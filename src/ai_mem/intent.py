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


# =============================================================================
# Query Expansion
# =============================================================================

# Technical synonyms for query expansion (LLM-agnostic)
TECHNICAL_SYNONYMS = {
    # Authentication
    "auth": ["authentication", "authorization", "login", "signin", "oauth", "jwt"],
    "authentication": ["auth", "login", "signin", "oauth", "jwt", "session"],
    "login": ["signin", "auth", "authentication", "logon"],
    "logout": ["signout", "logoff", "session-end"],

    # Database
    "database": ["db", "datastore", "storage", "persistence"],
    "db": ["database", "datastore", "storage"],
    "sql": ["query", "database", "mysql", "postgres", "sqlite"],
    "query": ["sql", "search", "find", "select", "fetch"],

    # API
    "api": ["endpoint", "rest", "graphql", "service", "route"],
    "endpoint": ["api", "route", "handler", "controller"],
    "rest": ["api", "restful", "http", "endpoint"],

    # Error handling
    "error": ["exception", "failure", "bug", "issue", "problem"],
    "exception": ["error", "throw", "catch", "failure"],
    "bug": ["error", "issue", "defect", "problem", "fix"],
    "fix": ["repair", "solve", "patch", "correct", "bugfix"],

    # Testing
    "test": ["spec", "unittest", "pytest", "jest", "testing", "check"],
    "unittest": ["test", "pytest", "testing", "spec"],

    # Code structure
    "function": ["method", "func", "def", "procedure", "routine"],
    "method": ["function", "func", "member", "procedure"],
    "class": ["type", "struct", "object", "model"],
    "component": ["widget", "element", "module", "part"],

    # Frontend
    "frontend": ["client", "ui", "view", "browser", "react", "vue"],
    "backend": ["server", "api", "service", "serverside"],
    "ui": ["interface", "frontend", "view", "display", "ux"],

    # Config
    "config": ["configuration", "settings", "options", "preferences", "env"],
    "settings": ["config", "configuration", "options", "preferences"],
    "env": ["environment", "config", "dotenv", "envvar"],

    # Performance
    "performance": ["speed", "optimization", "perf", "latency", "fast"],
    "cache": ["caching", "memoize", "redis", "memcached", "store"],
    "optimize": ["performance", "speed", "improve", "enhance"],

    # Security
    "security": ["auth", "encryption", "secure", "protection", "safety"],
    "encrypt": ["encryption", "hash", "cipher", "secure"],
    "token": ["jwt", "bearer", "session", "auth"],
}

# Case variations for technical terms
CASE_VARIATIONS = {
    # CamelCase / snake_case / kebab-case patterns
    "get_user": ["getUser", "GetUser", "get-user"],
    "set_value": ["setValue", "SetValue", "set-value"],
    "is_valid": ["isValid", "IsValid", "is-valid"],
    "has_error": ["hasError", "HasError", "has-error"],
    "on_click": ["onClick", "OnClick", "on-click"],
    "api_key": ["apiKey", "ApiKey", "api-key", "API_KEY"],
    "user_id": ["userId", "UserId", "user-id", "USER_ID"],
}


@dataclass
class ExpandedQuery:
    """Represents an expanded query with variants."""
    original: str
    expanded_terms: List[str] = field(default_factory=list)
    all_queries: List[str] = field(default_factory=list)
    expansion_count: int = 0


def expand_term(term: str, max_synonyms: int = 3) -> List[str]:
    """Expand a single term with synonyms and variations.

    Args:
        term: Term to expand
        max_synonyms: Maximum number of synonyms to include

    Returns:
        List of expanded terms including original
    """
    expanded = [term]
    term_lower = term.lower()

    # Add synonyms
    if term_lower in TECHNICAL_SYNONYMS:
        synonyms = TECHNICAL_SYNONYMS[term_lower][:max_synonyms]
        expanded.extend(synonyms)

    # Add case variations
    if term_lower in CASE_VARIATIONS:
        expanded.extend(CASE_VARIATIONS[term_lower])

    # Generate common case variations for unknown terms
    if len(term) > 2:
        # snake_case to camelCase
        if "_" in term:
            parts = term.split("_")
            camel = parts[0].lower() + "".join(p.title() for p in parts[1:])
            expanded.append(camel)
        # camelCase to snake_case
        elif any(c.isupper() for c in term[1:]):
            snake = re.sub(r'([A-Z])', r'_\1', term).lower().lstrip('_')
            expanded.append(snake)

    return list(dict.fromkeys(expanded))  # Deduplicate preserving order


def expand_query(
    query: str,
    max_synonyms_per_term: int = 2,
    max_total_terms: int = 15,
    include_case_variants: bool = True,
) -> ExpandedQuery:
    """Expand a search query with synonyms and variations.

    This LLM-agnostic expansion:
    1. Adds technical synonyms (e.g., "auth" → "authentication", "login")
    2. Adds case variations (e.g., "get_user" → "getUser", "GetUser")
    3. Preserves original terms as highest priority

    Args:
        query: Original search query
        max_synonyms_per_term: Max synonyms per term
        max_total_terms: Max total terms in expanded query
        include_case_variants: Include case variations

    Returns:
        ExpandedQuery with original and expanded terms
    """
    if not query:
        return ExpandedQuery(original="", all_queries=[""])

    words = query.lower().split()
    expanded_terms: List[str] = []
    original_terms: List[str] = []

    for word in words:
        # Skip very short words
        if len(word) < 2:
            continue

        original_terms.append(word)

        # Expand the term
        expansions = expand_term(word, max_synonyms_per_term)
        for exp in expansions:
            if exp not in expanded_terms:
                expanded_terms.append(exp)

                if len(expanded_terms) >= max_total_terms:
                    break

        if len(expanded_terms) >= max_total_terms:
            break

    # Generate query variants (original + expansions)
    all_queries = [query]  # Original query first

    # Add expanded query (all terms)
    if expanded_terms:
        expanded_query = " ".join(expanded_terms)
        if expanded_query != query:
            all_queries.append(expanded_query)

    # Add individual synonym-based queries for important terms
    for term in original_terms[:3]:  # Top 3 original terms
        if term in TECHNICAL_SYNONYMS:
            for synonym in TECHNICAL_SYNONYMS[term][:2]:
                variant = query.replace(term, synonym)
                if variant not in all_queries:
                    all_queries.append(variant)

    return ExpandedQuery(
        original=query,
        expanded_terms=expanded_terms,
        all_queries=all_queries[:5],  # Max 5 query variants
        expansion_count=len(expanded_terms) - len(original_terms),
    )


def generate_expanded_queries(
    prompt: str,
    max_queries: int = 3,
) -> List[str]:
    """Generate multiple search queries from a prompt using expansion.

    This is the main entry point for query expansion. It:
    1. Extracts intent from the prompt
    2. Expands the generated query with synonyms
    3. Returns multiple query variants for broader recall

    Args:
        prompt: User prompt text
        max_queries: Maximum number of queries to generate

    Returns:
        List of query strings (original + expanded variants)
    """
    base_query = generate_context_query(prompt)
    expanded = expand_query(base_query)
    return expanded.all_queries[:max_queries]

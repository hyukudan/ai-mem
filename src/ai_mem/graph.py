"""Graph Memory for ai-mem.

This module provides entity extraction and relationship tracking
to enable multi-hop reasoning and knowledge graph queries.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import logging

logger = logging.getLogger("ai_mem.graph")


class EntityType(str, Enum):
    """Types of entities that can be extracted."""
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    VARIABLE = "variable"
    ENDPOINT = "endpoint"
    ERROR = "error"
    CONCEPT = "concept"
    PERSON = "person"
    TECHNOLOGY = "technology"


class RelationType(str, Enum):
    """Types of relationships between entities."""
    MENTIONS = "mentions"  # Observation mentions entity
    USES = "uses"  # Entity A uses entity B
    DEFINES = "defines"  # File defines function/class
    IMPORTS = "imports"  # Module imports another
    CALLS = "calls"  # Function calls another
    EXTENDS = "extends"  # Class extends another
    IMPLEMENTS = "implements"  # Class implements interface
    RELATES_TO = "relates_to"  # Generic relationship
    DEPENDS_ON = "depends_on"  # Dependency relationship
    MODIFIES = "modifies"  # Entity modifies another
    FIXES = "fixes"  # Bugfix relationship
    PART_OF = "part_of"  # Component relationship


@dataclass
class Entity:
    """Represents an extracted entity."""
    id: str = ""
    name: str = ""
    entity_type: EntityType = EntityType.CONCEPT
    project: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    mention_count: int = 0
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None


@dataclass
class EntityRelation:
    """Represents a relationship between two entities."""
    id: str = ""
    source_entity_id: str = ""
    target_entity_id: str = ""
    relation_type: RelationType = RelationType.RELATES_TO
    observation_id: Optional[str] = None  # Source observation
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[float] = None


# Entity extraction patterns
ENTITY_PATTERNS = {
    EntityType.FILE: [
        # File paths with extensions
        re.compile(r'(?:^|[\s"\'])([a-zA-Z_][\w/\-\.]*\.[a-z]{1,5})(?:[\s"\']|$|:|\))', re.MULTILINE),
        # Relative paths
        re.compile(r'(?:in|from|to|file|path)\s+["\']?([a-zA-Z_][\w/\-\.]+\.[a-z]{1,5})["\']?', re.IGNORECASE),
    ],
    EntityType.FUNCTION: [
        # Function definitions
        re.compile(r'(?:function|def|fn|func|method)\s+(\w+)\s*\(', re.IGNORECASE),
        # Function calls
        re.compile(r'(\w+)\s*\([^)]*\)'),
        # Method references
        re.compile(r'\.(\w+)\s*\('),
    ],
    EntityType.CLASS: [
        # Class definitions
        re.compile(r'(?:class|struct|interface|type)\s+(\w+)', re.IGNORECASE),
        # TypeScript/Java style
        re.compile(r'(?:extends|implements)\s+(\w+)', re.IGNORECASE),
    ],
    EntityType.MODULE: [
        # Import statements
        re.compile(r'(?:import|from|require)\s+["\']?(\w+(?:\.\w+)*)["\']?', re.IGNORECASE),
        # Package references
        re.compile(r'(?:package|module)\s+(\w+(?:\.\w+)*)', re.IGNORECASE),
    ],
    EntityType.ENDPOINT: [
        # API endpoints
        re.compile(r'(/api/[\w/\-]+)'),
        # HTTP methods with paths
        re.compile(r'(?:GET|POST|PUT|DELETE|PATCH)\s+(/[\w/\-]+)', re.IGNORECASE),
        # Route definitions
        re.compile(r'(?:route|path|endpoint)\s*[=:]\s*["\']?(/[\w/\-]+)["\']?', re.IGNORECASE),
    ],
    EntityType.ERROR: [
        # Error names
        re.compile(r'\b([A-Z][a-z]+(?:Error|Exception|Failure))\b'),
        # Error codes
        re.compile(r'(?:error|err)\s*[=:]\s*["\']?(\w+)["\']?', re.IGNORECASE),
    ],
    EntityType.TECHNOLOGY: [
        # Common technologies (case-insensitive matching)
        re.compile(r'\b(React|Vue|Angular|Django|Flask|FastAPI|Express|Node\.?js|Python|JavaScript|TypeScript|Go|Rust|Java|PostgreSQL|MySQL|MongoDB|Redis|Docker|Kubernetes|AWS|GCP|Azure)\b', re.IGNORECASE),
    ],
}

# Relationship extraction patterns
RELATION_PATTERNS = {
    RelationType.USES: [
        re.compile(r'(\w+)\s+(?:uses?|using|utilizes?)\s+(\w+)', re.IGNORECASE),
        re.compile(r'(\w+)\s+(?:depends?\s+on|requires?)\s+(\w+)', re.IGNORECASE),
    ],
    RelationType.CALLS: [
        re.compile(r'(\w+)\s+(?:calls?|invokes?)\s+(\w+)', re.IGNORECASE),
        re.compile(r'(\w+)\s*\(\).*(?:calls?|triggers?)\s+(\w+)', re.IGNORECASE),
    ],
    RelationType.IMPORTS: [
        re.compile(r'(?:import|from)\s+(\w+).*(?:import|from)\s+(\w+)', re.IGNORECASE),
    ],
    RelationType.EXTENDS: [
        re.compile(r'(\w+)\s+(?:extends?|inherits?\s+from)\s+(\w+)', re.IGNORECASE),
    ],
    RelationType.IMPLEMENTS: [
        re.compile(r'(\w+)\s+(?:implements?)\s+(\w+)', re.IGNORECASE),
    ],
    RelationType.FIXES: [
        re.compile(r'(?:fix(?:ed|es)?|resolv(?:ed|es)?|patch(?:ed|es)?)\s+(?:the\s+)?(\w+)', re.IGNORECASE),
    ],
    RelationType.MODIFIES: [
        re.compile(r'(?:modif(?:ied|ies)|chang(?:ed|es)|updat(?:ed|es))\s+(\w+)', re.IGNORECASE),
    ],
}

# Common stopwords for entity extraction
ENTITY_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "can", "need", "want", "like",
    "this", "that", "these", "those", "it", "its", "they", "them",
    "we", "us", "our", "you", "your", "he", "she", "him", "her",
    "true", "false", "null", "none", "undefined", "return",
    "if", "else", "for", "while", "try", "catch", "finally",
    "new", "get", "set", "add", "remove", "delete", "update",
    "function", "class", "def", "var", "let", "const", "import",
}


def extract_entities(
    text: str,
    entity_types: Optional[List[EntityType]] = None,
    min_length: int = 2,
    max_entities: int = 50,
) -> List[Tuple[str, EntityType]]:
    """Extract entities from text.

    Args:
        text: Text to extract entities from
        entity_types: Types to extract (None = all)
        min_length: Minimum entity name length
        max_entities: Maximum entities to return

    Returns:
        List of (entity_name, entity_type) tuples
    """
    if not text:
        return []

    types_to_check = entity_types or list(EntityType)
    entities: List[Tuple[str, EntityType]] = []
    seen: Set[str] = set()

    for entity_type in types_to_check:
        patterns = ENTITY_PATTERNS.get(entity_type, [])
        for pattern in patterns:
            matches = pattern.findall(text)
            for match in matches:
                # Handle tuple matches from groups
                name = match if isinstance(match, str) else match[0] if match else ""
                name = name.strip()

                # Filter
                if len(name) < min_length:
                    continue
                if name.lower() in ENTITY_STOPWORDS:
                    continue
                if name.lower() in seen:
                    continue

                seen.add(name.lower())
                entities.append((name, entity_type))

                if len(entities) >= max_entities:
                    return entities

    return entities


def extract_relations(
    text: str,
    entities: List[Tuple[str, EntityType]],
    max_relations: int = 20,
) -> List[Tuple[str, str, RelationType]]:
    """Extract relationships between entities from text.

    Args:
        text: Text to extract relations from
        entities: Previously extracted entities
        max_relations: Maximum relations to return

    Returns:
        List of (source_name, target_name, relation_type) tuples
    """
    if not text or not entities:
        return []

    relations: List[Tuple[str, str, RelationType]] = []
    entity_names = {e[0].lower() for e in entities}

    # Try pattern-based extraction
    for relation_type, patterns in RELATION_PATTERNS.items():
        for pattern in patterns:
            matches = pattern.findall(text)
            for match in matches:
                if len(match) >= 2:
                    source, target = match[0], match[1]
                    if source.lower() in entity_names or target.lower() in entity_names:
                        relations.append((source, target, relation_type))
                        if len(relations) >= max_relations:
                            return relations

    # Infer MENTIONS relations from co-occurrence
    for i, (name1, type1) in enumerate(entities):
        for name2, type2 in entities[i + 1:]:
            if name1.lower() != name2.lower():
                relations.append((name1, name2, RelationType.RELATES_TO))
                if len(relations) >= max_relations:
                    return relations

    return relations


def extract_concepts(
    text: str,
    max_concepts: int = 10,
) -> List[str]:
    """Extract high-level concepts from text.

    Args:
        text: Text to extract concepts from
        max_concepts: Maximum concepts to return

    Returns:
        List of concept strings
    """
    # Technical concept keywords
    concept_keywords = {
        "authentication", "authorization", "auth", "login", "logout", "session",
        "api", "endpoint", "route", "controller", "middleware", "handler",
        "database", "db", "sql", "query", "migration", "schema", "model",
        "cache", "caching", "redis", "memcached",
        "config", "configuration", "settings", "env", "environment",
        "test", "testing", "spec", "unittest", "integration",
        "error", "exception", "bug", "issue", "fix", "debug",
        "performance", "optimization", "speed", "memory", "latency",
        "security", "encryption", "token", "jwt", "oauth", "ssl",
        "frontend", "backend", "fullstack", "client", "server",
        "component", "hook", "state", "props", "context", "redux",
        "service", "repository", "factory", "singleton", "pattern",
        "deployment", "ci", "cd", "pipeline", "docker", "kubernetes",
        "logging", "monitoring", "metrics", "tracing", "observability",
        "validation", "sanitization", "input", "output",
        "async", "await", "promise", "callback", "event",
        "refactor", "refactoring", "cleanup", "improvement",
    }

    words = set(text.lower().split())
    concepts = [c for c in concept_keywords if c in words]

    return concepts[:max_concepts]


class EntityGraph:
    """Manager for entity graph operations.

    This class handles entity extraction, storage, and graph queries.
    It works with the DatabaseManager for persistence.
    """

    def __init__(self, db=None):
        """Initialize EntityGraph.

        Args:
            db: DatabaseManager instance
        """
        self.db = db
        self._entity_cache: Dict[str, Entity] = {}

    async def extract_and_store(
        self,
        text: str,
        observation_id: str,
        project: str,
        created_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Extract entities and relations from text and store them.

        Args:
            text: Text to extract from
            observation_id: Source observation ID
            project: Project identifier
            created_at: Timestamp

        Returns:
            Dict with extracted entities and relations counts
        """
        # Extract entities
        raw_entities = extract_entities(text)
        concepts = extract_concepts(text)

        # Add concepts as CONCEPT entities
        for concept in concepts:
            if not any(e[0].lower() == concept for e in raw_entities):
                raw_entities.append((concept, EntityType.CONCEPT))

        # Extract relations
        raw_relations = extract_relations(text, raw_entities)

        # Store entities
        stored_entities = 0
        entity_id_map: Dict[str, str] = {}

        for name, entity_type in raw_entities:
            entity_id = await self._get_or_create_entity(
                name=name,
                entity_type=entity_type,
                project=project,
                created_at=created_at,
            )
            if entity_id:
                entity_id_map[name.lower()] = entity_id
                stored_entities += 1

                # Create MENTIONS relation
                await self._create_relation(
                    source_id=observation_id,
                    target_id=entity_id,
                    relation_type=RelationType.MENTIONS,
                    observation_id=observation_id,
                    created_at=created_at,
                )

        # Store relations
        stored_relations = 0
        for source_name, target_name, relation_type in raw_relations:
            source_id = entity_id_map.get(source_name.lower())
            target_id = entity_id_map.get(target_name.lower())

            if source_id and target_id and source_id != target_id:
                await self._create_relation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    observation_id=observation_id,
                    created_at=created_at,
                )
                stored_relations += 1

        return {
            "entities_extracted": len(raw_entities),
            "entities_stored": stored_entities,
            "relations_extracted": len(raw_relations),
            "relations_stored": stored_relations,
            "concepts": concepts,
        }

    async def _get_or_create_entity(
        self,
        name: str,
        entity_type: EntityType,
        project: str,
        created_at: Optional[float] = None,
    ) -> Optional[str]:
        """Get existing entity or create new one.

        Args:
            name: Entity name
            entity_type: Entity type
            project: Project identifier
            created_at: Timestamp

        Returns:
            Entity ID or None
        """
        if not self.db:
            return None

        return await self.db.get_or_create_entity(
            name=name,
            entity_type=entity_type.value,
            project=project,
            created_at=created_at,
        )

    async def _create_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        observation_id: Optional[str] = None,
        confidence: float = 1.0,
        created_at: Optional[float] = None,
    ) -> Optional[str]:
        """Create a relation between entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relation_type: Type of relation
            observation_id: Source observation ID
            confidence: Confidence score
            created_at: Timestamp

        Returns:
            Relation ID or None
        """
        if not self.db:
            return None

        return await self.db.create_entity_relation(
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type=relation_type.value,
            observation_id=observation_id,
            confidence=confidence,
            created_at=created_at,
        )

    async def get_related_entities(
        self,
        entity_id: str,
        relation_types: Optional[List[RelationType]] = None,
        max_depth: int = 1,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get entities related to a given entity.

        Args:
            entity_id: Starting entity ID
            relation_types: Filter by relation types
            max_depth: Maximum traversal depth (1 = direct, 2+ = multi-hop)
            limit: Maximum entities to return

        Returns:
            List of related entity dicts
        """
        if not self.db:
            return []

        type_values = [rt.value for rt in relation_types] if relation_types else None
        return await self.db.get_related_entities(
            entity_id=entity_id,
            relation_types=type_values,
            max_depth=max_depth,
            limit=limit,
        )

    async def find_entities(
        self,
        query: str,
        project: Optional[str] = None,
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search for entities by name.

        Args:
            query: Search query
            project: Filter by project
            entity_types: Filter by types
            limit: Maximum entities to return

        Returns:
            List of matching entity dicts
        """
        if not self.db:
            return []

        type_values = [et.value for et in entity_types] if entity_types else None
        return await self.db.search_entities(
            query=query,
            project=project,
            entity_types=type_values,
            limit=limit,
        )

    async def get_entity_observations(
        self,
        entity_id: str,
        limit: int = 20,
    ) -> List[str]:
        """Get observation IDs that mention an entity.

        Args:
            entity_id: Entity ID
            limit: Maximum observations to return

        Returns:
            List of observation IDs
        """
        if not self.db:
            return []

        return await self.db.get_entity_observations(
            entity_id=entity_id,
            limit=limit,
        )

    async def get_graph_stats(
        self,
        project: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get statistics about the entity graph.

        Args:
            project: Filter by project

        Returns:
            Dict with graph statistics
        """
        if not self.db:
            return {"entities": 0, "relations": 0}

        return await self.db.get_graph_stats(project=project)

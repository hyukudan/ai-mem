from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
import time
import uuid

ObservationType = Literal[
    "decision",
    "bugfix",
    "feature",
    "refactor",
    "discovery",
    "change",
    "interaction",
    "tool_output",
    "file_content",
    "note",
    "summary",
    # User-level memory types (Phase 4)
    "preference",
    "convention",
]

# Concepts categorize the semantic meaning of an observation
# These are orthogonal to types (you can have a bugfix that's also a gotcha)
ConceptType = Literal[
    # Knowledge capture
    "how-it-works",       # Explains how something functions
    "why-it-exists",      # Explains the rationale/history
    "what-changed",       # Documents what was modified
    # Problem-solution patterns
    "problem-solution",   # Problem + its solution
    "gotcha",            # Pitfall, trap, or common mistake (important!)
    "workaround",        # Temporary fix or hack
    # Patterns and practices
    "pattern",           # Reusable pattern or best practice
    "anti-pattern",      # What NOT to do
    "trade-off",         # Decision with pros/cons
    # Architecture and design
    "architecture",      # System/component design
    "interface",         # API or contract definition
    "dependency",        # External dependency info
]

# Icon mapping for concepts (for UI and context display)
CONCEPT_ICONS = {
    "how-it-works": "\u2139\ufe0f",      # Info
    "why-it-exists": "\ud83d\udcdc",     # Scroll
    "what-changed": "\ud83d\udcdd",      # Memo
    "problem-solution": "\u2705",         # Check
    "gotcha": "\ud83d\udd34",            # Red circle (important!)
    "workaround": "\u26a0\ufe0f",         # Warning
    "pattern": "\ud83d\udca1",           # Lightbulb
    "anti-pattern": "\u274c",             # X
    "trade-off": "\u2696\ufe0f",          # Balance scale
    "architecture": "\ud83c\udfd7\ufe0f", # Building construction
    "interface": "\ud83d\udd0c",          # Plug
    "dependency": "\ud83d\udd17",         # Link
}

# Icon mapping for observation types
TYPE_ICONS = {
    "bugfix": "\ud83d\udc1e",            # Bug
    "feature": "\u2728",                  # Sparkles
    "decision": "\ud83e\udd14",          # Thinking face
    "refactor": "\u267b\ufe0f",           # Recycle
    "discovery": "\ud83d\udd0d",         # Magnifying glass
    "change": "\ud83d\udd04",            # Arrows
    "note": "\ud83d\udcdd",              # Memo
    "code_review": "\ud83d\udc40",       # Eyes
    "test": "\ud83e\uddea",              # Test tube
    "tool_output": "\ud83d\udee0\ufe0f", # Tools
    "interaction": "\ud83d\udcac",       # Speech bubble
    "preference": "\u2764\ufe0f",         # Heart
    "summary": "\ud83d\udcca",           # Chart
}

class ObservationAsset(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    observation_id: Optional[str] = None
    type: str = "file"
    name: Optional[str] = None
    path: Optional[str] = None
    content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class Observation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    project: str
    type: ObservationType
    concept: Optional[ConceptType] = None  # Semantic categorization (gotcha, trade-off, etc.)
    title: Optional[str] = None
    content: str
    summary: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    importance_score: float = 0.5
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content_hash: Optional[str] = None
    diff: Optional[str] = None
    assets: List[ObservationAsset] = Field(default_factory=list)

    def get_icon(self) -> str:
        """Get display icon based on concept (if set) or type."""
        if self.concept and self.concept in CONCEPT_ICONS:
            return CONCEPT_ICONS[self.concept]
        # Default type icons
        type_icons = {
            "bugfix": "\ud83d\udc1e",        # Bug
            "feature": "\u2728",              # Sparkles
            "decision": "\ud83e\udd14",       # Thinking
            "refactor": "\u267b\ufe0f",       # Recycle
            "discovery": "\ud83d\udd0d",      # Magnifying glass
            "tool_output": "\ud83d\udee0\ufe0f",  # Wrench
            "note": "\ud83d\udcdd",           # Memo
            "summary": "\ud83d\udcca",        # Chart
        }
        return type_icons.get(self.type, "\ud83d\udccc")  # Default: pushpin

class ObservationIndex(BaseModel):
    id: str
    summary: str
    project: str
    type: Optional[str] = None
    created_at: float
    score: float = 0.0
    fts_score: Optional[float] = None
    vector_score: Optional[float] = None
    recency_factor: Optional[float] = None
    rerank_score: Optional[float] = None  # Two-stage retrieval rerank score

class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project: str
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    goal: Optional[str] = None
    summary: Optional[str] = None

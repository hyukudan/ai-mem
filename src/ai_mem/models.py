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
]

class Observation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    project: str
    type: ObservationType
    title: Optional[str] = None
    content: str
    summary: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    importance_score: float = 0.5
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content_hash: Optional[str] = None

class ObservationIndex(BaseModel):
    id: str
    summary: str
    project: str
    type: Optional[str] = None
    created_at: float
    score: float = 0.0

class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project: str
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    goal: Optional[str] = None
    summary: Optional[str] = None

"""Core data models for EdgeMemory."""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
from uuid import uuid4


class CausalMemoryObject(BaseModel):
    """The core primitive for all memories."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    content: str
    type: Literal["episodic", "semantic", "procedural", "reflective"] = "episodic"

    # Classification metadata
    topic: Optional[str] = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    emotion: Optional[str] = None

    # Causal links
    causes: List[str] = Field(default_factory=list)
    effects: List[str] = Field(default_factory=list)

    # Evolution tracking
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None

    # Entity extraction
    entities: List[str] = Field(default_factory=list)

    # Embedding reference
    embedding_id: Optional[str] = None


class MemoryQuery(BaseModel):
    """Structured query for memory retrieval."""
    raw_query: str
    intent: Optional[str] = None
    entities: List[str] = Field(default_factory=list)
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None
    memory_types: List[str] = Field(default_factory=list)
    abstraction_level: Literal["specific", "summary", "pattern"] = "specific"


class BeliefDelta(BaseModel):
    """Tracks how a belief/understanding changed over time."""
    topic: str
    old_belief_id: str
    new_belief_id: str
    change_timestamp: datetime = Field(default_factory=datetime.now)
    cause_event_ids: List[str] = Field(default_factory=list)
    change_type: Literal["contradiction", "refinement", "expansion", "abandonment"]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

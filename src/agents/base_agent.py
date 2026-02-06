"""Base agent framework for EdgeMemory."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.llm.local_llm import LocalLLM
from src.retrieval.fusion import HybridRetriever, RetrievalResult


@dataclass
class AgentResponse:
    """Standardized agent response."""
    answer: str
    evidence: List[RetrievalResult] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    confidence: float = 0.5
    follow_up_queries: Optional[List[str]] = None


class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(self, llm: LocalLLM, retriever: HybridRetriever, name: str = "BaseAgent"):
        self.llm = llm
        self.retriever = retriever
        self.name = name
        self.reasoning_trace: List[str] = []

    @abstractmethod
    def process(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        pass

    def _log_reasoning(self, step: str):
        self.reasoning_trace.append(f"[{self.name}] {step}")

    def _retrieve(self, query: str, query_embedding=None, **kwargs) -> List[RetrievalResult]:
        return self.retriever.retrieve(query=query, query_embedding=query_embedding, **kwargs)

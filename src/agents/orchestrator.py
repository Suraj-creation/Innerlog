"""Agent orchestrator — intelligent query routing and multi-agent coordination."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re

from src.agents.base_agent import AgentResponse
from src.agents.timeline_agent import TimelineAgent
from src.agents.causal_agent import CausalAgent
from src.agents.arbitration_agent import ArbitrationAgent
from src.llm.local_llm import LocalLLM
from src.llm.prompts import INTENT_CLASSIFICATION_PROMPT, SYNTHESIS_PROMPT
from src.retrieval.fusion import HybridRetriever
from src.utils import get_logger

logger = get_logger("orchestrator")

# Intent detection patterns (fast path)
INTENT_PATTERNS = {
    "timeline": [
        r"\b(when|what happened|timeline|last week|yesterday|this month|today|last year)\b",
        r"\b(chronolog|sequence of events|history of|over time)\b",
    ],
    "causal": [
        r"\b(why|because|caused|led to|result of|consequence|reason)\b",
        r"\b(what made me|reason for|how did .* affect)\b",
    ],
    "reflection": [
        r"\b(how have i changed|evolution of|used to think|changed my mind)\b",
        r"\b(progression of|belief about|my view on)\b",
    ],
    "conflict": [
        r"\b(contradict|conflict|inconsistent|but .* said|disagree)\b",
    ],
}


@dataclass
class OrchestratorResponse:
    """Response from the orchestrator."""
    answer: str
    evidence: List[Any] = field(default_factory=list)
    agent_trace: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    processing_time_ms: float = 0.0
    agent_used: str = "unknown"


class AgentOrchestrator:
    """Intelligent query routing and multi-agent coordination."""

    def __init__(
        self,
        llm: LocalLLM,
        retriever: HybridRetriever,
        timeline_agent: TimelineAgent,
        causal_agent: CausalAgent,
        arbitration_agent: ArbitrationAgent,
    ):
        self.llm = llm
        self.retriever = retriever
        self.agents = {
            "timeline": timeline_agent,
            "causal": causal_agent,
            "arbitration": arbitration_agent,
        }

    def process_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> OrchestratorResponse:
        """Process a query using intelligent routing."""
        start = datetime.now()
        context = context or {}
        agent_trace: List[Dict[str, Any]] = []

        # Classify intent (fast path first)
        intent, confidence = self._classify_intent_fast(query)
        agent_trace.append({"step": "intent_classification", "intent": intent, "confidence": confidence})

        # Route to agent
        agent_name = self._route_to_agent(intent)
        agent = self.agents.get(agent_name, self.agents["timeline"])

        try:
            response = agent.process(query, context)
            agent_trace.append({
                "step": "execution", "agent": agent_name,
                "success": True, "confidence": response.confidence,
            })
        except Exception as e:
            logger.error(f"Agent {agent_name} failed: {e}")
            agent_trace.append({"step": "execution", "agent": agent_name, "success": False, "error": str(e)})
            response = self._fallback_response(query, agent_name, context, agent_trace)

        elapsed = (datetime.now() - start).total_seconds() * 1000

        return OrchestratorResponse(
            answer=response.answer,
            evidence=response.evidence,
            agent_trace=agent_trace,
            confidence=response.confidence,
            processing_time_ms=elapsed,
            agent_used=agent_name,
        )

    def _classify_intent_fast(self, query: str) -> Tuple[str, float]:
        """Fast pattern-based intent classification."""
        query_lower = query.lower()
        scores: Dict[str, int] = {}
        for intent, patterns in INTENT_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, query_lower))
            scores[intent] = score

        if max(scores.values(), default=0) == 0:
            return "timeline", 0.5  # Default

        best = max(scores, key=scores.get)
        conf = min(scores[best] / 2, 1.0)
        return best, conf

    def _route_to_agent(self, intent: str) -> str:
        """Map intent to agent."""
        mapping = {
            "timeline": "timeline",
            "causal": "causal",
            "conflict": "arbitration",
            "reflection": "arbitration",
        }
        return mapping.get(intent, "timeline")

    def _fallback_response(self, query, failed_agent, context, trace) -> AgentResponse:
        """Try fallback agents."""
        for name, agent in self.agents.items():
            if name == failed_agent:
                continue
            try:
                response = agent.process(query, context)
                trace.append({"step": "fallback", "agent": name, "success": True})
                return response
            except Exception:
                continue

        return AgentResponse(
            answer="I was unable to process this query. Please try rephrasing.",
            evidence=[], reasoning_trace=["All agents failed"], confidence=0.1,
        )

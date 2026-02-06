"""Timeline agent for temporal queries."""

from src.agents.base_agent import BaseAgent, AgentResponse
from src.llm.prompts import TIMELINE_NARRATIVE_PROMPT
from typing import Dict, Any, List
import json


class TimelineAgent(BaseAgent):
    """Agent for timeline-based queries. Handles 'when' and 'what happened' questions."""

    def __init__(self, llm, retriever, memory_db=None, embedding_model=None):
        super().__init__(llm, retriever, name="TimelineAgent")
        self.memory_db = memory_db
        self.embedding_model = embedding_model

    def process(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        self.reasoning_trace = []
        self._log_reasoning(f"Processing timeline query: {query}")

        # Get query embedding for dense retrieval
        query_embedding = None
        if self.embedding_model:
            query_embedding = self.embedding_model.encode_single(query)

        # Retrieve relevant memories
        results = self._retrieve(query, query_embedding=query_embedding, k=15)
        self._log_reasoning(f"Retrieved {len(results)} memories")

        # Enrich with content from DB
        enriched = self._enrich_results(results)

        # Build timeline narrative
        evidence_text = "\n".join([
            f"- [{r.metadata.get('timestamp', 'unknown') if r.metadata else 'unknown'}] {r.content}"
            for r in enriched[:10] if r.content
        ])

        if not evidence_text.strip():
            return AgentResponse(
                answer="I don't have enough memories to answer this timeline question yet.",
                evidence=enriched,
                reasoning_trace=self.reasoning_trace,
                confidence=0.2,
            )

        prompt = TIMELINE_NARRATIVE_PROMPT.format(
            query=query, time_focus="general", evidence=evidence_text
        )
        answer = self.llm.generate(prompt)

        return AgentResponse(
            answer=answer,
            evidence=enriched[:10],
            reasoning_trace=self.reasoning_trace,
            confidence=0.8 if results else 0.3,
        )

    def _enrich_results(self, results) -> list:
        """Add content from database to retrieval results."""
        if not self.memory_db:
            return results
        for r in results:
            mem = self.memory_db.get_memory(r.memory_id)
            if mem:
                r.content = mem.get("content", "")
                r.metadata = mem
        return results

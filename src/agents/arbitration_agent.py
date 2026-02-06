"""Arbitration agent for resolving conflicting memories."""

from src.agents.base_agent import BaseAgent, AgentResponse
from typing import Dict, Any, List


class ArbitrationAgent(BaseAgent):
    """Agent for resolving conflicting memories and belief changes."""

    def __init__(self, llm, retriever, memory_db=None, embedding_model=None):
        super().__init__(llm, retriever, name="ArbitrationAgent")
        self.memory_db = memory_db
        self.embedding_model = embedding_model

    def process(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        self.reasoning_trace = []
        self._log_reasoning(f"Processing conflict query: {query}")

        query_embedding = None
        if self.embedding_model:
            query_embedding = self.embedding_model.encode_single(query)

        results = self._retrieve(query, query_embedding=query_embedding, k=20)

        # Enrich
        if self.memory_db:
            for r in results:
                mem = self.memory_db.get_memory(r.memory_id)
                if mem:
                    r.content = mem.get("content", "")
                    r.metadata = mem

        evidence_list = [r.content for r in results[:10] if r.content]

        if not evidence_list:
            return AgentResponse(
                answer="I don't have enough information to resolve this question.",
                evidence=results,
                reasoning_trace=self.reasoning_trace,
                confidence=0.2,
            )

        prompt = f"""Answer this query, noting any belief changes or evolved understanding over time.

Query: {query}

Evidence (may contain evolved/changed beliefs):
{chr(10).join(f'- {e}' for e in evidence_list)}

Provide a coherent answer that acknowledges how understanding may have evolved.
If there are contradictory memories, explain the most recent understanding."""

        answer = self.llm.generate(prompt)

        return AgentResponse(
            answer=answer,
            evidence=results[:10],
            reasoning_trace=self.reasoning_trace,
            confidence=0.6,
        )

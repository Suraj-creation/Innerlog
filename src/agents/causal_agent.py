"""Causal agent for 'why' and cause-effect queries."""

from src.agents.base_agent import BaseAgent, AgentResponse
from src.llm.prompts import CAUSAL_EXPLANATION_PROMPT
from typing import Dict, Any


class CausalAgent(BaseAgent):
    """Agent for causal reasoning queries — 'why did I...', 'what led to...'."""

    def __init__(self, llm, retriever, memory_db=None, embedding_model=None):
        super().__init__(llm, retriever, name="CausalAgent")
        self.memory_db = memory_db
        self.embedding_model = embedding_model

    def process(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        self.reasoning_trace = []
        self._log_reasoning(f"Processing causal query: {query}")

        query_embedding = None
        if self.embedding_model:
            query_embedding = self.embedding_model.encode_single(query)

        results = self._retrieve(query, query_embedding=query_embedding, k=15)
        self._log_reasoning(f"Retrieved {len(results)} memories")

        # Enrich with content
        if self.memory_db:
            for r in results:
                mem = self.memory_db.get_memory(r.memory_id)
                if mem:
                    r.content = mem.get("content", "")
                    r.metadata = mem

        # Trace causal chains
        causal_chain = []
        if self.memory_db and results:
            for r in results[:5]:
                causes = self.memory_db.get_causal_chain(r.memory_id, direction="causes")
                effects = self.memory_db.get_causal_chain(r.memory_id, direction="effects")
                for c in causes:
                    causal_chain.append(f"CAUSE: {c.get('source_id')} → {c.get('target_id')} ({c.get('link_type')})")
                for e in effects:
                    causal_chain.append(f"EFFECT: {e.get('source_id')} → {e.get('target_id')} ({e.get('link_type')})")

        self._log_reasoning(f"Found {len(causal_chain)} causal links")

        evidence = "\n".join([r.content for r in results[:7] if r.content])
        chain_text = "\n".join(causal_chain[:10]) if causal_chain else "No explicit causal links found."

        if not evidence.strip():
            return AgentResponse(
                answer="I don't have enough memories to trace the causal relationships for this question.",
                evidence=results,
                reasoning_trace=self.reasoning_trace,
                confidence=0.2,
            )

        prompt = CAUSAL_EXPLANATION_PROMPT.format(
            query=query, evidence=evidence, chain_text=chain_text
        )
        answer = self.llm.generate(prompt)

        return AgentResponse(
            answer=answer,
            evidence=results[:10],
            reasoning_trace=self.reasoning_trace,
            confidence=0.7 if causal_chain else 0.4,
        )

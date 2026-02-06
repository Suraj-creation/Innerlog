"""Build structured memory objects from raw text."""

from datetime import datetime
from typing import Optional, List
import json
import re

from src.models.memory_types import CausalMemoryObject
from src.ingestion.classifier import ImportanceScorer
from src.llm.local_llm import LocalLLM
from src.llm.prompts import MEMORY_CLASSIFICATION_PROMPT
from src.utils import get_logger

logger = get_logger("memory_builder")

# Simple keyword-based memory type classifier (fast path)
TYPE_KEYWORDS = {
    "reflective": [
        "realized", "looking back", "i notice", "changed my mind",
        "used to think", "learned that i", "in retrospect", "reflection",
        "i've grown", "my perspective", "i was wrong",
    ],
    "procedural": [
        "i always", "my routine", "my process", "step by step",
        "how i do", "my method", "when i debug", "my approach",
        "habit", "workflow",
    ],
    "semantic": [
        "the fact is", "it turns out", "i learned that",
        "the concept", "the principle", "the theory", "defined as",
        "works by", "is used for", "means that",
    ],
    # Default: episodic (events, conversations, occurrences)
}


def _classify_type_fast(text: str) -> str:
    """Fast keyword-based memory type classification."""
    text_lower = text.lower()
    for mtype, keywords in TYPE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return mtype
    return "episodic"


class MemoryEventBuilder:
    """Build structured memory objects from raw text.

    Uses lightweight classifiers for fast classification with LLM fallback.
    """

    def __init__(self, llm: LocalLLM, use_lightweight: bool = True):
        self.llm = llm
        self.use_lightweight = use_lightweight
        self.importance_scorer = ImportanceScorer()
        self._llm_fallback_count = 0
        self._lightweight_count = 0

    def build(
        self,
        raw_text: str,
        timestamp: Optional[datetime] = None,
        source: str = "text",
    ) -> CausalMemoryObject:
        """Build a memory object from raw text."""
        timestamp = timestamp or datetime.now()

        if self.use_lightweight:
            # FAST PATH (~5ms)
            memory_type = _classify_type_fast(raw_text)
            importance = self.importance_scorer.score(raw_text)
            self._lightweight_count += 1

            return CausalMemoryObject(
                timestamp=timestamp,
                content=raw_text,
                type=memory_type,
                importance=importance,
                emotion=None,  # Will be enriched later if LLM available
            )

        # SLOW PATH: Full LLM classification
        return self._build_with_llm(raw_text, timestamp)

    def _build_with_llm(self, raw_text: str, timestamp: datetime) -> CausalMemoryObject:
        """Build memory using LLM for full classification."""
        self._llm_fallback_count += 1

        if not self.llm.is_available:
            # Fallback to fast path
            return CausalMemoryObject(
                timestamp=timestamp,
                content=raw_text,
                type=_classify_type_fast(raw_text),
                importance=self.importance_scorer.score(raw_text),
            )

        prompt = MEMORY_CLASSIFICATION_PROMPT.format(text=raw_text)
        response = self.llm.generate(prompt)

        try:
            metadata = self._parse_json(response)
        except Exception:
            metadata = {}

        return CausalMemoryObject(
            timestamp=timestamp,
            content=raw_text,
            type=metadata.get("type", _classify_type_fast(raw_text)),
            topic=metadata.get("topic"),
            importance=metadata.get("importance", self.importance_scorer.score(raw_text)),
            emotion=metadata.get("emotion"),
            entities=metadata.get("entities", []),
        )

    def build_batch(
        self, texts: List[str], timestamps: Optional[List[datetime]] = None
    ) -> List[CausalMemoryObject]:
        timestamps = timestamps or [datetime.now()] * len(texts)
        return [self.build(text, ts) for text, ts in zip(texts, timestamps)]

    def get_stats(self) -> dict:
        return {
            "lightweight_calls": self._lightweight_count,
            "llm_fallback_calls": self._llm_fallback_count,
        }

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if match:
                text = match.group(1).strip()
        return json.loads(text)

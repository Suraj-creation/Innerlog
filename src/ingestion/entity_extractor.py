"""Entity extraction from memory content using LLM."""

from typing import List, Dict, Tuple
import re
import json

from src.llm.local_llm import LocalLLM
from src.llm.prompts import ENTITY_EXTRACTION_PROMPT
from src.utils import get_logger

logger = get_logger("entity_extractor")


class EntityExtractor:
    """Extract and categorize entities from memory content."""

    ENTITY_TYPES = ["person", "project", "topic", "place", "event", "artifact"]

    def __init__(self, llm: LocalLLM):
        self.llm = llm

    def extract(self, text: str) -> List[Dict]:
        """Extract entities from text using LLM with regex fallback."""
        if not self.llm.is_available:
            return self._fallback_extraction(text)

        try:
            prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
            response = self.llm.generate(prompt)
            entities = self._parse_entities(response)
            if entities:
                return entities
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}")

        return self._fallback_extraction(text)

    def _parse_entities(self, response: str) -> List[Dict]:
        """Parse entity JSON from LLM response."""
        response = response.strip()
        if "```" in response:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
            if match:
                response = match.group(1).strip()
        try:
            result = json.loads(response)
            if isinstance(result, list):
                return result
            if isinstance(result, dict) and "entities" in result:
                return result["entities"]
        except json.JSONDecodeError:
            pass
        return []

    def _fallback_extraction(self, text: str) -> List[Dict]:
        """Regex-based fallback for entity extraction."""
        entities = []
        # Proper noun detection
        proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        seen = set()
        for noun in proper_nouns:
            if noun.lower() not in seen and noun not in ("I", "The", "This", "That", "My", "We", "He", "She", "It"):
                seen.add(noun.lower())
                entities.append({"name": noun, "type": "unknown", "relations": []})
        return entities

    def build_relations(self, memory_id: str, entities: List[Dict]) -> List[Tuple[str, str, str]]:
        """Build (subject, predicate, object) triples for knowledge graph."""
        triples = []
        for entity in entities:
            name = entity.get("name", "")
            if not name:
                continue
            triples.append((memory_id, "mentions", name))
            for rel in entity.get("relations", []):
                if isinstance(rel, dict) and "type" in rel and "target" in rel:
                    triples.append((name, rel["type"], rel["target"]))
        return triples

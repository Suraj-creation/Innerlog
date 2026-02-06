"""Graph-based retrieval using the knowledge graph."""

from typing import List, Tuple, Optional
from src.storage.knowledge_graph import KnowledgeGraph


class GraphRetriever:
    """Retrieve memories through knowledge graph traversal."""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph

    def search(self, entities: List[str], k: int = 10) -> List[Tuple[str, float]]:
        """Search for memories mentioning given entities or their neighbors."""
        memory_scores: dict = {}

        for entity in entities:
            # Direct mentions
            triples = self.kg.query_pattern(obj=entity)
            for subj, pred, obj in triples:
                if pred == "mentions":
                    memory_scores[subj] = memory_scores.get(subj, 0) + 1.0

            # Also check as subject
            triples = self.kg.query_pattern(subject=entity)
            for subj, pred, obj in triples:
                if pred == "mentions":
                    memory_scores[obj] = memory_scores.get(obj, 0) + 0.8

            # One-hop neighbors
            neighbors = self.kg.get_neighbors(entity, direction="both")
            for neighbor in neighbors:
                n_entity = neighbor.get("entity", "")
                n_triples = self.kg.query_pattern(obj=n_entity)
                for subj, pred, obj in n_triples:
                    if pred == "mentions":
                        memory_scores[subj] = memory_scores.get(subj, 0) + 0.5

        # Sort and return top-k
        sorted_results = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]

"""Lightweight knowledge graph using NetworkX."""

import networkx as nx
from typing import List, Tuple, Dict, Optional, Set
import pickle
import json
from pathlib import Path

from src.utils import get_logger

logger = get_logger("knowledge_graph")


class KnowledgeGraph:
    """Lightweight knowledge graph using NetworkX."""

    def __init__(self, graph_path: Optional[str] = None):
        self.graph_path = graph_path
        if graph_path and Path(graph_path).exists():
            self.graph = self._load(graph_path)
            logger.info(f"Loaded knowledge graph from {graph_path}: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        else:
            self.graph = nx.MultiDiGraph()
            logger.info("Created new empty knowledge graph")

    def add_entity(self, name: str, entity_type: str, **attributes):
        """Add an entity node."""
        self.graph.add_node(name, type=entity_type, **attributes)

    def add_relation(self, subject: str, predicate: str, obj: str, **attributes):
        """Add a relation (edge) between entities."""
        if subject not in self.graph:
            self.graph.add_node(subject, type="unknown")
        if obj not in self.graph:
            self.graph.add_node(obj, type="unknown")
        self.graph.add_edge(subject, obj, relation=predicate, **attributes)

    def add_triple(self, triple: Tuple[str, str, str], **attributes):
        """Add a (subject, predicate, object) triple."""
        subject, predicate, obj = triple
        self.add_relation(subject, predicate, obj, **attributes)

    def add_triples(self, triples: List[Tuple[str, str, str]]):
        """Add multiple triples."""
        for triple in triples:
            if len(triple) == 3:
                self.add_triple(triple)

    def get_neighbors(
        self, entity: str, relation_type: Optional[str] = None, direction: str = "both"
    ) -> List[Dict]:
        """Get neighboring entities."""
        if entity not in self.graph:
            return []
        neighbors = []
        if direction in ["out", "both"]:
            for _, target, data in self.graph.out_edges(entity, data=True):
                rel = data.get("relation", "related_to")
                if relation_type is None or rel == relation_type:
                    neighbors.append({"entity": target, "relation": rel, "direction": "out", **data})
        if direction in ["in", "both"]:
            for source, _, data in self.graph.in_edges(entity, data=True):
                rel = data.get("relation", "related_to")
                if relation_type is None or rel == relation_type:
                    neighbors.append({"entity": source, "relation": rel, "direction": "in", **data})
        return neighbors

    def find_path(self, source: str, target: str, max_depth: int = 3) -> Optional[List[str]]:
        """Find shortest path between entities."""
        try:
            return nx.shortest_path(self.graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_subgraph(self, entities: List[str], depth: int = 1) -> Dict:
        """Extract subgraph data around given entities for visualization."""
        nodes: Set[str] = set(entities)
        for _ in range(depth):
            new_nodes: Set[str] = set()
            for n in list(nodes):
                if n in self.graph:
                    new_nodes.update(self.graph.successors(n))
                    new_nodes.update(self.graph.predecessors(n))
            nodes.update(new_nodes)

        # Limit for performance
        nodes = set(list(nodes)[:200])

        graph_nodes = []
        for n in nodes:
            if n in self.graph:
                data = dict(self.graph.nodes[n])
                graph_nodes.append({"id": n, "label": n, "type": data.get("type", "unknown")})

        graph_edges = []
        for s, t, data in self.graph.edges(data=True):
            if s in nodes and t in nodes:
                graph_edges.append({
                    "from": s, "to": t,
                    "label": data.get("relation", "related_to"),
                })

        return {"nodes": graph_nodes, "edges": graph_edges}

    def get_all_entities(self) -> List[Dict]:
        """Get all entities in the graph."""
        entities = []
        for node, data in self.graph.nodes(data=True):
            entities.append({"name": node, "type": data.get("type", "unknown"), **data})
        return entities

    def get_full_graph_data(self) -> Dict:
        """Get full graph data for visualization."""
        return self.get_subgraph(list(self.graph.nodes)[:100])

    def query_pattern(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> List[Tuple[str, str, str]]:
        """Query triples matching a pattern (None = wildcard)."""
        results = []
        for s, o, data in self.graph.edges(data=True):
            p = data.get("relation")
            if subject and s != subject:
                continue
            if predicate and p != predicate:
                continue
            if obj and o != obj:
                continue
            results.append((s, p, o))
        return results

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def save(self, path: Optional[str] = None):
        """Save graph to disk as pickle."""
        path = path or self.graph_path
        if not path:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.graph, f)
        logger.info(f"Saved knowledge graph ({self.node_count} nodes, {self.edge_count} edges) to {path}")

    def _load(self, path: str) -> nx.MultiDiGraph:
        """Load graph from disk, auto-detecting pickle vs JSON format."""
        try:
            with open(path, "rb") as f:
                header = f.read(2)
            # Pickle files start with \x80 (protocol marker)
            if header and header[0] == 0x80:
                with open(path, "rb") as f:
                    return pickle.load(f)
        except Exception:
            pass
        # Try JSON node-link format as fallback
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and ("nodes" in data or "directed" in data):
                return nx.node_link_graph(data, directed=True, multigraph=True)
        except Exception:
            pass
        # Last resort: try pickle again (for other pickle protocols)
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load knowledge graph from {path}: {e}")
            return nx.MultiDiGraph()

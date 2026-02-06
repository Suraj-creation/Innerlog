"""Multi-channel retrieval with Reciprocal Rank Fusion."""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.sql_retriever import SQLRetriever
from src.utils import get_logger

logger = get_logger("hybrid_retriever")


@dataclass
class RetrievalResult:
    """Unified retrieval result."""
    memory_id: str
    score: float
    source: str  # dense, sparse, graph, sql, or combined
    content: Optional[str] = None
    metadata: Optional[Dict] = None


class HybridRetriever:
    """Multi-channel retrieval with Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        graph_retriever: Optional[GraphRetriever] = None,
        sql_retriever: Optional[SQLRetriever] = None,
        fusion_weights: Optional[Dict[str, float]] = None,
        rrf_k: int = 60,
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.graph = graph_retriever
        self.sql = sql_retriever
        self.weights = fusion_weights or {
            "dense": 0.4, "sparse": 0.2, "graph": 0.2, "sql": 0.2
        }
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        entities: Optional[List[str]] = None,
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
        memory_types: Optional[List[str]] = None,
        topic: Optional[str] = None,
        k: int = 10,
        channels: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """Multi-channel retrieval with RRF fusion."""
        channels = channels or ["dense", "sparse", "graph", "sql"]
        all_results: Dict[str, List[Tuple[str, float]]] = {}

        # Dense retrieval
        if "dense" in channels and self.dense and query_embedding is not None:
            dense_results = self.dense.search(query_embedding, k=k * 2)
            all_results["dense"] = dense_results

        # Sparse retrieval
        if "sparse" in channels and self.sparse:
            sparse_results = self.sparse.search(query, k=k * 2)
            all_results["sparse"] = sparse_results

        # Graph retrieval
        if "graph" in channels and self.graph and entities:
            graph_results = self.graph.search(entities, k=k * 2)
            all_results["graph"] = graph_results

        # SQL retrieval
        if "sql" in channels and self.sql:
            sql_results = self.sql.search(
                query=query, time_start=time_start, time_end=time_end,
                memory_types=memory_types, topic=topic, k=k * 2,
            )
            all_results["sql"] = sql_results

        if not all_results:
            return []

        # Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(all_results, k=k)
        return fused

    def _reciprocal_rank_fusion(
        self,
        channel_results: Dict[str, List[Tuple[str, float]]],
        k: int = 10,
    ) -> List[RetrievalResult]:
        """Reciprocal Rank Fusion across channels."""
        rrf_scores: Dict[str, Dict] = {}

        for channel, results in channel_results.items():
            weight = self.weights.get(channel, 0.25)

            for rank, (memory_id, score) in enumerate(results, 1):
                rrf_score = weight * (1.0 / (self.rrf_k + rank))

                if memory_id not in rrf_scores:
                    rrf_scores[memory_id] = {
                        "score": 0.0,
                        "sources": [],
                        "original_scores": {},
                    }
                rrf_scores[memory_id]["score"] += rrf_score
                rrf_scores[memory_id]["sources"].append(channel)
                rrf_scores[memory_id]["original_scores"][channel] = score

        # Build final results
        fused_results = []
        for memory_id, data in rrf_scores.items():
            fused_results.append(RetrievalResult(
                memory_id=memory_id,
                score=data["score"],
                source="+".join(sorted(set(data["sources"]))),
            ))

        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results[:k]

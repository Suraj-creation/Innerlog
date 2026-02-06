"""FAISS-based vector storage for memory embeddings."""

import faiss
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import pickle
import threading

from src.utils import get_logger

logger = get_logger("vector_store")


class VectorStore:
    """FAISS-based vector storage for memory embeddings."""

    def __init__(self, dimension: int = 768, index_path: Optional[str] = None):
        self.dimension = dimension
        self.index_path = index_path
        self._lock = threading.Lock()

        if index_path and Path(f"{index_path}.index").exists():
            self._load_index(index_path)
            logger.info(f"Loaded existing index from {index_path} with {self.index.ntotal} vectors")
        else:
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalize)
            self.id_map: Dict[int, str] = {}
            self.reverse_map: Dict[str, int] = {}
            logger.info(f"Created new FAISS index with dimension={dimension}")

    def add(self, memory_id: str, embedding: np.ndarray) -> int:
        """Add a single embedding."""
        with self._lock:
            embedding = embedding.reshape(1, -1).astype("float32")
            faiss.normalize_L2(embedding)
            idx = self.index.ntotal
            self.index.add(embedding)
            self.id_map[idx] = memory_id
            self.reverse_map[memory_id] = idx
            return idx

    def add_batch(self, memory_ids: List[str], embeddings: np.ndarray) -> List[int]:
        """Add multiple embeddings."""
        with self._lock:
            embeddings = embeddings.astype("float32")
            faiss.normalize_L2(embeddings)
            start_idx = self.index.ntotal
            self.index.add(embeddings)
            indices = []
            for i, memory_id in enumerate(memory_ids):
                idx = start_idx + i
                self.id_map[idx] = memory_id
                self.reverse_map[memory_id] = idx
                indices.append(idx)
            return indices

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar memories. Returns list of (memory_id, score)."""
        if self.index.ntotal == 0:
            return []
        query = query_embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(query)
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx in self.id_map:
                results.append((self.id_map[idx], float(score)))
        return results

    def remove(self, memory_id: str) -> bool:
        """Mark a memory as deleted (FAISS doesn't support true deletion with IndexFlat)."""
        if memory_id in self.reverse_map:
            idx = self.reverse_map.pop(memory_id)
            del self.id_map[idx]
            return True
        return False

    @property
    def total(self) -> int:
        return self.index.ntotal

    def count(self) -> int:
        return self.index.ntotal

    def save(self, path: Optional[str] = None):
        """Save index to disk."""
        path = path or self.index_path
        if not path:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.map", "wb") as f:
            pickle.dump((self.id_map, self.reverse_map), f)
        logger.info(f"Saved FAISS index ({self.index.ntotal} vectors) to {path}")

    def _load_index(self, path: str):
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.map", "rb") as f:
            self.id_map, self.reverse_map = pickle.load(f)

"""Dense retrieval using embeddings + FAISS."""

from src.storage.vector_store import VectorStore
from src.models.embeddings import EmbeddingModel
import numpy as np
from typing import List, Tuple


class DenseRetriever:
    """Dense retrieval using embeddings."""

    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar memories by embedding."""
        return self.vector_store.search(query_embedding, k=k)

    def search_text(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search using text query."""
        embedding = self.embedding_model.encode_single(query)
        return self.search(embedding, k=k)

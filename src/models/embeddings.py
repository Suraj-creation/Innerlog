"""Embedding model wrapper — uses Ollama nomic-embed-text by default."""

import numpy as np
from typing import List, Union
import ollama as ollama_client

from src.utils import get_logger

logger = get_logger("embeddings")


class EmbeddingModel:
    """Unified embedding interface. Supports Ollama (nomic-embed-text) primarily."""

    def __init__(
        self,
        provider: str = "ollama",
        model_name: str = "nomic-embed-text",
        dimension: int = 768,
    ):
        self.provider = provider
        self.model_name = model_name
        self.dimension = dimension
        logger.info(f"EmbeddingModel initialized: provider={provider}, model={model_name}, dim={dimension}")

    def encode(self, texts: Union[str, List[str]], is_query: bool = False) -> np.ndarray:
        """Encode text(s) to embeddings. Returns shape (n, dim)."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            try:
                resp = ollama_client.embeddings(model=self.model_name, prompt=text)
                embeddings.append(resp["embedding"])
            except Exception as e:
                logger.error(f"Embedding failed for text chunk: {e}")
                embeddings.append([0.0] * self.dimension)

        arr = np.array(embeddings, dtype="float32")
        return arr

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text. Returns shape (dim,)."""
        return self.encode(text)[0]

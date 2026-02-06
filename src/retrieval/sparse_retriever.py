"""BM25-based sparse retrieval."""

from rank_bm25 import BM25Okapi
from typing import List, Tuple, Dict
import re


class SparseRetriever:
    """BM25-based sparse retrieval with lazy index rebuilding."""

    def __init__(self):
        self.bm25 = None
        self.documents: List[Dict] = []
        self.doc_ids: List[str] = []
        self._dirty = False  # Track if index needs rebuild

    def index(self, documents: List[Dict]):
        """Index documents for BM25. Each doc: {id, content}."""
        self.documents = documents
        self.doc_ids = [d["id"] for d in documents]
        self._rebuild_index()

    def add_document(self, doc_id: str, content: str):
        """Add a single document. Index is rebuilt lazily on next search."""
        self.documents.append({"id": doc_id, "content": content})
        self.doc_ids.append(doc_id)
        self._dirty = True  # Defer rebuild until search time

    def add_batch(self, doc_ids: List[str], contents: List[str]):
        """Add multiple documents efficiently."""
        for doc_id, content in zip(doc_ids, contents):
            self.documents.append({"id": doc_id, "content": content})
            self.doc_ids.append(doc_id)
        self._dirty = True

    def _rebuild_index(self):
        """Rebuild BM25 index from documents."""
        tokenized = [self._tokenize(d["content"]) for d in self.documents]
        if tokenized:
            self.bm25 = BM25Okapi(tokenized)
        self._dirty = False

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        return re.findall(r"\b\w+\b", text)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25. Rebuilds index if dirty."""
        if self._dirty:
            self._rebuild_index()
        if self.bm25 is None or not self.doc_ids:
            return []
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-k:][::-1]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.doc_ids[idx], float(scores[idx])))
        return results

    @property
    def total(self) -> int:
        return len(self.doc_ids)

    def count(self) -> int:
        return len(self.doc_ids)

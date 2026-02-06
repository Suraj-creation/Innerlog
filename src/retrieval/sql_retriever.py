"""SQL-based retrieval with exact filters (time, type, topic)."""

from typing import List, Tuple, Optional, Dict
from datetime import datetime
from src.storage.relational_db import MemoryDatabase


class SQLRetriever:
    """SQL-based retrieval for exact filters."""

    def __init__(self, memory_db: MemoryDatabase):
        self.memory_db = memory_db

    def search(
        self,
        query: Optional[str] = None,
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
        memory_types: Optional[List[str]] = None,
        topic: Optional[str] = None,
        k: int = 20,
    ) -> List[Tuple[str, float]]:
        """Search memories using SQL filters."""
        results = []

        if time_start and time_end:
            memories = self.memory_db.query_by_timerange(time_start, time_end, memory_types)
            for m in memories[:k]:
                results.append((m["event_id"], 1.0))
        elif topic:
            memories = self.memory_db.query_by_topic(topic)
            for m in memories[:k]:
                results.append((m["event_id"], 0.8))
        elif query:
            memories = self.memory_db.search_memories(query, limit=k)
            for m in memories:
                results.append((m["event_id"], 0.7))

        return results

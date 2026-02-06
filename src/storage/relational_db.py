"""DuckDB-based relational storage for memories."""

import duckdb
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from src.utils import get_logger

logger = get_logger("relational_db")


class MemoryDatabase:
    """DuckDB-based relational storage for memories."""

    def __init__(self, db_path: str = "data/memories.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(db_path)
        self._init_schema()
        logger.info(f"MemoryDatabase initialized at {db_path}")

    def _init_schema(self):
        """Initialize database schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                event_id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                content TEXT NOT NULL,
                type VARCHAR NOT NULL,
                topic VARCHAR,
                importance FLOAT DEFAULT 0.5,
                emotion VARCHAR,
                confidence FLOAT DEFAULT 0.7,
                embedding_id VARCHAR,
                supersedes VARCHAR,
                superseded_by VARCHAR,
                entities JSON,
                causes JSON,
                effects JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS causal_links (
                id INTEGER,
                source_id VARCHAR NOT NULL,
                target_id VARCHAR NOT NULL,
                link_type VARCHAR DEFAULT 'caused',
                confidence FLOAT DEFAULT 0.5,
                detection_method VARCHAR DEFAULT 'inferred',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS belief_deltas (
                id INTEGER,
                topic VARCHAR NOT NULL,
                old_belief_id VARCHAR NOT NULL,
                new_belief_id VARCHAR NOT NULL,
                change_timestamp TIMESTAMP NOT NULL,
                change_type VARCHAR NOT NULL,
                confidence FLOAT DEFAULT 0.5,
                validated BOOLEAN,
                validation_timestamp TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS consolidated_memories (
                consolidated_id VARCHAR PRIMARY KEY,
                level VARCHAR NOT NULL,
                start_date TIMESTAMP NOT NULL,
                end_date TIMESTAMP NOT NULL,
                summary TEXT NOT NULL,
                key_events JSON,
                entities JSON,
                emotional_tone VARCHAR,
                source_ids JSON,
                importance FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indices
        try:
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_timestamp ON memories(timestamp)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(type)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_topic ON memories(topic)")
        except Exception:
            pass  # Indices may already exist

    def insert_memory(self, memory: Dict[str, Any]) -> str:
        """Insert a memory object."""
        import json
        self.conn.execute("""
            INSERT INTO memories
            (event_id, timestamp, content, type, topic, importance, emotion,
             confidence, embedding_id, supersedes, superseded_by, entities, causes, effects)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            memory.get("event_id"),
            memory.get("timestamp"),
            memory.get("content"),
            memory.get("type", "episodic"),
            memory.get("topic"),
            memory.get("importance", 0.5),
            memory.get("emotion"),
            memory.get("confidence", 0.7),
            memory.get("embedding_id"),
            memory.get("supersedes"),
            memory.get("superseded_by"),
            json.dumps(memory.get("entities", [])),
            json.dumps(memory.get("causes", [])),
            json.dumps(memory.get("effects", [])),
        ])
        return memory.get("event_id", "")

    def get_memory(self, event_id: str) -> Optional[Dict]:
        """Get a single memory by ID."""
        result = self.conn.execute(
            "SELECT * FROM memories WHERE event_id = ?", [event_id]
        ).fetchone()
        if result:
            columns = [desc[0] for desc in self.conn.description]
            return dict(zip(columns, result))
        return None

    def get_all_memories(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get all memories with pagination."""
        result = self.conn.execute(
            "SELECT * FROM memories ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            [limit, offset],
        ).fetchall()
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]

    def search_memories(self, search_term: str, limit: int = 50) -> List[Dict]:
        """Full-text search on content."""
        result = self.conn.execute(
            "SELECT * FROM memories WHERE content ILIKE ? ORDER BY timestamp DESC LIMIT ?",
            [f"%{search_term}%", limit],
        ).fetchall()
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]

    def query_by_timerange(
        self,
        start: datetime,
        end: datetime,
        memory_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Query memories within a time range."""
        query = "SELECT * FROM memories WHERE timestamp BETWEEN ? AND ?"
        params: list = [start, end]
        if memory_types:
            placeholders = ",".join(["?" for _ in memory_types])
            query += f" AND type IN ({placeholders})"
            params.extend(memory_types)
        query += " ORDER BY timestamp DESC"
        result = self.conn.execute(query, params).fetchall()
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]

    def query_by_topic(self, topic: str) -> List[Dict]:
        """Query memories by topic."""
        result = self.conn.execute(
            "SELECT * FROM memories WHERE topic ILIKE ? ORDER BY timestamp DESC",
            [f"%{topic}%"],
        ).fetchall()
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]

    def query_by_type(self, memory_type: str, limit: int = 50) -> List[Dict]:
        """Query memories by type."""
        result = self.conn.execute(
            "SELECT * FROM memories WHERE type = ? ORDER BY timestamp DESC LIMIT ?",
            [memory_type, limit],
        ).fetchall()
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]

    def delete_memory(self, event_id: str) -> bool:
        """Delete a memory."""
        self.conn.execute("DELETE FROM memories WHERE event_id = ?", [event_id])
        return True

    def count_memories(self) -> int:
        """Count total memories."""
        return self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def add_causal_link(
        self, source_id: str, target_id: str,
        link_type: str = "caused", confidence: float = 0.5,
        detection_method: str = "inferred",
    ):
        """Add a causal link between memories."""
        self.conn.execute("""
            INSERT INTO causal_links (source_id, target_id, link_type, confidence, detection_method)
            VALUES (?, ?, ?, ?, ?)
        """, [source_id, target_id, link_type, confidence, detection_method])

    def get_causal_chain(self, event_id: str, direction: str = "effects") -> List[Dict]:
        """Get causal chain from a memory."""
        if direction == "effects":
            query = "SELECT * FROM causal_links WHERE source_id = ?"
        else:
            query = "SELECT * FROM causal_links WHERE target_id = ?"
        result = self.conn.execute(query, [event_id]).fetchall()
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]

    def add_belief_delta(self, delta: Dict[str, Any]):
        """Record a belief change."""
        self.conn.execute("""
            INSERT INTO belief_deltas
            (topic, old_belief_id, new_belief_id, change_timestamp, change_type, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            delta["topic"], delta["old_belief_id"], delta["new_belief_id"],
            delta["change_timestamp"], delta["change_type"], delta.get("confidence", 0.5),
        ])

    def get_belief_timeline(self, topic: Optional[str] = None) -> List[Dict]:
        """Get belief evolution timeline."""
        if topic:
            result = self.conn.execute(
                "SELECT * FROM belief_deltas WHERE topic ILIKE ? ORDER BY change_timestamp",
                [f"%{topic}%"],
            ).fetchall()
        else:
            result = self.conn.execute(
                "SELECT * FROM belief_deltas ORDER BY change_timestamp"
            ).fetchall()
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        total = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        by_type = self.conn.execute(
            "SELECT type, COUNT(*) as cnt FROM memories GROUP BY type"
        ).fetchall()
        causal_count = self.conn.execute("SELECT COUNT(*) FROM causal_links").fetchone()[0]
        belief_count = self.conn.execute("SELECT COUNT(*) FROM belief_deltas").fetchone()[0]

        return {
            "total_memories": total,
            "by_type": {row[0]: row[1] for row in by_type},
            "causal_links": causal_count,
            "belief_deltas": belief_count,
        }

    def close(self):
        """Close connection."""
        self.conn.close()

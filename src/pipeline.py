"""EdgeMemory Pipeline — Central orchestrator connecting all components."""

import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.utils import get_logger, setup_logging
from src.utils.config import load_config, get_config, get as cfg_get
from src.models.memory_types import CausalMemoryObject, MemoryQuery
from src.models.embeddings import EmbeddingModel
from src.storage.vector_store import VectorStore
from src.storage.relational_db import MemoryDatabase
from src.storage.knowledge_graph import KnowledgeGraph
from src.llm.local_llm import LocalLLM
from src.ingestion.asr import LocalASR
from src.ingestion.entity_extractor import EntityExtractor
from src.ingestion.classifier import ImportanceScorer
from src.ingestion.memory_builder import MemoryEventBuilder
from src.ingestion.entity_resolver import EntityResolver
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.sql_retriever import SQLRetriever
from src.retrieval.fusion import HybridRetriever
from src.agents.timeline_agent import TimelineAgent
from src.agents.causal_agent import CausalAgent
from src.agents.arbitration_agent import ArbitrationAgent
from src.agents.orchestrator import AgentOrchestrator

logger = get_logger("pipeline")


class EdgeMemoryPipeline:
    """Central orchestrator for the EdgeMemory system."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        setup_logging()
        logger.info("Initializing EdgeMemory pipeline...")
        self.config = load_config(config_path)
        self._initialized = False

        # --- Core models ---
        logger.info("Loading embedding model...")
        self.embedding_model = EmbeddingModel(
            model_name=cfg_get("embeddings.model", "nomic-embed-text"),
            dimension=cfg_get("embeddings.dimension", 768),
        )

        logger.info("Connecting to LLM...")
        self.llm = LocalLLM(
            model=cfg_get("llm.model", "phi3"),
            temperature=cfg_get("llm.temperature", 0.7),
            max_tokens=cfg_get("llm.max_tokens", 1024),
            provider=cfg_get("llm.provider", "ollama"),
            lmstudio_model=cfg_get("llm.lmstudio.model", "mistral-7b-instruct-v0.1"),
            lmstudio_url=cfg_get("llm.lmstudio.base_url", "http://localhost:1234"),
        )

        # --- Storage ---
        storage_dir = cfg_get("storage.base_path", "data")
        os.makedirs(storage_dir, exist_ok=True)

        logger.info("Initializing storage layer...")
        self.vector_store = VectorStore(
            dimension=cfg_get("embeddings.dimension", 768),
            index_path=os.path.join(storage_dir, "faiss_index"),
        )
        self.memory_db = MemoryDatabase(
            db_path=os.path.join(storage_dir, "memories.duckdb"),
        )
        self.knowledge_graph = KnowledgeGraph(
            graph_path=os.path.join(storage_dir, "knowledge_graph.json"),
        )

        # --- Ingestion ---
        logger.info("Initializing ingestion pipeline...")
        self.asr = LocalASR(model_size=cfg_get("asr.model_size", "base"))
        self.entity_extractor = EntityExtractor(llm=self.llm)
        self.entity_resolver = EntityResolver(
            state_path=os.path.join(storage_dir, "entity_state.json"),
        )
        self.importance_scorer = ImportanceScorer()
        self.memory_builder = MemoryEventBuilder(
            llm=self.llm,
        )

        # --- Retrieval ---
        logger.info("Initializing retrieval layer...")
        self.dense_retriever = DenseRetriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
        )
        self.sparse_retriever = SparseRetriever()
        self.graph_retriever = GraphRetriever(
            knowledge_graph=self.knowledge_graph,
        )
        self.sql_retriever = SQLRetriever(memory_db=self.memory_db)
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            graph_retriever=self.graph_retriever,
            sql_retriever=self.sql_retriever,
            fusion_weights=cfg_get("retrieval.fusion_weights", {
                "dense": 0.4, "sparse": 0.2, "graph": 0.2, "sql": 0.2,
            }),
        )

        # --- Agents ---
        logger.info("Initializing agents...")
        self.timeline_agent = TimelineAgent(
            llm=self.llm,
            retriever=self.hybrid_retriever,
            embedding_model=self.embedding_model,
            memory_db=self.memory_db,
        )
        self.causal_agent = CausalAgent(
            llm=self.llm,
            retriever=self.hybrid_retriever,
            embedding_model=self.embedding_model,
            memory_db=self.memory_db,
        )
        self.arbitration_agent = ArbitrationAgent(
            llm=self.llm,
            retriever=self.hybrid_retriever,
            embedding_model=self.embedding_model,
            memory_db=self.memory_db,
        )
        self.orchestrator = AgentOrchestrator(
            llm=self.llm,
            retriever=self.hybrid_retriever,
            timeline_agent=self.timeline_agent,
            causal_agent=self.causal_agent,
            arbitration_agent=self.arbitration_agent,
        )

        self._initialized = True
        logger.info("EdgeMemory pipeline initialized successfully.")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_text(
        self, text: str, source: str = "manual", metadata: Optional[Dict[str, Any]] = None
    ) -> CausalMemoryObject:
        """Ingest a text memory into the system."""
        start = time.time()

        # Build memory object
        memory = self.memory_builder.build(text, source=source)

        # Generate embedding
        embedding = self.embedding_model.encode_single(memory.content)

        # Store in vector store
        self.vector_store.add(memory.event_id, embedding)

        # Store in relational DB (convert Pydantic model to dict)
        mem_dict = memory.model_dump()
        mem_dict["timestamp"] = memory.timestamp.isoformat()
        self.memory_db.insert_memory(mem_dict)

        # Extract entities, resolve to canonical, and update knowledge graph
        entities = self.entity_extractor.extract(memory.content)
        resolved_entities = self.entity_resolver.resolve(entities)
        for entity in resolved_entities:
            name = entity.get("canonical_name", entity.get("name", ""))
            etype = entity.get("type", "unknown")
            if name:
                self.knowledge_graph.add_entity(name, etype)

        relations = self.entity_extractor.build_relations(memory.event_id, resolved_entities)
        for subj, rel, obj in relations:
            self.knowledge_graph.add_relation(subj, obj, rel)

        # Index in sparse retriever
        self.sparse_retriever.add_document(memory.event_id, memory.content)

        elapsed = (time.time() - start) * 1000
        logger.info(f"Ingested memory {memory.event_id} in {elapsed:.1f}ms (type={memory.type}, importance={memory.importance:.2f})")
        return memory

    def ingest_audio(self, audio_path: str, source: str = "voice") -> CausalMemoryObject:
        """Transcribe audio and ingest as memory."""
        logger.info(f"Transcribing audio: {audio_path}")
        result = self.asr.transcribe(audio_path)
        text = result.get("text", "")
        if not text.strip():
            raise ValueError("ASR produced empty transcription")
        return self.ingest_text(text, source=source, metadata={"audio_path": audio_path})

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the EdgeMemory system."""
        start = time.time()
        result = self.orchestrator.process_query(query_text, context)
        elapsed = (time.time() - start) * 1000

        return {
            "answer": result.answer,
            "evidence": [
                {"event_id": e.event_id, "content": e.content, "type": e.type,
                 "timestamp": e.timestamp, "importance": e.importance}
                for e in result.evidence
                if hasattr(e, "event_id")
            ],
            "agent_used": result.agent_used,
            "confidence": result.confidence,
            "processing_time_ms": elapsed,
            "agent_trace": result.agent_trace,
        }

    # ------------------------------------------------------------------
    # Data access helpers
    # ------------------------------------------------------------------
    def get_all_memories(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Return all memories."""
        memories = self.memory_db.get_all_memories(limit=limit, offset=offset)
        # DB already returns list of dicts
        for m in memories:
            if "timestamp" in m and not isinstance(m["timestamp"], str):
                m["timestamp"] = str(m["timestamp"])
        return memories

    def get_memory(self, event_id: str) -> Optional[Dict[str, Any]]:
        mem = self.memory_db.get_memory(event_id)
        if mem:
            if "timestamp" in mem and not isinstance(mem["timestamp"], str):
                mem["timestamp"] = str(mem["timestamp"])
            return mem
        return None

    def get_graph_data(self) -> Dict[str, Any]:
        """Return full graph data for vis.js."""
        return self.knowledge_graph.get_full_graph_data()

    def get_system_stats(self) -> Dict[str, Any]:
        db_stats = self.memory_db.get_stats()
        graph_data = self.knowledge_graph.get_full_graph_data()
        return {
            "memories": db_stats,
            "vector_count": self.vector_store.count(),
            "graph_nodes": len(graph_data.get("nodes", [])),
            "graph_edges": len(graph_data.get("edges", [])),
            "sparse_docs": self.sparse_retriever.count() if hasattr(self.sparse_retriever, "count") else 0,
            "llm_model": self.llm.model,
            "llm_provider": self.llm.active_provider,
            "llm_status": self.llm.get_status(),
            "embedding_model": self.embedding_model.model_name,
            "initialized": self._initialized,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self):
        """Persist all stores to disk."""
        logger.info("Saving all stores...")
        self.vector_store.save()
        self.knowledge_graph.save()
        self.entity_resolver.save_state()
        logger.info("Save complete.")

    def close(self):
        """Shut down the pipeline."""
        self.save()
        self.memory_db.close()
        logger.info("Pipeline closed.")

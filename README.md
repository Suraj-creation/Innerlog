<p align="center">
  <h1 align="center">🧠 InnerLog — EdgeMemory</h1>
  <p align="center">
    <strong>A laptop-buildable, fully open-source personal causal intelligence system</strong>
  </p>
  <p align="center">
    <em>Your entire life stream → structured, time-aware, causal knowledge graph — all running 100% locally on your device.</em>
  </p>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#api-reference">API</a> •
  <a href="#web-ui">Web UI</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#project-structure">Structure</a> •
  <a href="#roadmap">Roadmap</a>
</p>

---

## The Problem

Current AI assistants suffer from critical limitations:

| Limitation | Impact | InnerLog Solution |
|---|---|---|
| **No deep personal grounding** | LLMs know fragments from short chats; beliefs and goals never become stable memory | Continuous structured, persistent memory |
| **Context-window forgetting** | Past reasoning is lost when pushed out of context | Infinite temporal memory with compression |
| **Weak non-temporal memory** | No understanding of *when* or *how* thinking evolved | Time-aware causal memory graph |
| **Cloud-first, privacy-weak** | Server-side logs expose personal data | **100% on-device** — zero cloud dependency |

## The Core Insight

> **Memory ≠ Logging. Memory = Intentional Brain Duplication.**

The LLM is **not** the brain. The LLM is a **reasoning lens** over structured, evolving memory. Memory must be explicit, inspectable, and causal. Retrieval must be agentic and multi-step, not top-K similarity.

---

## Features

### 🎙️ Multi-Modal Ingestion
- **Text input** — Type thoughts, reflections, decisions directly
- **Voice input** — Local Whisper-based speech-to-text (no cloud ASR)
- **Automatic classification** — Lightweight classifiers (~50ms) categorize memory type, emotion, and importance
- **Entity extraction** — Identifies people, projects, places, and concepts with fuzzy resolution

### 🗄️ Multi-Layer Storage
- **Vector Store** (FAISS) — Semantic similarity search over memory embeddings
- **Relational DB** (DuckDB) — Structured metadata with temporal indexing
- **Knowledge Graph** (NetworkX) — Entity relationships as triples with traversal
- **Causal Links** — Explicit cause-effect chains between memories

### 🔍 Hybrid Multi-Channel Retrieval
- **Dense retrieval** — Embedding-based semantic search
- **Sparse retrieval** — BM25 keyword matching
- **Graph traversal** — Follow entity and causal relationships
- **SQL filtering** — Exact temporal/type/topic filters
- **Reciprocal Rank Fusion** — Merges all channels into one ranked result

### 🤖 Agent-Based Reasoning
- **Timeline Agent** — Reconstruct chronological sequences ("What happened last month?")
- **Causal Agent** — Trace cause-effect chains ("Why did I make that decision?")
- **Arbitration Agent** — Resolve conflicting memories and belief changes
- **Orchestrator** — Intelligent routing with multi-step query decomposition

### 🧪 Local LLM Integration
- **Ollama** — Primary provider (Phi-3, Mistral 7B, Qwen2.5)
- **LM Studio** — Secondary provider with automatic failover
- **Dual-provider failover** — If one backend is down, the other takes over seamlessly

### 🌐 Full Web Command Center
- Dashboard with system health monitoring
- Memory ingestion (text & voice)
- Memory browser with timeline view
- Chat-like query interface with reasoning traces
- Knowledge graph visualization
- Real-time WebSocket updates

### 🔒 Privacy First
- **100% offline** — No network calls, no telemetry
- **All processing local** — LLM, embeddings, ASR, everything on-device
- **Your data stays yours** — DuckDB + FAISS files on your disk

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            EDGEMEMORY SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────────────────────────────────────────┐   │
│  │   INPUT      │     │              INGESTION PIPELINE                │   │
│  │  ─────────   │     │  ┌─────────┐  ┌──────────┐  ┌──────────────┐  │   │
│  │  Voice       │────▶│  │ Whisper │─▶│ Memory   │─▶│ Embedding +  │  │   │
│  │  Text        │     │  │  (ASR)  │  │ Builder  │  │ Classify     │  │   │
│  │  Reflection  │     │  └─────────┘  └──────────┘  └──────────────┘  │   │
│  └─────────────┘     └──────────────────────┬──────────────────────────┘   │
│                                              │                              │
│                                              ▼                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        STORAGE LAYER                                 │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐   │   │
│  │  │ Vector DB  │  │ Relational │  │  Causal    │  │  Knowledge   │   │   │
│  │  │  (FAISS)   │  │    DB      │  │   Graph    │  │    Graph     │   │   │
│  │  │            │  │ (DuckDB)   │  │   Links    │  │  (Triples)   │   │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └──────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                              │                              │
│                                              ▼                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    MULTI-CHANNEL RETRIEVAL                           │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │   │
│  │  │ Dense   │  │ Sparse  │  │  Graph  │  │  SQL    │  │Re-ranker│  │   │
│  │  │(Embed)  │  │ (BM25)  │  │Traverse │  │ Filter  │  │(Fusion) │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                              │                              │
│                                              ▼                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         AGENT LAYER                                  │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │   │
│  │  │ Timeline │  │ Causal   │  │Arbitrate │  │   Orchestrator     │   │   │
│  │  │  Agent   │  │  Agent   │  │  Agent   │  │  (Router/Planner)  │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                              │                              │
│                                              ▼                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    LOCAL LLM (Reasoning Only)                        │   │
│  │               Phi-3 / Mistral 7B via Ollama + LM Studio              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

**Ingestion (Memory Creation):**
```
Input (voice/text) → Whisper ASR → Memory Builder → Classification + Embedding
                                                          ↓
                    Entity Resolution ← Storage (Vector + SQL + Graph)
                                                          ↓
                                           Store + Link + Update
```

**Query (Reasoning):**
```
Query → Intent Detection → Multi-Channel Retrieval → Fusion + Re-ranking
                                                          ↓
             Answer + Evidence ← LLM Reasoning ← Agent Orchestration
```

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Ollama** installed and running ([ollama.com](https://ollama.com))
- **8GB+ RAM** (no GPU required)

### 1. Clone & Setup

```bash
git clone https://github.com/Suraj-creation/Innerlog.git
cd Innerlog

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Pull Required Models

```bash
# LLM for reasoning
ollama pull phi3

# Embedding model
ollama pull nomic-embed-text
```

### 3. Run

```bash
python run.py
```

The server starts at **http://localhost:8000**. Open it in your browser to access the Web Command Center.

---

## Web UI

The Web Command Center is a single-page application served directly by the FastAPI backend — no npm or build steps required.

### Dashboard
System health overview showing status of all components (Ollama, FAISS, DuckDB, embeddings), memory count, and storage statistics.

### Memory Ingestion
- **Text input** — Type memories, reflections, decisions
- **Voice recording** — Record via browser microphone, transcribed locally by Whisper
- Real-time classification preview (type, emotion, importance)

### Memory Browser
- Chronological timeline of all memories
- Filter by type (episodic/semantic/procedural/reflective), topic, emotion
- Full-text search across all memories
- Memory detail view with causal links and entity connections

### Query Interface
- Chat-like interface for asking questions about your memories
- Shows reasoning trace, evidence cards, and confidence scores
- Agent routing information visible for transparency

### Knowledge Graph
- Interactive visualization of entity relationships
- Explore connections between people, projects, topics, and events

---

## API Reference

### Health & System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | System health check |
| `GET` | `/api/system/stats` | Memory count, storage stats |

### Memory Ingestion

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/memories/ingest/text` | Ingest text memory |
| `POST` | `/api/memories/ingest/audio` | Ingest audio file (Whisper transcription) |

### Memory Browser

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/memories` | List memories (paginated) |
| `GET` | `/api/memories/search?q=...` | Search memories |
| `GET` | `/api/memories/{id}` | Get memory detail |
| `DELETE` | `/api/memories/{id}` | Delete memory |

### Query

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/query` | Query the memory system with natural language |

### Knowledge Graph

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/graph/entities` | Get all entities and relations |
| `GET` | `/api/graph/neighbors/{entity}` | Get entity neighbors |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `WS /ws` | Real-time updates (memory ingested, status changes) |

Full interactive API documentation available at **http://localhost:8000/docs** (Swagger UI).

---

## Configuration

All settings are in [`configs/config.yaml`](configs/config.yaml):

```yaml
# LLM Settings
llm:
  provider: "ollama"              # "ollama" or "lmstudio"
  model: "phi3"                   # Model name
  temperature: 0.7
  max_tokens: 1024
  lmstudio:
    model: "mistral-7b-instruct-v0.1"
    base_url: "http://localhost:1234"

# Embedding Settings
embeddings:
  provider: "ollama"
  model: "nomic-embed-text"
  dimension: 768

# Retrieval Fusion Weights
retrieval:
  fusion_weights:
    dense: 0.4                    # Semantic similarity
    sparse: 0.2                   # Keyword matching
    graph: 0.2                    # Entity/causal traversal
    sql: 0.2                      # Exact filters

# Agent Settings
agents:
  enabled: [timeline, causal, arbitration]
  fallback_agent: "timeline"
```

---

## Project Structure

```
Innerlog/
├── run.py                    # Entry point — starts the server
├── requirements.txt          # Python dependencies
├── configs/
│   └── config.yaml           # System configuration
├── src/
│   ├── pipeline.py           # Central orchestrator
│   ├── api/
│   │   └── server.py         # FastAPI endpoints + WebSocket
│   ├── ingestion/
│   │   ├── asr.py            # Whisper speech-to-text
│   │   ├── memory_builder.py # Memory event construction
│   │   ├── classifier.py     # Lightweight classifiers
│   │   ├── entity_extractor.py  # Entity extraction (LLM + regex)
│   │   └── entity_resolver.py   # Fuzzy entity resolution
│   ├── storage/
│   │   ├── vector_store.py   # FAISS vector storage
│   │   ├── relational_db.py  # DuckDB relational storage
│   │   └── knowledge_graph.py # NetworkX knowledge graph
│   ├── retrieval/
│   │   ├── dense_retriever.py   # Embedding-based retrieval
│   │   ├── sparse_retriever.py  # BM25 retrieval
│   │   ├── graph_retriever.py   # Graph traversal retrieval
│   │   ├── sql_retriever.py     # SQL filter retrieval
│   │   └── fusion.py           # Reciprocal Rank Fusion
│   ├── agents/
│   │   ├── base_agent.py     # Agent interface
│   │   ├── orchestrator.py   # Multi-agent routing & chaining
│   │   ├── timeline_agent.py # Chronological queries
│   │   ├── causal_agent.py   # Cause-effect reasoning
│   │   └── arbitration_agent.py # Conflict resolution
│   ├── llm/
│   │   ├── local_llm.py      # Ollama + LM Studio dual-provider
│   │   └── prompts.py        # Prompt templates
│   ├── models/
│   │   ├── memory_types.py   # Pydantic data models
│   │   └── embeddings.py     # Embedding model wrapper
│   ├── utils/
│   │   └── config.py         # YAML config loader
│   └── web/
│       └── index.html        # Web Command Center (SPA)
├── data/                     # Runtime data (gitignored)
│   ├── memories.duckdb       # Relational database
│   ├── faiss_index.index     # Vector embeddings
│   └── knowledge_graph.json  # Entity graph
├── plan.md                   # Full implementation plan
├── model_strategy.md         # Model selection strategy
└── DL-project.md             # Project overview
```

---

## Memory Types

InnerLog classifies every memory into one of four types:

| Type | Description | Example | Decay Rate |
|------|-------------|---------|------------|
| **Episodic** | What happened (events, conversations) | "Had a meeting with John about Project X" | Medium |
| **Semantic** | What you learned (facts, insights) | "Transformers use self-attention" | Low |
| **Procedural** | How you do things (habits, methods) | "I always review code before committing" | Very Low |
| **Reflective** | Why you changed (meta-cognition) | "I realized I was avoiding hard problems" | Low |

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Ingestion Latency | < 500ms per text memory |
| Query Latency (avg) | < 2000ms |
| Query Latency (P95) | < 3500ms |
| Memory Footprint | < 4GB RAM with 10,000 memories |
| Storage | < 1GB per year of daily journaling |
| LLM Fallback Rate | < 15% (most classification by lightweight models) |
| Precision@5 | > 65% |
| Recall@10 | > 70% |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Ollama (Phi-3, Mistral 7B) + LM Studio |
| **Embeddings** | nomic-embed-text via Ollama |
| **ASR** | OpenAI Whisper (local) |
| **Vector Store** | FAISS |
| **Relational DB** | DuckDB |
| **Knowledge Graph** | NetworkX |
| **Sparse Retrieval** | BM25 (rank-bm25) |
| **Entity Resolution** | RapidFuzz |
| **Web Framework** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML/CSS/JS (no build step) |
| **Data Models** | Pydantic v2 |

---

## Roadmap

- [x] Core pipeline (ingestion → storage → retrieval → agents → LLM)
- [x] Multi-layer storage (FAISS + DuckDB + Knowledge Graph)
- [x] Hybrid retrieval with Reciprocal Rank Fusion
- [x] Agent-based reasoning (Timeline, Causal, Arbitration)
- [x] Dual LLM provider (Ollama + LM Studio failover)
- [x] Web Command Center with real-time updates
- [x] Voice ingestion via Whisper
- [ ] Memory consolidation (hierarchical summarization)
- [ ] Belief evolution tracker
- [ ] Causal link detection (explicit + inferred)
- [ ] Evaluation framework with synthetic datasets
- [ ] Fine-tuned lightweight classifiers (SetFit)
- [ ] Mobile deployment (ONNX export)
- [ ] Multi-modal memories (images, documents)

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open-source and available under the [MIT License](LICENSE).

---

<p align="center">
  <strong>Built with the philosophy: Memory ≠ Logging. Memory = Intentional Brain Duplication.</strong>
</p>

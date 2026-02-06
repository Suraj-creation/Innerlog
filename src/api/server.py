"""FastAPI application — EdgeMemory Web API + WebSocket + static UI."""

import os
import json
import asyncio
import traceback
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline import EdgeMemoryPipeline
from src.utils import get_logger

logger = get_logger("api")

# ---------------------------------------------------------------------------
# Pipeline singleton
# ---------------------------------------------------------------------------
pipeline: Optional[EdgeMemoryPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown."""
    global pipeline
    logger.info("Starting EdgeMemory API server...")
    try:
        pipeline = EdgeMemoryPipeline()
        logger.info("Pipeline ready.")
    except Exception as e:
        logger.error(f"Pipeline init failed: {e}\n{traceback.format_exc()}")
        pipeline = None
    yield
    if pipeline:
        pipeline.close()
        logger.info("Pipeline shut down.")


app = FastAPI(
    title="EdgeMemory API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "web")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class IngestTextRequest(BaseModel):
    text: str
    source: str = "manual"


class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None


class MemoryResponse(BaseModel):
    event_id: str
    content: str
    type: str
    timestamp: str
    importance: float
    topic: Optional[str] = None
    emotion: Optional[str] = None
    entities: List[str] = []


# ---------------------------------------------------------------------------
# WebSocket manager for real-time updates
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        for ws in self.active[:]:
            try:
                await ws.send_json(message)
            except Exception:
                self.active.remove(ws)


ws_manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Health & System
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {
        "status": "ok" if pipeline else "degraded",
        "timestamp": datetime.now().isoformat(),
        "pipeline_ready": pipeline is not None,
    }


@app.get("/api/system/stats")
async def system_stats():
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    return pipeline.get_system_stats()


# ---------------------------------------------------------------------------
# Memory Ingestion
# ---------------------------------------------------------------------------
@app.post("/api/memories/ingest/text")
async def ingest_text(req: IngestTextRequest):
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    try:
        mem = pipeline.ingest_text(req.text, source=req.source)
        result = {
            "event_id": mem.event_id,
            "content": mem.content,
            "type": mem.type,
            "importance": mem.importance,
            "topic": mem.topic,
            "timestamp": str(mem.timestamp),
        }
        await ws_manager.broadcast({"type": "memory_ingested", "data": result})
        return result
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/memories/ingest/audio")
async def ingest_audio(file: UploadFile = File(...)):
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    tmp_path = f"data/tmp_{file.filename}"
    os.makedirs("data", exist_ok=True)
    try:
        with open(tmp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        mem = pipeline.ingest_audio(tmp_path)
        result = {
            "event_id": mem.event_id,
            "content": mem.content,
            "type": mem.type,
            "importance": mem.importance,
            "timestamp": str(mem.timestamp),
        }
        await ws_manager.broadcast({"type": "memory_ingested", "data": result})
        return result
    except Exception as e:
        logger.error(f"Audio ingest error: {e}")
        raise HTTPException(500, str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Memory Browser
# ---------------------------------------------------------------------------
@app.get("/api/memories")
async def list_memories(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    return pipeline.get_all_memories(limit=limit, offset=offset)


@app.get("/api/memories/search")
async def search_memories(
    q: str = Query("", description="Search text"),
    type: Optional[str] = Query(None, description="Memory type filter"),
    topic: Optional[str] = Query(None, description="Topic filter"),
    limit: int = Query(50, ge=1, le=500),
):
    """Search memories by content, type, or topic."""
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    try:
        if type:
            results = pipeline.memory_db.query_by_type(type, limit=limit)
        elif topic:
            results = pipeline.memory_db.query_by_topic(topic)
        elif q:
            results = pipeline.memory_db.search_memories(q, limit=limit)
        else:
            results = pipeline.get_all_memories(limit=limit)
        # Normalize timestamps
        for m in results:
            if "timestamp" in m and not isinstance(m["timestamp"], str):
                m["timestamp"] = str(m["timestamp"])
        return results
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/memories/{event_id}")
async def get_memory(event_id: str):
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    mem = pipeline.get_memory(event_id)
    if not mem:
        raise HTTPException(404, "Memory not found")
    return mem


@app.delete("/api/memories/{event_id}")
async def delete_memory(event_id: str):
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    try:
        pipeline.memory_db.delete_memory(event_id)
        pipeline.vector_store.remove(event_id)
        await ws_manager.broadcast({"type": "memory_deleted", "data": {"event_id": event_id}})
        return {"deleted": event_id}
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Query / Chat
# ---------------------------------------------------------------------------
@app.post("/api/query")
async def query_memories(req: QueryRequest):
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    try:
        result = pipeline.query(req.query, context=req.context)
        return result
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------
@app.get("/api/graph")
async def get_graph():
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    return pipeline.get_graph_data()


@app.get("/api/graph/entities")
async def get_entities():
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    return pipeline.knowledge_graph.get_all_entities()


# ---------------------------------------------------------------------------
# Belief Timeline
# ---------------------------------------------------------------------------
@app.get("/api/beliefs")
async def get_beliefs(topic: Optional[str] = Query(None)):
    """Get belief evolution timeline, optionally filtered by topic."""
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    try:
        timeline = pipeline.memory_db.get_belief_timeline(topic=topic)
        for item in timeline:
            for key in item:
                if hasattr(item[key], 'isoformat'):
                    item[key] = item[key].isoformat()
        return {"beliefs": timeline, "count": len(timeline)}
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
@app.post("/api/system/save")
async def save_system():
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    pipeline.save()
    return {"status": "saved"}


# ---------------------------------------------------------------------------
# LLM Provider Management
# ---------------------------------------------------------------------------
@app.get("/api/llm/status")
async def llm_status():
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    return pipeline.llm.get_status()


class SwitchProviderRequest(BaseModel):
    provider: str  # "ollama" or "lmstudio"


@app.post("/api/llm/switch")
async def switch_llm_provider(req: SwitchProviderRequest):
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    ok = pipeline.llm.switch_provider(req.provider)
    if not ok:
        raise HTTPException(400, f"Cannot switch to '{req.provider}' — not available")
    return pipeline.llm.get_status()


class LLMTestRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = None


@app.post("/api/llm/test")
async def test_llm(req: LLMTestRequest):
    """Direct LLM test — sends prompt to active provider and returns raw response."""
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    try:
        import time
        start = time.time()
        response = pipeline.llm.generate(req.prompt, temperature=req.temperature)
        elapsed = (time.time() - start) * 1000
        return {
            "response": response,
            "provider": pipeline.llm.active_provider,
            "model": pipeline.llm.model,
            "latency_ms": round(elapsed, 1),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "query":
                result = pipeline.query(msg["query"]) if pipeline else {"error": "Pipeline not ready"}
                await ws.send_json({"type": "query_result", "data": result})
            elif msg.get("type") == "ingest":
                if pipeline:
                    mem = pipeline.ingest_text(msg["text"], source=msg.get("source", "ws"))
                    await ws.send_json({"type": "memory_ingested", "data": {
                        "event_id": mem.event_id, "content": mem.content,
                        "type": mem.type, "importance": mem.importance,
                    }})
            elif msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


# ---------------------------------------------------------------------------
# Serve frontend SPA
# ---------------------------------------------------------------------------
@app.get("/")
async def serve_frontend():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>EdgeMemory API</h1><p>Frontend not found. Place index.html in src/web/</p>")

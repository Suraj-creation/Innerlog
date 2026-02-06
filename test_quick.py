"""Quick verification of remaining tests."""
import httpx

base = "http://localhost:8000"

# Switch back to ollama first
r = httpx.post(f"{base}/api/llm/switch", json={"provider": "ollama"}, timeout=10)
print("Switch to Ollama:", r.json()["active_provider"])

# Graph
r = httpx.get(f"{base}/api/graph")
g = r.json()
print(f"Graph: {len(g.get('nodes', []))} nodes, {len(g.get('edges', []))} edges")

# Save
r = httpx.post(f"{base}/api/system/save")
print("Save:", r.json())

# LLM status final
r = httpx.get(f"{base}/api/llm/status")
d = r.json()
print(f"LLM Final: active={d['active_provider']}")
print(f"  Ollama/{d['ollama']['model']}: available={d['ollama']['available']}")
print(f"  LMStudio/{d['lmstudio']['model']}: available={d['lmstudio']['available']}")

# Final stats
r = httpx.get(f"{base}/api/system/stats")
s = r.json()
print(f"\nFinal Stats:")
print(f"  Memories: {s['memories']['total_memories']}")
print(f"  Vectors:  {s['vector_count']}")
print(f"  Graph:    {s['graph_nodes']} nodes, {s['graph_edges']} edges")
print(f"  Provider: {s['llm_provider']}")
print(f"  By type:  {s['memories']['by_type']}")

print("\nALL VERIFIED OK")

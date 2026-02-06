"""End-to-end API test."""
import httpx
import json
import sys

base = "http://localhost:8000"
passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        result = fn()
        print(f"  PASS  {name}")
        passed += 1
        return result
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        failed += 1
        return None

# 1. Health
def t_health():
    r = httpx.get(f"{base}/api/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    print(f"        pipeline_ready={d['pipeline_ready']}")
    return d
test("API Health", t_health)

# 2. LLM Status
def t_llm_status():
    r = httpx.get(f"{base}/api/llm/status")
    assert r.status_code == 200
    d = r.json()
    print(f"        active={d['active_provider']}, ollama={d['ollama']['available']}, lmstudio={d['lmstudio']['available']}")
    return d
test("LLM Status", t_llm_status)

# 3. System Stats
def t_stats():
    r = httpx.get(f"{base}/api/system/stats")
    assert r.status_code == 200
    d = r.json()
    print(f"        memories={d['memories']['total_memories']}, vectors={d['vector_count']}, graphs={d['graph_nodes']}")
    return d
test("System Stats", t_stats)

# 4. Ingest memories
memories_to_ingest = [
    ("Today I decided to learn about causal reasoning in AI. I read papers by Pearl and found them fascinating.", "journal"),
    ("Met Sarah at the coffee shop. She recommended the book Thinking Fast and Slow by Daniel Kahneman.", "conversation"),
    ("I used to think AI was just about pattern matching, but after reading Pearl, I now believe causality is fundamental to intelligence.", "journal"),
]
for i, (text, source) in enumerate(memories_to_ingest, 1):
    def t_ingest(t=text, s=source):
        r = httpx.post(f"{base}/api/memories/ingest/text", json={"text": t, "source": s}, timeout=60)
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        d = r.json()
        print(f"        id={d['event_id'][:8]}... type={d['type']} importance={d.get('importance', '?')}")
        return d
    test(f"Ingest Memory #{i}", t_ingest)

# 5. List memories
def t_list():
    r = httpx.get(f"{base}/api/memories?limit=10")
    assert r.status_code == 200
    mems = r.json()
    print(f"        {len(mems)} memories returned")
    assert len(mems) >= 3
    return mems
test("List Memories", t_list)

# 6. Query (uses Ollama/phi3)
def t_query():
    r = httpx.post(f"{base}/api/query", json={"query": "What do I think about AI?"}, timeout=120)
    assert r.status_code == 200
    d = r.json()
    answer = d.get("answer", "")
    print(f"        agent={d.get('agent_used')} conf={d.get('confidence')} time={d.get('processing_time_ms', 0):.0f}ms")
    print(f"        answer={answer[:150]}...")
    return d
test("Query (Ollama/phi3)", t_query)

# 7. Switch to LM Studio
def t_switch_lmstudio():
    r = httpx.post(f"{base}/api/llm/switch", json={"provider": "lmstudio"}, timeout=10)
    assert r.status_code == 200
    d = r.json()
    assert d["active_provider"] == "lmstudio"
    print(f"        switched to: {d['active_provider']}")
    return d
test("Switch to LM Studio", t_switch_lmstudio)

# 8. Query (uses LM Studio/mistral)
def t_query_mistral():
    r = httpx.post(f"{base}/api/query", json={"query": "Why did I change my mind about AI?"}, timeout=120)
    assert r.status_code == 200
    d = r.json()
    answer = d.get("answer", "")
    print(f"        agent={d.get('agent_used')} conf={d.get('confidence')} time={d.get('processing_time_ms', 0):.0f}ms")
    print(f"        answer={answer[:150]}...")
    return d
test("Query (LMStudio/mistral)", t_query_mistral)

# 9. Switch back to Ollama
def t_switch_ollama():
    r = httpx.post(f"{base}/api/llm/switch", json={"provider": "ollama"}, timeout=10)
    assert r.status_code == 200
    d = r.json()
    assert d["active_provider"] == "ollama"
    print(f"        switched to: {d['active_provider']}")
    return d
test("Switch back to Ollama", t_switch_ollama)

# 10. Knowledge Graph
def t_graph():
    r = httpx.get(f"{base}/api/graph")
    assert r.status_code == 200
    g = r.json()
    print(f"        {len(g.get('nodes', []))} nodes, {len(g.get('edges', []))} edges")
    return g
test("Knowledge Graph", t_graph)

# 11. Save
def t_save():
    r = httpx.post(f"{base}/api/system/save")
    assert r.status_code == 200
    return r.json()
test("Save System", t_save)

print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed+failed}")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)

# EdgeMemory: Model Selection Strategy

## Task-to-Model Mapping

| Task | Primary Model | Fallback | Reasoning |
|------|---------------|----------|-----------|
| **Memory Type Classification** | SetFit Classifier | Phi-3 | Classifier handles 90%, Phi-3 for edge cases |
| **Emotion Detection** | DistilBERT | Phi-3 | Fast pre-trained model sufficient |
| **Importance Scoring** | Rule-based heuristic | Phi-3 | Keywords + ML hybrid |
| **Entity Extraction (Simple)** | Phi-3 | Mistral | Names, dates, single entities |
| **Entity Extraction (Complex)** | Mistral | - | Multi-entity relationships, ambiguous references |
| **Entity Resolution/Linking** | Rule-based fuzzy match | Mistral | Canonical entity matching |
| **Query Intent Parsing** | Phi-3 | Mistral | Most queries are straightforward |
| **Complex Query Decomposition** | Mistral | - | Multi-hop, temporal reasoning |
| **Causal Link Detection** | Mistral | - | Requires deep reasoning |
| **Belief Contradiction Detection** | Mistral | - | Needs semantic understanding |
| **Timeline Summarization** | Phi-3 | - | Fast aggregation over dates |
| **Pattern Explanation** | Mistral | - | "Why" questions need depth |
| **Memory Consolidation** | Mistral | - | Summarizing weeks/months |
| **Reflection Agent** | Mistral | - | Meta-cognitive analysis |

---

## Performance Benchmarks (Expected)

### Latency (CPU, Q4 quantization)

| Operation | Phi-3 | Mistral-7B | Notes |
|-----------|-------|------------|-------|
| Simple classification | 200-500ms | 2-3s | Phi-3 10x faster |
| Entity extraction | 500ms-1s | 2-4s | Context-dependent |
| Causal reasoning | 1-2s | 3-5s | Multi-turn may be needed |
| Query understanding | 300-700ms | 2-3s | |
| Memory summarization (10 items) | 1-2s | 4-6s | |

### Memory Footprint (RAM)

| Model | Q4 | Q5 | Q8 | F16 |
|-------|-----|-----|-----|-----|
| Phi-3 Mini (3.8B) | ~2.2GB | ~2.8GB | ~4.5GB | ~7.5GB |
| Mistral-7B | ~4.1GB | ~5.2GB | ~8GB | ~14GB |

**Recommendation:** Use Q4 for both models to keep total RAM under 7GB.

---

## Hybrid Inference Pipeline

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   INGESTION PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INPUT TEXT                                                 │
│     ↓                                                       │
│  ┌──────────────────────────────────────────┐              │
│  │ 1. FAST CLASSIFICATION (0-100ms)         │              │
│  │    • SetFit Memory Type Classifier       │              │
│  │    • DistilBERT Emotion                  │              │
│  │    • Rule-based Importance               │              │
│  └──────────────────────────────────────────┘              │
│     ↓                                                       │
│  ┌──────────────────────────────────────────┐              │
│  │ 2. CONFIDENCE CHECK                      │              │
│  │    if confidence < 0.7:                  │              │
│  │       → Use Phi-3 (500ms)                │              │
│  │    else:                                 │              │
│  │       → Skip LLM                         │              │
│  └──────────────────────────────────────────┘              │
│     ↓                                                       │
│  ┌──────────────────────────────────────────┐              │
│  │ 3. ENTITY EXTRACTION                     │              │
│  │    Simple entities: Phi-3 (500ms)        │              │
│  │    Complex/ambiguous: Mistral (3s)       │              │
│  └──────────────────────────────────────────┘              │
│     ↓                                                       │
│  ┌──────────────────────────────────────────┐              │
│  │ 4. CAUSAL LINK DETECTION (ASYNC)         │              │
│  │    Background job: Mistral               │              │
│  │    (Runs nightly on memory batches)      │              │
│  └──────────────────────────────────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  USER QUERY                                                 │
│     ↓                                                       │
│  ┌──────────────────────────────────────────┐              │
│  │ 1. QUERY TYPE DETECTION (Phi-3, 300ms)   │              │
│  │    • Simple: "What did I do yesterday?"  │              │
│  │    • Complex: "What patterns led to X?"  │              │
│  └──────────────────────────────────────────┘              │
│     ↓                                      ↓                │
│  [Simple]                             [Complex]             │
│     ↓                                      ↓                │
│  Phi-3:                               Mistral:              │
│  • Timeline queries                   • Causal reasoning    │
│  • Date filtering                     • Multi-hop retrieval │
│  • Entity lookup                      • Pattern analysis    │
│  • Quick summaries                    • Belief evolution    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Critical Optimizations

### 1. Lazy Loading
```python
class DualModelManager:
    """Lazy-load models only when needed."""
    
    def __init__(self):
        self._phi3 = None
        self._mistral = None
        self.phi3_calls = 0
        self.mistral_calls = 0
    
    def get_phi3(self):
        if self._phi3 is None:
            self._phi3 = Ollama(model="phi3")
        return self._phi3
    
    def get_mistral(self):
        if self._mistral is None:
            self._mistral = Ollama(model="mistral")
        return self._mistral
    
    def choose_model(self, task_complexity: str):
        """Route to appropriate model."""
        if task_complexity in ["simple", "classification", "extraction"]:
            self.phi3_calls += 1
            return self.get_phi3()
        else:  # complex, causal, reasoning
            self.mistral_calls += 1
            return self.get_mistral()
```

### 2. Batch Processing
- **Nightly jobs**: Use Mistral for batch causal analysis (100+ memories)
- **Real-time**: Use Phi-3 for immediate feedback
- **Async processing**: Queue complex tasks for Mistral

### 3. Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_entity_extraction(text: str, model: str = "phi3"):
    """Cache frequent entity extractions."""
    pass
```

---

## When You NEED Better Models

Your current models will work for MVP and serious prototyping. Consider upgrading if:

| Limitation | Current Impact | Upgrade Option |
|------------|----------------|----------------|
| Mistral v0.1 outdated | ~10% worse reasoning than v0.3 | `ollama pull mistral:7b-instruct` (v0.3) |
| Phi-3 too small for complex NER | May miss entities in dense text | Add `qwen2.5:7b` for entity tasks |
| No specialized tools model | Can't reliably call functions | Add `mistral:7b-instruct-v0.3` (has tools) |
| Context window too small | Can't summarize long timelines | Add `qwen2.5:14b` (32k context) |

---

## Model Upgrade Path

### Phase 1: MVP (Current Setup) ✅
- Phi-3 + Mistral-7B-v0.1
- Target: 100-1000 memories
- Performance: Good enough for daily use

### Phase 2: Production (Recommended Upgrades)
```bash
# Better reasoning
ollama pull mistral:7b-instruct-v0.3

# Better entity extraction (optional)
ollama pull qwen2.5:7b

# Specialized embedding model
pip install sentence-transformers
# Use bge-small-en-v1.5 (already in requirements)
```

### Phase 3: Scale (1M+ memories)
- Add quantized 14B model for consolidation
- Use GGUF models with llama.cpp for production
- Consider model distillation for mobile

---

## Critical Bottleneck: Entity Resolution

**Problem:** Your plan mentions entity extraction but weak entity resolution will cause:
- "Bob", "Robert", "my manager" = 3 separate entities ❌
- Duplicate memories due to entity fragmentation
- Graph queries failing due to broken links

**Solution:** Add deterministic entity resolution BEFORE using LLMs:

```python
# Phase 1: Rule-based (0 LLM calls)
from rapidfuzz import fuzz

def resolve_entity(name: str, canonical_entities: Dict) -> str:
    """Fuzzy match to canonical entity."""
    matches = process.extractOne(
        name, 
        canonical_entities.keys(), 
        scorer=fuzz.ratio
    )
    if matches[1] > 85:  # 85% similarity
        return canonical_entities[matches[0]]
    return name  # New entity

# Phase 2: LLM fallback (Mistral, only for ambiguous cases)
# "John from college" vs "John from work" → needs context
```

---

## Final Recommendations

### ✅ What Will Work Great
1. **Memory classification** - Classifiers + Phi-3 fallback (90% classifier, 10% LLM)
2. **Simple queries** - Phi-3 is perfect for timeline/date queries
3. **Background causal analysis** - Mistral batches at night
4. **Entity extraction** - Phi-3 for simple, Mistral for complex

### ⚠️ What Needs Careful Design
1. **Entity resolution** - Add fuzzy matching BEFORE LLM
2. **Long-context summarization** - Mistral v0.1 has 8k context (may need chunking)
3. **Multi-hop reasoning** - May need agent chaining with Mistral

### 🚀 Quick Wins
1. Use Phi-3 by default, Mistral only when needed (save 80% inference time)
2. Run Mistral tasks async/nightly (user never waits)
3. Cache entity extractions aggressively
4. Use classifiers for 90% of classification

### 📊 Expected Performance
- **Ingestion latency**: 200-800ms per memory (mostly classifier)
- **Query latency**: 
  - Simple (Phi-3): 500ms-1s
  - Complex (Mistral): 3-5s
- **Background jobs**: Process 1000 memories in ~30 minutes (nightly)

---

## Action Items

1. **Set up model router** (choose Phi-3 vs Mistral per task)
2. **Implement entity resolution** with fuzzy matching
3. **Fine-tune classifiers** to minimize LLM calls
4. **Profile both models** on sample data to confirm latencies
5. **Consider upgrading** Mistral to v0.3 (minor improvement)

Your models are **sufficient for serious development**. The architecture's strength is using classifiers + smart routing, not relying on LLM for everything.

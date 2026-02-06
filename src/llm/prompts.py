"""Prompt templates for EdgeMemory LLM interactions."""

MEMORY_CLASSIFICATION_PROMPT = """Analyze the following text and extract structured information.

Text: {text}

Return a JSON object with:
{{
    "type": "episodic|semantic|procedural|reflective",
    "topic": "main topic/theme",
    "importance": 0.0-1.0 (how significant is this),
    "emotion": "neutral|happy|sad|anxious|excited|frustrated|angry|confused",
    "entities": ["person names", "project names", "places", "concepts"],
    "intent": "what is the purpose/intent of this memory"
}}

Only respond with valid JSON."""

ENTITY_EXTRACTION_PROMPT = """Extract all named entities from the following text.

Text: {text}

For each entity, provide:
- name: the entity name
- type: person|project|topic|place|event|artifact
- relations: any relationships mentioned

Return as JSON array:
[{{"name": "...", "type": "...", "relations": []}}]

Only respond with valid JSON array."""

INTENT_CLASSIFICATION_PROMPT = """Classify the intent of this memory query.

Query: {query}

Categories:
- timeline: Questions about when things happened, sequences of events
- causal: Questions about why things happened, causes and effects
- recall: Simple factual recall about entities, projects, or events
- reflection: Questions about how beliefs/understanding changed over time
- planning: Questions seeking recommendations based on past experience
- conflict: Queries that might involve contradictory information

Return JSON:
{{"primary_intent": "...", "secondary_intent": null, "is_complex": false, "sub_queries": [], "confidence": 0.8}}

Only respond with valid JSON."""

QUERY_DECOMPOSITION_PROMPT = """Break this complex query into simpler sub-queries that can be answered independently.

Query: {query}

Return a JSON array of sub-queries (max 3).
Only respond with valid JSON array."""

TIMELINE_NARRATIVE_PROMPT = """Based on these memories, answer the user's question about their timeline.

Query: {query}
Time focus: {time_focus}

Memories (chronological):
{evidence}

Provide a clear, chronological answer that traces the evolution of events/thoughts.
If there aren't enough memories to answer fully, say so honestly."""

CAUSAL_EXPLANATION_PROMPT = """Explain the causal relationships for this query.

Query: {query}

Related memories:
{evidence}

Causal links found:
{chain_text}

Provide a clear explanation of what caused what, and what effects followed.
Be specific about the cause-effect chain."""

SYNTHESIS_PROMPT = """Synthesize these answers into a coherent response to the original question.

Original question: {original_query}

Sub-answers:
{sub_answers}

Provide a unified, flowing answer that addresses the original question."""

STANCE_DETECTION_PROMPT = """Compare these two statements from the same person at different times.

Statement A (earlier): {content_a}
Statement B (later): {content_b}

Determine the stance of Statement B relative to Statement A:
- "agree": B supports or confirms A
- "disagree": B contradicts or opposes A
- "refines": B adds nuance or precision to A
- "expands": B builds upon A with new information
- "neutral": B is unrelated or discusses something different

Respond with JSON: {{"stance": "...", "explanation": "brief reason"}}"""

CONSOLIDATION_PROMPT = """Summarize the following memories from {date_range}.

Memories:
{memories}

Create a summary that:
1. Captures the key events and decisions
2. Notes any emotional highs/lows
3. Preserves causal relationships (X led to Y)
4. Mentions important people/projects/places

Keep the summary under {max_length} words.
Focus on what would be important to remember years later.

Return JSON:
{{
    "summary": "...",
    "key_events": ["event1", "event2"],
    "emotional_tone": "positive|negative|neutral|mixed",
    "entities": ["person1", "project1"]
}}"""

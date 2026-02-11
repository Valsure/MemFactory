# =============================================================================
# 记忆检索 Prompt 模板 - Search Prompts
# =============================================================================

"""
记忆检索相关的 Prompt 模板
"""

PROACTIVE_RECALL_PROMPT = """Analyze the following context and extract topics of historical information that may need to be recalled.

Context:
{context}

Output JSON format:
{{"topics": ["Topic 1", "Topic 2", ...], "reason": "Analysis reason"}}

Output JSON only."""


DTR_DECISION_PROMPT = """Generate a brief draft answer to the following question and evaluate your level of certainty.

Question: {query}

Output JSON format:
{{"draft_answer": "Draft answer", "confidence": 0.0-1.0, "reason": "Reason for certainty level"}}

Output JSON only."""


PSEUDO_CONTEXT_PROMPT = """Expand the following user query into a more complete and searchable description.
Do not fabricate facts, just make the query more specific and clear.

User query: {query}

Output the expanded description (one paragraph):"""

# =============================================================================
# 记忆更新 Prompt 模板 - Update Prompts
# =============================================================================

"""
记忆更新相关的 Prompt 模板
"""

MERGE_PROMPT = """Merge the following {count} related memories into one concise and complete comprehensive memory.

Memory content:
{memory_contents}

Principles:
1. Retain all important information, remove redundancy
2. Organize chronologically
3. Use latest information as reference
4. Maintain objective conciseness

Output the merged memory content directly:"""


CONFLICT_DETECTION_PROMPT = """Determine whether the following two memories have logical conflicts.

Memory A: {memory_a_value} (Time: {memory_a_time})
Memory B: {memory_b_value} (Time: {memory_b_time})

Output JSON format:
- With conflict: {{"has_conflict": true, "conflict_type": "FACTUAL/PREFERENCE/TEMPORAL", "description": "Conflict description"}}
- No conflict: {{"has_conflict": false}}

Output JSON only."""


CONFLICT_RESOLUTION_PROMPT = """Resolve the following memory conflict.

Memory A: {memory_a_value}
- Updated time: {memory_a_updated_at}
- Source credibility: {memory_a_credibility}

Memory B: {memory_b_value}
- Updated time: {memory_b_updated_at}
- Source credibility: {memory_b_credibility}

Rules:
1. User explicit statement > System inference
2. When source credibility is similar, use the latest memory
3. Suggest merging when complementary

Output JSON format:
{{"action": "keep_a/keep_b/merge/coexist", "reason": "Reason"}}

Output JSON only."""

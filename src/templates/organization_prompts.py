# =============================================================================
# 记忆组织 Prompt 模板 - Organization Prompts
# =============================================================================

"""
记忆组织相关的 Prompt 模板
"""

EVENT_EXTRACTION_PROMPT = """Extract event elements from the following memory.

Memory content: {memory_value}

Output JSON format:
{{
    "agent": "Executing entity (e.g., user, system, someone)",
    "action": "Action",
    "object": "Target object",
    "outcome": "Result",
    "context": "Context"
}}

If unable to extract a valid event, return: {{"valid": false}}

Output JSON only."""


ABSTRACTION_PROMPT = """Extract a general pattern/rule from the following related memories.

Memory content:
{memory_contents}

Output JSON format:
{{
    "label": "Pattern name",
    "condition": "Applicable conditions",
    "solution": "Recommended solution",
    "verification": "Verification method"
}}

Output JSON only."""

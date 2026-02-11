# =============================================================================
# 模板模块 - Templates Module
# =============================================================================

"""
模板模块：包含所有 Prompt 模板
"""

try:
    from .extraction_prompts import (
        EXTRACTION_PROMPT_EN,
        EXTRACTION_PROMPT_ZH,
        REACT_SYSTEM_PROMPT_EN,
        REACT_SYSTEM_PROMPT_ZH,
    )

    from .organization_prompts import (
        EVENT_EXTRACTION_PROMPT,
        ABSTRACTION_PROMPT,
    )

    from .search_prompts import (
        PROACTIVE_RECALL_PROMPT,
        DTR_DECISION_PROMPT,
        PSEUDO_CONTEXT_PROMPT,
    )

    from .update_prompts import (
        MERGE_PROMPT,
        CONFLICT_DETECTION_PROMPT,
        CONFLICT_RESOLUTION_PROMPT,
    )
except ImportError:
    from templates.extraction_prompts import (
        EXTRACTION_PROMPT_EN,
        EXTRACTION_PROMPT_ZH,
        REACT_SYSTEM_PROMPT_EN,
        REACT_SYSTEM_PROMPT_ZH,
    )

    from templates.organization_prompts import (
        EVENT_EXTRACTION_PROMPT,
        ABSTRACTION_PROMPT,
    )

    from templates.search_prompts import (
        PROACTIVE_RECALL_PROMPT,
        DTR_DECISION_PROMPT,
        PSEUDO_CONTEXT_PROMPT,
    )

    from templates.update_prompts import (
        MERGE_PROMPT,
        CONFLICT_DETECTION_PROMPT,
        CONFLICT_RESOLUTION_PROMPT,
    )

__all__ = [
    # Extraction prompts
    "EXTRACTION_PROMPT_EN",
    "EXTRACTION_PROMPT_ZH",
    "REACT_SYSTEM_PROMPT_EN",
    "REACT_SYSTEM_PROMPT_ZH",
    # Organization prompts
    "EVENT_EXTRACTION_PROMPT",
    "ABSTRACTION_PROMPT",
    # Search prompts
    "PROACTIVE_RECALL_PROMPT",
    "DTR_DECISION_PROMPT",
    "PSEUDO_CONTEXT_PROMPT",
    # Update prompts
    "MERGE_PROMPT",
    "CONFLICT_DETECTION_PROMPT",
    "CONFLICT_RESOLUTION_PROMPT",
]

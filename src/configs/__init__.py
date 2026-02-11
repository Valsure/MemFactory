# =============================================================================
# 配置模块 - Configs Module
# =============================================================================

"""
配置模块：包含所有模块的配置类
"""

try:
    from .extraction import ExtractionConfig
    from .organization import OrganizationConfig
    from .search import SearchConfig
    from .update import UpdateConfig
except ImportError:
    from configs.extraction import ExtractionConfig
    from configs.organization import OrganizationConfig
    from configs.search import SearchConfig
    from configs.update import UpdateConfig

__all__ = [
    "ExtractionConfig",
    "OrganizationConfig",
    "SearchConfig",
    "UpdateConfig",
]

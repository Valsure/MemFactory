# =============================================================================
# 记忆更新模块 - Memory Update Module
# =============================================================================

"""
记忆更新模块：实现记忆更新的核心操作
"""

try:
    from .data_structures import ConflictType, ConflictRecord, UpdateResult
    from .overwriter import MemoryOverwriter
    from .merger import MemoryMerger
    from .version_manager import MemoryVersionManager
    from .conflict_resolver import ConflictResolver
    from .forgetter import MemoryForgetter
    from .semantic_judge import SemanticRelationJudge
    from .updater import MemoryUpdater, update_memory
except ImportError:
    from update.data_structures import ConflictType, ConflictRecord, UpdateResult
    from update.overwriter import MemoryOverwriter
    from update.merger import MemoryMerger
    from update.version_manager import MemoryVersionManager
    from update.conflict_resolver import ConflictResolver
    from update.forgetter import MemoryForgetter
    from update.semantic_judge import SemanticRelationJudge
    from update.updater import MemoryUpdater, update_memory

__all__ = [
    # 数据结构
    "ConflictType",
    "ConflictRecord",
    "UpdateResult",
    # 子模块
    "MemoryOverwriter",
    "MemoryMerger",
    "MemoryVersionManager",
    "ConflictResolver",
    "MemoryForgetter",
    "SemanticRelationJudge",
    # 主类
    "MemoryUpdater",
    "update_memory",
]

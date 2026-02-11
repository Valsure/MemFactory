# =============================================================================
# 记忆组织模块 - Memory Organization Module
# =============================================================================

"""
记忆组织模块：将零散的记忆节点组织成可推理的结构化网络
"""

try:
    from .data_structures import Session, Phase, EventUnit, AbstractionNode, MemoryGraph
    from .temporal_builder import TemporalStructureBuilder
    from .event_builder import EventStructureBuilder
    from .semantic_builder import SemanticStructureBuilder
    from .hierarchy_builder import HierarchyBuilder
    from .organizer import MemoryOrganizer, organize_memories
except ImportError:
    from organization.data_structures import Session, Phase, EventUnit, AbstractionNode, MemoryGraph
    from organization.temporal_builder import TemporalStructureBuilder
    from organization.event_builder import EventStructureBuilder
    from organization.semantic_builder import SemanticStructureBuilder
    from organization.hierarchy_builder import HierarchyBuilder
    from organization.organizer import MemoryOrganizer, organize_memories

__all__ = [
    # 数据结构
    "Session",
    "Phase",
    "EventUnit",
    "AbstractionNode",
    "MemoryGraph",
    # 构建器
    "TemporalStructureBuilder",
    "EventStructureBuilder",
    "SemanticStructureBuilder",
    "HierarchyBuilder",
    # 主类
    "MemoryOrganizer",
    "organize_memories",
]

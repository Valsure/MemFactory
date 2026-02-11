# =============================================================================
# 记忆组织主类 - Memory Organizer
# =============================================================================

"""
记忆组织器：整合所有组织模块的主入口
"""

from typing import List

try:
    from ..common import MemoryItem, Edge, get_neo4j_client
    from ..configs.organization import OrganizationConfig
    from .data_structures import MemoryGraph
    from .temporal_builder import TemporalStructureBuilder
    from .event_builder import EventStructureBuilder
    from .semantic_builder import SemanticStructureBuilder
    from .hierarchy_builder import HierarchyBuilder
except ImportError:
    from common import MemoryItem, Edge, get_neo4j_client
    from configs.organization import OrganizationConfig
    from organization.data_structures import MemoryGraph
    from organization.temporal_builder import TemporalStructureBuilder
    from organization.event_builder import EventStructureBuilder
    from organization.semantic_builder import SemanticStructureBuilder
    from organization.hierarchy_builder import HierarchyBuilder


class MemoryOrganizer:
    """
    记忆组织器：整合所有组织模块的主入口
    
    使用方式：
        organizer = MemoryOrganizer(config)
        graph = organizer.run(memories)
    """
    
    def __init__(self, config: OrganizationConfig = None):
        self.config = config or OrganizationConfig()
        
        # 子模块
        self.temporal_builder = TemporalStructureBuilder(self.config)
        self.event_builder = EventStructureBuilder(self.config)
        self.semantic_builder = SemanticStructureBuilder(self.config)
        self.hierarchy_builder = HierarchyBuilder(self.config)
        
        # 数据库客户端
        self.neo4j = get_neo4j_client()
        
        if self.config.verbose:
            print(f"[MemoryOrganizer] 初始化完成")
    
    def run(self, memories: List[MemoryItem]) -> MemoryGraph:
        """
        执行完整的记忆组织流程
        
        Args:
            memories: 记忆列表
            
        Returns:
            组织好的记忆图谱
        """
        all_edges = []
        sessions = []
        phases = []
        abstractions = {}
        events = []
        
        # 1. 时间结构
        if self.config.use_temporal:
            sessions, phases, temporal_edges = self.temporal_builder.build(memories)
            all_edges.extend(temporal_edges)
        
        # 2. 事件结构
        if self.config.use_event:
            events, event_edges = self.event_builder.build(memories)
            all_edges.extend(event_edges)
        
        # 3. 语义结构
        if self.config.use_semantic:
            semantic_edges = self.semantic_builder.build(memories)
            all_edges.extend(semantic_edges)
        
        # 4. 层级抽象
        if self.config.use_hierarchy:
            abstractions, abstraction_edges = self.hierarchy_builder.build(memories, events)
            all_edges.extend(abstraction_edges)
        
        # 5. 去重边
        unique_edges = self._deduplicate_edges(all_edges)
        
        # 6. 保存边到数据库
        if self.config.auto_save_edges:
            self._save_edges(unique_edges)
        
        # 构建图谱
        graph = MemoryGraph(
            memories={m.id: m for m in memories},
            edges=unique_edges,
            sessions=sessions,
            phases=phases,
            abstractions=abstractions
        )
        
        if self.config.verbose:
            print(f"[MemoryOrganizer] 完成: 节点={graph.get_node_count()}, 边={graph.get_edge_count()}")
        
        return graph
    
    def _deduplicate_edges(self, edges: List[Edge]) -> List[Edge]:
        """去重边（保留权重最高的）"""
        edge_map = {}
        for edge in edges:
            key = (edge.source_id, edge.target_id, edge.relation_type)
            if key not in edge_map or edge.weight > edge_map[key].weight:
                edge_map[key] = edge
        return list(edge_map.values())
    
    def _save_edges(self, edges: List[Edge]):
        """保存边到数据库"""
        for edge in edges:
            self.neo4j.save_edge(edge)


def organize_memories(
    memories: List[MemoryItem],
    use_temporal: bool = True,
    use_event: bool = True,
    use_semantic: bool = True,
    use_hierarchy: bool = True,
    verbose: bool = False
) -> MemoryGraph:
    """
    便捷函数：组织记忆
    
    Args:
        memories: 记忆列表
        use_temporal: 使用时间结构
        use_event: 使用事件结构
        use_semantic: 使用语义结构
        use_hierarchy: 使用层级抽象
        verbose: 是否打印调试信息
        
    Returns:
        记忆图谱
    """
    config = OrganizationConfig(
        use_temporal=use_temporal,
        use_event=use_event,
        use_semantic=use_semantic,
        use_hierarchy=use_hierarchy,
        verbose=verbose
    )
    organizer = MemoryOrganizer(config)
    return organizer.run(memories)

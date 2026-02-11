# =============================================================================
# 记忆组织数据结构 - Organization Data Structures
# =============================================================================

"""
记忆组织相关的数据结构定义
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

try:
    from ..common import MemoryItem, Edge
except ImportError:
    from common import MemoryItem, Edge


@dataclass
class Session:
    """会话/片段：时间连续的记忆集合"""
    session_id: str
    memory_ids: List[str]
    start_time: str
    end_time: str
    topic: Optional[str] = None


@dataclass
class Phase:
    """阶段：更高层次的时间划分"""
    phase_id: str
    label: str
    session_ids: List[str]
    start_time: str
    end_time: str
    summary: Optional[str] = None


@dataclass
class EventUnit:
    """事件单元：结构化的事件表示"""
    event_id: str
    agent: Optional[str]        # 执行主体
    action: str                 # 动作
    object: Optional[str]       # 作用对象
    outcome: Optional[str]      # 结果
    context: Optional[str]      # 上下文
    timestamp: str
    source_memory_id: str
    confidence: float = 0.9


@dataclass
class AbstractionNode:
    """抽象节点：从具体案例中提炼的模式"""
    node_id: str
    label: str
    condition: str
    solution: str
    verification: str
    support_ids: List[str]
    confidence: float = 0.8


@dataclass
class MemoryGraph:
    """记忆图谱：整合所有组织结构的完整图"""
    memories: Dict[str, MemoryItem]
    edges: List[Edge]
    sessions: List[Session]
    phases: List[Phase]
    abstractions: Dict[str, AbstractionNode]
    
    def get_node_count(self) -> int:
        return len(self.memories)
    
    def get_edge_count(self) -> int:
        return len(self.edges)
    
    def get_edges_by_type(self, relation_type: str) -> List[Edge]:
        return [e for e in self.edges if e.relation_type == relation_type]
    
    def get_neighbors(self, node_id: str) -> List[str]:
        neighbors = set()
        for edge in self.edges:
            if edge.source_id == node_id:
                neighbors.add(edge.target_id)
            elif edge.target_id == node_id:
                neighbors.add(edge.source_id)
        return list(neighbors)

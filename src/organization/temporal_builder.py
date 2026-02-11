# =============================================================================
# 时间结构构建器 - Temporal Structure Builder
# =============================================================================

"""
时间结构构建器：负责将记忆按时间维度组织成会话和阶段
"""

from typing import List, Dict, Tuple
from datetime import datetime
from collections import defaultdict

try:
    from ..common import MemoryItem, Edge, RelationType, get_embedding_client, generate_id
    from ..configs.organization import OrganizationConfig
    from .data_structures import Session, Phase
except ImportError:
    from common import MemoryItem, Edge, RelationType, get_embedding_client, generate_id
    from configs.organization import OrganizationConfig
    from organization.data_structures import Session, Phase


class TemporalStructureBuilder:
    """
    时间结构构建器
    负责将记忆按时间维度组织成会话和阶段
    """
    
    def __init__(self, config: OrganizationConfig):
        self.config = config
        self.embedding = get_embedding_client()
        self.sessions: List[Session] = []
        self.phases: List[Phase] = []
        self.memory_to_session: Dict[str, str] = {}
    
    def build(self, memories: List[MemoryItem]) -> Tuple[List[Session], List[Phase], List[Edge]]:
        """
        构建时间结构
        
        Returns:
            (会话列表, 阶段列表, 时间边列表)
        """
        if not memories:
            return [], [], []
        
        # 1. 会话切分
        self.sessions = self._segment_into_sessions(memories)
        
        # 2. 阶段构建
        self.phases = self._build_phases(self.sessions)
        
        # 3. 生成时间边
        edges = self._get_temporal_edges(memories)
        
        if self.config.verbose:
            print(f"[TemporalStructureBuilder] 会话: {len(self.sessions)}, 阶段: {len(self.phases)}, 边: {len(edges)}")
        
        return self.sessions, self.phases, edges
    
    def _segment_into_sessions(self, memories: List[MemoryItem]) -> List[Session]:
        """会话切分：基于时间间隔和主题漂移"""
        # 按时间排序
        sorted_memories = sorted(memories, key=lambda m: m.created_at)
        
        sessions = []
        current_session_memories = [sorted_memories[0]]
        
        for i in range(1, len(sorted_memories)):
            prev_memory = sorted_memories[i - 1]
            curr_memory = sorted_memories[i]
            
            # 计算时间间隔
            try:
                prev_time = datetime.fromisoformat(prev_memory.created_at.replace('Z', ''))
                curr_time = datetime.fromisoformat(curr_memory.created_at.replace('Z', ''))
                time_gap = (curr_time - prev_time).total_seconds() / 60
            except:
                time_gap = 0
            
            # 判断是否切分
            time_break = time_gap > self.config.time_threshold_minutes
            
            # 主题漂移检测（使用tag重叠度）
            prev_tags = set(prev_memory.tags)
            curr_tags = set(curr_memory.tags)
            overlap = len(prev_tags & curr_tags) / max(len(prev_tags | curr_tags), 1)
            topic_break = overlap < self.config.topic_drift_threshold
            
            if time_break or topic_break:
                # 保存当前会话
                session = self._create_session(current_session_memories)
                sessions.append(session)
                current_session_memories = [curr_memory]
            else:
                current_session_memories.append(curr_memory)
        
        # 保存最后一个会话
        if current_session_memories:
            session = self._create_session(current_session_memories)
            sessions.append(session)
        
        # 建立映射
        for session in sessions:
            for mem_id in session.memory_ids:
                self.memory_to_session[mem_id] = session.session_id
        
        return sessions
    
    def _create_session(self, memories: List[MemoryItem]) -> Session:
        """创建会话对象"""
        memory_ids = [m.id for m in memories]
        start_time = min(m.created_at for m in memories)
        end_time = max(m.created_at for m in memories)
        
        # 提取主题
        all_tags = []
        for m in memories:
            all_tags.extend(m.tags)
        topic = max(set(all_tags), key=all_tags.count) if all_tags else None
        
        return Session(
            session_id=generate_id(),
            memory_ids=memory_ids,
            start_time=start_time,
            end_time=end_time,
            topic=topic
        )
    
    def _build_phases(self, sessions: List[Session]) -> List[Phase]:
        """阶段构建：将会话归并为更高层次的阶段"""
        if not sessions:
            return []
        
        # 简化：每天作为一个阶段
        day_groups = defaultdict(list)
        for session in sessions:
            try:
                day = datetime.fromisoformat(session.start_time.replace('Z', '')).date()
                day_groups[day].append(session)
            except:
                pass
        
        phases = []
        for day, day_sessions in sorted(day_groups.items()):
            phase = Phase(
                phase_id=generate_id(),
                label=f"Phase_{day.isoformat()}",
                session_ids=[s.session_id for s in day_sessions],
                start_time=min(s.start_time for s in day_sessions),
                end_time=max(s.end_time for s in day_sessions),
                summary=f"包含{len(day_sessions)}个会话"
            )
            phases.append(phase)
        
        return phases
    
    def _get_temporal_edges(self, memories: List[MemoryItem]) -> List[Edge]:
        """生成时间关系边"""
        edges = []
        sorted_memories = sorted(memories, key=lambda m: m.created_at)
        
        for i in range(len(sorted_memories) - 1):
            curr = sorted_memories[i]
            next_mem = sorted_memories[i + 1]
            
            # 检查时间间隔
            try:
                curr_time = datetime.fromisoformat(curr.created_at.replace('Z', ''))
                next_time = datetime.fromisoformat(next_mem.created_at.replace('Z', ''))
                time_gap_minutes = (next_time - curr_time).total_seconds() / 60
            except:
                time_gap_minutes = 0
            
            # 在时间阈值内建立边
            if time_gap_minutes <= self.config.time_threshold_minutes * 2:
                same_session = self.memory_to_session.get(curr.id) == self.memory_to_session.get(next_mem.id)
                weight = 0.9 if same_session else 0.6
                
                edge = Edge(
                    source_id=curr.id,
                    target_id=next_mem.id,
                    relation_type=RelationType.FOLLOWS.value,
                    weight=weight,
                    metadata={"type": "temporal", "same_session": same_session}
                )
                edges.append(edge)
        
        return edges

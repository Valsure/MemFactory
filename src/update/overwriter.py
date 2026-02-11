# =============================================================================
# 记忆覆盖器 - Memory Overwriter
# =============================================================================

"""
记忆覆盖器：用新信息完全替代旧信息

适用场景：
1. 用户明确修改偏好
2. 配置项被替换
3. 明确纠错
"""

from typing import List
import copy

try:
    from ..common import MemoryItem, MemoryStatus, current_timestamp
    from ..configs.update import UpdateConfig
except ImportError:
    from common import MemoryItem, MemoryStatus, current_timestamp
    from configs.update import UpdateConfig


class MemoryOverwriter:
    """
    记忆覆盖器：用新信息完全替代旧信息
    
    适用场景：
    1. 用户明确修改偏好
    2. 配置项被替换
    3. 明确纠错
    """
    
    def __init__(self, config: UpdateConfig):
        self.config = config
    
    def overwrite(self, old_memory: MemoryItem, new_content: str,
                  new_key: str = None, new_tags: List[str] = None,
                  reason: str = "用户主动更新") -> MemoryItem:
        """
        覆盖记忆内容
        
        Args:
            old_memory: 原记忆
            new_content: 新内容
            new_key: 新关键词（可选）
            new_tags: 新标签（可选）
            reason: 覆盖原因
            
        Returns:
            更新后的记忆节点
        """
        now = current_timestamp()
        
        updated_memory = MemoryItem(
            id=old_memory.id,  # 保留原ID
            key=new_key or old_memory.key,
            value=new_content,
            memory_type=old_memory.memory_type,
            tags=new_tags or old_memory.tags,
            confidence=old_memory.confidence,
            created_at=old_memory.created_at,
            updated_at=now,
            user_id=old_memory.user_id,
            session_id=old_memory.session_id,
            status=MemoryStatus.ACTIVATED.value,
            source_type=old_memory.source_type,
            source_credibility=old_memory.source_credibility,
            access_count=old_memory.access_count,
            decay_score=1.0,  # 重置衰减分数
            version=old_memory.version + 1,
            embedding=None  # 需要重新生成
        )
        
        if self.config.verbose:
            print(f"[MemoryOverwriter] 覆盖记忆 {old_memory.id}")
            print(f"  原内容: {old_memory.value[:50]}...")
            print(f"  新内容: {new_content[:50]}...")
        
        return updated_memory
    
    def deprecate(self, memory: MemoryItem, reason: str = "信息过时") -> MemoryItem:
        """将记忆标记为废弃"""
        deprecated_memory = copy.deepcopy(memory)
        deprecated_memory.status = MemoryStatus.DEPRECATED.value
        deprecated_memory.updated_at = current_timestamp()
        
        if self.config.verbose:
            print(f"[MemoryOverwriter] 废弃记忆 {memory.id}")
        
        return deprecated_memory

# =============================================================================
# 版本管理器 - Memory Version Manager
# =============================================================================

"""
记忆版本管理器：在不删除历史的情况下记录记忆的变化轨迹

适用场景：
1. 企业文档知识库的版本迭代
2. 需要审计追溯的场景
3. 误更新后需要回滚的场景
"""

from typing import Dict, List, Tuple
import copy

try:
    from ..common import MemoryItem, MemoryStatus, generate_id, current_timestamp
    from ..configs.update import UpdateConfig
except ImportError:
    from common import MemoryItem, MemoryStatus, generate_id, current_timestamp
    from configs.update import UpdateConfig


class MemoryVersionManager:
    """
    记忆版本管理器：在不删除历史的情况下记录记忆的变化轨迹
    
    适用场景：
    1. 企业文档知识库的版本迭代
    2. 需要审计追溯的场景
    3. 误更新后需要回滚的场景
    """
    
    def __init__(self, config: UpdateConfig):
        self.config = config
        # 版本链存储：root_id -> [version_ids]
        self.version_chains: Dict[str, List[str]] = {}
    
    def create_version(self, old_memory: MemoryItem, new_content: str,
                       change_description: str) -> Tuple[MemoryItem, MemoryItem]:
        """
        创建新版本
        
        Args:
            old_memory: 原记忆
            new_content: 新内容
            change_description: 变更描述
            
        Returns:
            (归档的旧版本, 新版本)
        """
        now = current_timestamp()
        
        # 归档旧版本
        archived_old = copy.deepcopy(old_memory)
        archived_old.status = MemoryStatus.ARCHIVED.value
        archived_old.updated_at = now
        
        # 创建新版本
        new_version = MemoryItem(
            id=generate_id(),
            key=old_memory.key,
            value=new_content,
            memory_type=old_memory.memory_type,
            tags=old_memory.tags,
            confidence=old_memory.confidence,
            created_at=now,
            updated_at=now,
            user_id=old_memory.user_id,
            session_id=old_memory.session_id,
            status=MemoryStatus.ACTIVATED.value,
            source_type=old_memory.source_type,
            source_credibility=old_memory.source_credibility,
            access_count=0,
            decay_score=1.0,
            version=old_memory.version + 1
        )
        
        # 更新版本链
        root_id = self._find_root_id(old_memory.id)
        if root_id not in self.version_chains:
            self.version_chains[root_id] = [old_memory.id]
        self.version_chains[root_id].append(new_version.id)
        
        if self.config.verbose:
            print(f"[MemoryVersionManager] 创建新版本")
            print(f"  旧版本: {old_memory.id} (v{old_memory.version}) -> 已归档")
            print(f"  新版本: {new_version.id} (v{new_version.version})")
        
        return archived_old, new_version
    
    def _find_root_id(self, memory_id: str) -> str:
        """查找版本链的根ID"""
        for root_id, chain in self.version_chains.items():
            if memory_id in chain:
                return root_id
        return memory_id
    
    def get_version_history(self, memory_id: str) -> List[str]:
        """获取版本历史"""
        root_id = self._find_root_id(memory_id)
        return self.version_chains.get(root_id, [memory_id])

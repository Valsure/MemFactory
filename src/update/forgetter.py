# =============================================================================
# 遗忘策略器 - Memory Forgetter
# =============================================================================

"""
记忆遗忘器：实现工程化的遗忘机制

遗忘动作（风险从低到高）：
1. 降权：降低检索优先级
2. 归档：从在线索引迁移到冷存储
3. 删除：彻底移除
"""

from typing import List, Dict
from datetime import datetime
import math

try:
    from ..common import MemoryItem, MemoryStatus
    from ..configs.update import UpdateConfig
except ImportError:
    from common import MemoryItem, MemoryStatus
    from configs.update import UpdateConfig


class MemoryForgetter:
    """
    记忆遗忘器：实现工程化的遗忘机制
    
    遗忘动作（风险从低到高）：
    1. 降权：降低检索优先级
    2. 归档：从在线索引迁移到冷存储
    3. 删除：彻底移除
    """
    
    def __init__(self, config: UpdateConfig):
        self.config = config
    
    def calculate_decay(self, memory: MemoryItem,
                        current_time: datetime = None) -> float:
        """
        计算记忆的衰减分数
        
        衰减公式（类似艾宾浩斯曲线）：
        decay_score = e^(-decay_rate * hours / 24) * (1 + log(1 + access_count))
        """
        if current_time is None:
            current_time = datetime.now()
        
        # 计算时间差
        try:
            last_time = datetime.fromisoformat(memory.updated_at.replace('Z', ''))
            hours_elapsed = (current_time - last_time).total_seconds() / 3600
        except:
            hours_elapsed = 0
        
        # 时间衰减
        time_decay = math.exp(-self.config.decay_rate * hours_elapsed / 24)
        
        # 访问次数加成
        access_boost = 1 + math.log(1 + memory.access_count)
        
        # 最终分数
        decay_score = min(time_decay * access_boost, 1.0)
        decay_score = max(decay_score, self.config.min_decay_score)
        
        return decay_score
    
    def update_decay_scores(self, memories: List[MemoryItem]) -> List[MemoryItem]:
        """批量更新衰减分数"""
        current_time = datetime.now()
        
        for memory in memories:
            if memory.status == MemoryStatus.ACTIVATED.value:
                memory.decay_score = self.calculate_decay(memory, current_time)
        
        return memories
    
    def auto_cleanup(self, memories: List[MemoryItem]) -> Dict[str, List[MemoryItem]]:
        """
        自动清理：根据衰减分数执行分层遗忘
        
        Returns:
            {action: [memories]}
        """
        self.update_decay_scores(memories)
        
        result = {
            "kept": [],
            "archived": [],
            "deleted": []
        }
        
        for memory in memories:
            if memory.status != MemoryStatus.ACTIVATED.value:
                continue
            
            score = memory.decay_score
            
            if score >= self.config.archive_threshold:
                result["kept"].append(memory)
            elif score >= self.config.delete_threshold:
                memory.status = MemoryStatus.ARCHIVED.value
                result["archived"].append(memory)
            else:
                memory.status = MemoryStatus.ARCHIVED.value  # 保守策略
                result["archived"].append(memory)
        
        if self.config.verbose:
            print(f"[MemoryForgetter] 清理完成: 保留={len(result['kept'])}, 归档={len(result['archived'])}")
        
        return result
    
    def reinforce(self, memory: MemoryItem) -> MemoryItem:
        """强化：当记忆被访问时，增强其保留概率"""
        memory.access_count += 1
        memory.decay_score = min(memory.decay_score * 1.2, 1.0)
        
        if self.config.verbose:
            print(f"[MemoryForgetter] 强化记忆 {memory.id}, 访问次数={memory.access_count}")
        
        return memory

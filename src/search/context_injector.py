# =============================================================================
# 上下文注入器 - Context Injector
# =============================================================================

"""
上下文注入器：将检索结果注入到推理上下文中
支持多种注入策略：前缀、后缀、分区
支持多种呈现格式：摘要、槽位、约束
"""

from typing import List, Tuple

try:
    from ..common import MemoryItem
    from ..configs.search import SearchConfig
except ImportError:
    from common import MemoryItem
    from configs.search import SearchConfig


class ContextInjector:
    """
    上下文注入器：将检索结果注入到推理上下文中
    支持多种注入策略：前缀、后缀、分区
    支持多种呈现格式：摘要、槽位、约束
    """
    
    def __init__(self, config: SearchConfig):
        self.config = config
    
    def inject(self, query: str, 
               memories: List[Tuple[MemoryItem, float]]) -> str:
        """
        将记忆注入到上下文中
        
        Args:
            query: 用户查询
            memories: 检索到的记忆列表
            
        Returns:
            注入后的上下文
        """
        if not memories:
            return query
        
        # 1. 格式化记忆
        formatted = self._format_memories(memories)
        
        # 2. 截断到最大长度
        if len(formatted) > self.config.max_inject_chars:
            formatted = formatted[:self.config.max_inject_chars] + "..."
        
        # 3. 根据策略注入
        if self.config.injection_mode == "prefix":
            return f"{formatted}\n\n用户查询：{query}"
        elif self.config.injection_mode == "suffix":
            return f"用户查询：{query}\n\n相关记忆：\n{formatted}"
        else:  # partition
            return f"=== 记忆证据 ===\n{formatted}\n\n=== 用户查询 ===\n{query}"
    
    def _format_memories(self, memories: List[Tuple[MemoryItem, float]]) -> str:
        """根据格式策略格式化记忆"""
        if self.config.injection_format == "summary":
            return self._format_as_summary(memories)
        elif self.config.injection_format == "slots":
            return self._format_as_slots(memories)
        else:  # constraints
            return self._format_as_constraints(memories)
    
    def _format_as_summary(self, memories: List[Tuple[MemoryItem, float]]) -> str:
        """格式化为摘要形式"""
        lines = []
        for mem, score in memories:
            date = mem.updated_at[:10] if mem.updated_at else "未知"
            lines.append(f"- [{mem.memory_type}|{date}] {mem.value}")
        return "\n".join(lines)
    
    def _format_as_slots(self, memories: List[Tuple[MemoryItem, float]]) -> str:
        """格式化为槽位形式"""
        slots = {}
        for mem, score in memories:
            key = mem.key
            if key not in slots or score > slots[key][1]:
                slots[key] = (mem.value, score)
        
        lines = []
        for key, (value, score) in slots.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)
    
    def _format_as_constraints(self, memories: List[Tuple[MemoryItem, float]]) -> str:
        """格式化为约束形式"""
        lines = ["约束条件："]
        for mem, score in memories:
            lines.append(f"- [constraint:{mem.memory_type}] {mem.value}")
        return "\n".join(lines)

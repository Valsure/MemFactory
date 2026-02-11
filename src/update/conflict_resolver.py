# =============================================================================
# 冲突消解器 - Conflict Resolver
# =============================================================================

"""
冲突消解器：检测并解决记忆之间的冲突

冲突类型：
1. 显式事实冲突：硬性矛盾
2. 隐式偏好冲突：行为偏离
"""

from typing import List, Dict, Optional

try:
    from ..common import MemoryItem, MemoryStatus, get_llm_client, generate_id
    from ..configs.update import UpdateConfig
    from ..templates.update_prompts import CONFLICT_DETECTION_PROMPT, CONFLICT_RESOLUTION_PROMPT
    from .data_structures import ConflictType, ConflictRecord
except ImportError:
    from common import MemoryItem, MemoryStatus, get_llm_client, generate_id
    from configs.update import UpdateConfig
    from templates.update_prompts import CONFLICT_DETECTION_PROMPT, CONFLICT_RESOLUTION_PROMPT
    from update.data_structures import ConflictType, ConflictRecord


class ConflictResolver:
    """
    冲突消解器：检测并解决记忆之间的冲突
    
    冲突类型：
    1. 显式事实冲突：硬性矛盾
    2. 隐式偏好冲突：行为偏离
    """
    
    def __init__(self, config: UpdateConfig):
        self.config = config
        self.llm = get_llm_client()
        self.conflict_records: List[ConflictRecord] = []
    
    def detect_conflict(self, new_memory: MemoryItem,
                        existing_memories: List[MemoryItem]) -> List[ConflictRecord]:
        """检测冲突"""
        conflicts = []
        
        for existing in existing_memories:
            if existing.status != MemoryStatus.ACTIVATED.value:
                continue
            
            conflict_info = self._check_conflict(new_memory, existing)
            
            if conflict_info:
                record = ConflictRecord(
                    conflict_id=generate_id(),
                    memory_id_a=new_memory.id,
                    memory_id_b=existing.id,
                    conflict_type=conflict_info["type"],
                    description=conflict_info["description"]
                )
                conflicts.append(record)
                self.conflict_records.append(record)
        
        return conflicts
    
    def _check_conflict(self, mem_a: MemoryItem, mem_b: MemoryItem) -> Optional[Dict]:
        """使用LLM检测冲突"""
        prompt = CONFLICT_DETECTION_PROMPT.format(
            memory_a_value=mem_a.value,
            memory_a_time=mem_a.created_at,
            memory_b_value=mem_b.value,
            memory_b_time=mem_b.created_at
        )
        
        response = self.llm.chat(
            system_prompt="You are a memory conflict detection expert.",
            user_prompt=prompt
        )
        
        result = self.llm.parse_json(response)
        
        if result and result.get("has_conflict"):
            type_map = {
                "FACTUAL": ConflictType.FACTUAL,
                "PREFERENCE": ConflictType.PREFERENCE,
                "TEMPORAL": ConflictType.TEMPORAL
            }
            return {
                "type": type_map.get(result.get("conflict_type", "FACTUAL"), ConflictType.FACTUAL),
                "description": result.get("description", "Conflict detected")
            }
        
        return None
    
    def resolve(self, conflict: ConflictRecord,
                memory_store: Dict[str, MemoryItem]) -> Dict:
        """
        解决冲突
        
        Returns:
            解决方案
        """
        mem_a = memory_store.get(conflict.memory_id_a)
        mem_b = memory_store.get(conflict.memory_id_b)
        
        if not mem_a or not mem_b:
            return {"error": "Memory does not exist"}
        
        # 使用LLM解决冲突
        prompt = CONFLICT_RESOLUTION_PROMPT.format(
            memory_a_value=mem_a.value,
            memory_a_updated_at=mem_a.updated_at,
            memory_a_credibility=mem_a.source_credibility,
            memory_b_value=mem_b.value,
            memory_b_updated_at=mem_b.updated_at,
            memory_b_credibility=mem_b.source_credibility
        )
        
        response = self.llm.chat(
            system_prompt="You are a memory conflict arbitration expert.",
            user_prompt=prompt
        )
        
        result = self.llm.parse_json(response)
        
        if result:
            action = result.get("action", "keep_b")
            reason = result.get("reason", "LLM decision")
            
            if action == "keep_a":
                return {"action": "keep", "keep_id": mem_a.id, "deprecate_id": mem_b.id, "reason": reason}
            elif action == "keep_b":
                return {"action": "keep", "keep_id": mem_b.id, "deprecate_id": mem_a.id, "reason": reason}
            elif action == "merge":
                return {"action": "merge", "memory_ids": [mem_a.id, mem_b.id], "reason": reason}
            else:
                return {"action": "coexist", "reason": reason}
        
        # 默认：时效性优先
        if mem_a.updated_at > mem_b.updated_at:
            return {"action": "keep", "keep_id": mem_a.id, "deprecate_id": mem_b.id, "reason": "Timeliness priority"}
        return {"action": "keep", "keep_id": mem_b.id, "deprecate_id": mem_a.id, "reason": "Timeliness priority"}

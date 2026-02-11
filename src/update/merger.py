# =============================================================================
# 记忆合并器 - Memory Merger
# =============================================================================

"""
记忆合并器：将多条相似或互补的记忆合并为一条

适用场景：
1. 用户多次表达同一主题的偏好
2. 同一事实的多个片段需要整合
3. 重复记忆的去重与压缩
"""

from typing import List, Tuple

try:
    from ..common import (
        MemoryItem, MemoryStatus,
        get_llm_client, get_embedding_client, generate_id, current_timestamp
    )
    from ..configs.update import UpdateConfig
    from ..templates.update_prompts import MERGE_PROMPT
except ImportError:
    from common import (
        MemoryItem, MemoryStatus,
        get_llm_client, get_embedding_client, generate_id, current_timestamp
    )
    from configs.update import UpdateConfig
    from templates.update_prompts import MERGE_PROMPT


class MemoryMerger:
    """
    记忆合并器：将多条相似或互补的记忆合并为一条
    
    适用场景：
    1. 用户多次表达同一主题的偏好
    2. 同一事实的多个片段需要整合
    3. 重复记忆的去重与压缩
    """
    
    def __init__(self, config: UpdateConfig):
        self.config = config
        self.llm = get_llm_client()
        self.embedding = get_embedding_client()
    
    def should_merge(self, memory_a: MemoryItem, memory_b: MemoryItem) -> Tuple[bool, float]:
        """
        判断两条记忆是否应该合并
        
        Returns:
            (是否应该合并, 相似度)
        """
        # 必须是同一用户
        if memory_a.user_id != memory_b.user_id:
            return False, 0.0
        
        # 必须是同一类型
        if memory_a.memory_type != memory_b.memory_type:
            return False, 0.0
        
        # 计算相似度
        if memory_a.embedding and memory_b.embedding:
            similarity = self.embedding.similarity(memory_a.embedding, memory_b.embedding)
        else:
            emb_a = self.embedding.embed(f"{memory_a.key} {memory_a.value}")
            emb_b = self.embedding.embed(f"{memory_b.key} {memory_b.value}")
            similarity = self.embedding.similarity(emb_a, emb_b)
        
        should = similarity >= self.config.similarity_threshold
        return should, similarity
    
    def merge_two(self, memory_a: MemoryItem, memory_b: MemoryItem) -> MemoryItem:
        """合并两条记忆"""
        return self.merge_batch([memory_a, memory_b])
    
    def merge_batch(self, memories: List[MemoryItem]) -> MemoryItem:
        """
        批量合并多条记忆
        
        Args:
            memories: 待合并的记忆列表
            
        Returns:
            合并后的记忆
        """
        if not memories:
            raise ValueError("记忆列表不能为空")
        
        if len(memories) == 1:
            return memories[0]
        
        now = current_timestamp()
        
        # 使用LLM合并内容
        merged_content = self._llm_merge(memories)
        
        # 合并标签
        all_tags = []
        for m in memories:
            all_tags.extend(m.tags)
        merged_tags = list(set(all_tags))
        
        # 选择最具代表性的key
        merged_key = max(memories, key=lambda m: len(m.key)).key
        
        # 置信度
        merged_confidence = min(
            max(m.confidence for m in memories) + self.config.merge_confidence_boost,
            1.0
        )
        
        # 可信度
        merged_credibility = max(m.source_credibility for m in memories)
        
        # 访问次数累加
        total_access = sum(m.access_count for m in memories)
        
        merged_memory = MemoryItem(
            id=generate_id(),
            key=merged_key,
            value=merged_content,
            memory_type=memories[0].memory_type,
            tags=merged_tags,
            confidence=merged_confidence,
            created_at=min(m.created_at for m in memories),
            updated_at=now,
            user_id=memories[0].user_id,
            session_id=memories[0].session_id,
            status=MemoryStatus.ACTIVATED.value,
            source_type="merged",
            source_credibility=merged_credibility,
            access_count=total_access,
            decay_score=1.0,
            version=1
        )
        
        if self.config.verbose:
            print(f"[MemoryMerger] 合并了 {len(memories)} 条记忆")
            print(f"  新记忆ID: {merged_memory.id}")
        
        return merged_memory
    
    def _llm_merge(self, memories: List[MemoryItem]) -> str:
        """使用LLM合并记忆内容"""
        memory_contents = "\n".join(
            f"- [{m.created_at[:10]}] {m.value}" 
            for m in sorted(memories, key=lambda x: x.created_at)
        )
        
        prompt = MERGE_PROMPT.format(
            count=len(memories),
            memory_contents=memory_contents
        )
        
        response = self.llm.chat(
            system_prompt="You are a memory integration expert.",
            user_prompt=prompt
        )
        
        if not response:
            # 回退：简单拼接
            contents = [m.value for m in sorted(memories, key=lambda x: x.created_at)]
            return "Comprehensive memory: " + "；".join(contents)
        
        return response.strip()

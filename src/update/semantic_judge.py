# =============================================================================
# LLM 语义相关性判定器 - Semantic Relation Judge
# =============================================================================

"""
LLM 语义相关性判定器：当 embedding 相似度不足以判断时，使用 LLM 辅助判断

解决问题：
1. embedding 模型对语义相关但表述不同的文本相似度较低
2. 需要更精准地判断两条记忆是否语义相关
"""

from typing import List, Dict, Any, Tuple

try:
    from ..common import MemoryItem, get_llm_client
    from ..configs.update import UpdateConfig
except ImportError:
    from common import MemoryItem, get_llm_client
    from configs.update import UpdateConfig


class SemanticRelationJudge:
    """
    LLM 语义相关性判定器：当 embedding 相似度不足以判断时，使用 LLM 辅助判断
    
    解决问题：
    1. embedding 模型对语义相关但表述不同的文本相似度较低
    2. 需要更精准地判断两条记忆是否语义相关
    """
    
    def __init__(self, config: UpdateConfig):
        self.config = config
        self.llm = get_llm_client()
    
    def judge_semantic_relation(self, new_memory: MemoryItem, 
                                 existing_memory: MemoryItem,
                                 embedding_similarity: float) -> Dict[str, Any]:
        """
        使用 LLM 判断两条记忆的语义关系
        
        Args:
            new_memory: 新记忆
            existing_memory: 已有记忆
            embedding_similarity: embedding 计算的相似度（供参考）
            
        Returns:
            {
                "is_related": bool,           # 是否语义相关
                "relation_type": str,         # 关系类型: "same_topic" / "update" / "conflict" / "complement" / "unrelated"
                "confidence": float,          # 置信度 0-1
                "suggested_action": str,      # 建议操作: "MERGE" / "OVERWRITE" / "VERSION" / "ADD" / "IGNORE"
                "reason": str                 # 判断理由
            }
        """
        system_prompt = """You are a memory management expert responsible for judging the semantic relationship between two memories.

You need to determine:
1. Whether the two memories discuss the same or related topics
2. Whether the new memory is an update, supplement, or completely different information from the old memory
3. Whether there are conflicts (factual contradictions, preference changes, etc.)

Relation type descriptions:
- same_topic: Discussing exactly the same topic, highly overlapping content
- update: New memory is an update to the old memory (e.g., date change, status change)
- conflict: Obvious conflict exists (factual contradiction or preference change)
- complement: New memory supplements the old memory (new information)
- unrelated: Two memories discuss different topics, no association

Suggested action descriptions:
- IGNORE: New memory content is almost identical to old memory, no need to add
- MERGE: New and old memories can be merged, information is complementary
- OVERWRITE: New memory should overwrite old memory (e.g., fact correction)
- VERSION: Create new version (e.g., preference change, need to keep history)
- ADD: Add as new memory (unrelated or weakly related to old memory)

Please output strictly in JSON format."""

        user_prompt = f"""Please judge the semantic relationship between the following two memories:

【New Memory】
Title: {new_memory.key}
Content: {new_memory.value}
Time: {new_memory.created_at}

【Existing Memory】
Title: {existing_memory.key}
Content: {existing_memory.value}
Time: {existing_memory.created_at}

【Reference Information】
Embedding similarity: {embedding_similarity:.3f} (for reference only, may not be accurate)

Please output the judgment result in JSON format:
{{"is_related": true/false, "relation_type": "same_topic/update/conflict/complement/unrelated", "confidence": 0.0-1.0, "suggested_action": "MERGE/OVERWRITE/VERSION/ADD/IGNORE", "reason": "judgment reason"}}

Output JSON only, no other content."""

        response = self.llm.chat(system_prompt, user_prompt)
        result = self.llm.parse_json(response)
        
        if result:
            # 确保所有必要字段存在
            return {
                "is_related": result.get("is_related", False),
                "relation_type": result.get("relation_type", "unrelated"),
                "confidence": result.get("confidence", 0.5),
                "suggested_action": result.get("suggested_action", "ADD"),
                "reason": result.get("reason", "LLM judgment")
            }
        
        # 解析失败，回退到基于相似度的默认判断
        if self.config.verbose:
            print(f"[SemanticRelationJudge] LLM parsing failed, using default judgment")
        
        return {
            "is_related": embedding_similarity > self.config.medium_similarity_threshold,
            "relation_type": "unrelated" if embedding_similarity < self.config.medium_similarity_threshold else "complement",
            "confidence": 0.5,
            "suggested_action": "ADD",
            "reason": f"LLM parsing failed, default judgment based on embedding similarity ({embedding_similarity:.3f})"
        }
    
    def batch_judge(self, new_memory: MemoryItem,
                    candidates: List[Tuple[MemoryItem, float]],
                    top_k: int = 3) -> List[Dict[str, Any]]:
        """
        批量判断新记忆与多个候选记忆的关系
        
        Args:
            new_memory: 新记忆
            candidates: [(记忆, embedding相似度), ...]
            top_k: 只判断相似度最高的前k个
            
        Returns:
            判断结果列表
        """
        results = []
        
        # 只处理 top_k 个
        for memory, similarity in candidates[:top_k]:
            result = self.judge_semantic_relation(new_memory, memory, similarity)
            result["memory_id"] = memory.id
            result["embedding_similarity"] = similarity
            results.append(result)
            
            if self.config.verbose:
                print(f"[SemanticRelationJudge] {memory.key[:20]}... → "
                      f"related={result['is_related']}, type={result['relation_type']}, "
                      f"action={result['suggested_action']}")
        
        return results

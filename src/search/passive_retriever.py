# =============================================================================
# 被动检索器 - Passive Retriever
# =============================================================================

"""
被动检索器：响应式的信息补全机制
当外部请求或当前推理显式暴露出信息缺口时，立即调用长期记忆
"""

from typing import List, Tuple
from datetime import datetime
import math

try:
    from ..common import (
        MemoryItem, SearchResult, MemoryStatus,
        get_embedding_client, get_neo4j_client, get_milvus_client
    )
    from ..configs.search import SearchConfig
    from .query import Query
except ImportError:
    from common import (
        MemoryItem, SearchResult, MemoryStatus,
        get_embedding_client, get_neo4j_client, get_milvus_client
    )
    from configs.search import SearchConfig
    from search.query import Query


class PassiveRetriever:
    """
    被动检索器：响应式的信息补全机制
    当外部请求或当前推理显式暴露出信息缺口时，立即调用长期记忆
    """
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.embedding = get_embedding_client()
        self.neo4j = get_neo4j_client()
        self.milvus = get_milvus_client()
    
    def search(self, query: Query) -> SearchResult:
        """
        执行被动检索
        
        Args:
            query: 检索查询
            
        Returns:
            检索结果
        """
        # 1. 生成查询向量
        query_emb = self.embedding.embed(query.text)
        
        # 2. 向量检索
        vector_results = self.milvus.search(query_emb, top_k=self.config.top_k * 2)
        
        # 3. 获取记忆详情并计算最终分数
        scored_memories = []
        now = datetime.now()
        
        for memory_id, vec_score in vector_results:
            memory = self.neo4j.get_memory(memory_id)
            if not memory:
                continue
            
            # 过滤非活跃记忆
            if memory.status != MemoryStatus.ACTIVATED.value:
                continue
            
            # 过滤用户
            if self.config.user_id and memory.user_id != self.config.user_id:
                if memory.user_id != "default_user":
                    continue
            
            # 计算最终分数
            final_score = vec_score
            
            # 时间衰减
            if self.config.use_time_decay:
                time_score = self._calculate_time_score(memory, now)
                final_score = 0.7 * vec_score + 0.3 * time_score
            
            if final_score >= self.config.similarity_threshold:
                scored_memories.append((memory, final_score))
        
        # 4. 排序并截取
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        scored_memories = scored_memories[:self.config.top_k]
        
        if self.config.verbose:
            print(f"[PassiveRetriever] 检索到 {len(scored_memories)} 条记忆")
        
        return SearchResult(
            memories=scored_memories,
            query=query.text,
            total_found=len(scored_memories)
        )
    
    def _calculate_time_score(self, memory: MemoryItem, now: datetime) -> float:
        """计算时间新鲜度分数"""
        try:
            mem_time = datetime.fromisoformat(memory.updated_at.replace('Z', ''))
            days = max((now - mem_time).days, 0)
            return math.exp(-self.config.time_decay_lambda * days)
        except:
            return 0.5

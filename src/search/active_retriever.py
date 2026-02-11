# =============================================================================
# 主动检索器 - Active Retriever
# =============================================================================

"""
主动检索器：预测式的记忆准备机制
由系统内部状态变化、长期策略或预测性判断所触发
"""

from typing import List

try:
    from ..common import (
        SearchResult, MemoryStatus,
        get_embedding_client, get_neo4j_client, get_milvus_client, get_llm_client
    )
    from ..configs.search import SearchConfig
    from ..templates.search_prompts import PROACTIVE_RECALL_PROMPT
    from .query import Query
except ImportError:
    from common import (
        SearchResult, MemoryStatus,
        get_embedding_client, get_neo4j_client, get_milvus_client, get_llm_client
    )
    from configs.search import SearchConfig
    from templates.search_prompts import PROACTIVE_RECALL_PROMPT
    from search.query import Query


class ActiveRetriever:
    """
    主动检索器：预测式的记忆准备机制
    由系统内部状态变化、长期策略或预测性判断所触发
    """
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.embedding = get_embedding_client()
        self.neo4j = get_neo4j_client()
        self.milvus = get_milvus_client()
        self.llm = get_llm_client()
    
    def search(self, query: Query, 
               related_topics: List[str] = None) -> SearchResult:
        """
        执行主动检索
        
        Args:
            query: 检索查询
            related_topics: 相关主题（用于扩展检索）
            
        Returns:
            检索结果
        """
        all_memories = []
        
        # 1. 基础检索
        query_emb = self.embedding.embed(query.text)
        base_results = self.milvus.search(query_emb, top_k=self.config.top_k)
        
        for memory_id, score in base_results:
            memory = self.neo4j.get_memory(memory_id)
            if memory and memory.status == MemoryStatus.ACTIVATED.value:
                all_memories.append((memory, score))
        
        # 2. 扩展检索：相关主题
        if related_topics:
            for topic in related_topics:
                topic_emb = self.embedding.embed(topic)
                topic_results = self.milvus.search(topic_emb, top_k=3)
                
                for memory_id, score in topic_results:
                    memory = self.neo4j.get_memory(memory_id)
                    if memory and memory.status == MemoryStatus.ACTIVATED.value:
                        # 降低扩展检索的分数
                        all_memories.append((memory, score * 0.8))
        
        # 3. 去重并排序
        seen_ids = set()
        unique_memories = []
        for mem, score in sorted(all_memories, key=lambda x: x[1], reverse=True):
            if mem.id not in seen_ids:
                seen_ids.add(mem.id)
                unique_memories.append((mem, score))
        
        unique_memories = unique_memories[:self.config.top_k]
        
        if self.config.verbose:
            print(f"[ActiveRetriever] 检索到 {len(unique_memories)} 条记忆")
        
        return SearchResult(
            memories=unique_memories,
            query=query.text,
            total_found=len(unique_memories)
        )
    
    def proactive_recall(self, context: str) -> SearchResult:
        """
        主动回忆：基于当前上下文预测可能需要的记忆
        
        Args:
            context: 当前上下文
            
        Returns:
            检索结果
        """
        # 使用LLM分析上下文，提取可能需要的记忆主题
        prompt = PROACTIVE_RECALL_PROMPT.format(context=context)
        
        response = self.llm.chat(
            system_prompt="You are a memory analysis expert.",
            user_prompt=prompt
        )
        
        result = self.llm.parse_json(response)
        if not result:
            return SearchResult(memories=[], query=context, total_found=0)
        
        topics = result.get("topics", [])
        
        if self.config.verbose:
            print(f"[ActiveRetriever] Identified topics: {topics}")
        
        # 基于主题进行检索
        query = Query(text=context)
        return self.search(query, related_topics=topics)

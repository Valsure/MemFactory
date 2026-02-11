# =============================================================================
# DTR自适应检索器 - DTR Retriever
# =============================================================================

"""
DTR（Decide-Then-Retrieve）自适应检索器
先决策、再检索：当系统判断外部信息确实能带来收益时才触发检索
"""

from typing import List, Tuple
from datetime import datetime
import math

try:
    from ..common import (
        MemoryItem, SearchResult, MemoryStatus,
        get_embedding_client, get_neo4j_client, get_milvus_client, get_llm_client
    )
    from ..configs.search import SearchConfig
    from ..templates.search_prompts import DTR_DECISION_PROMPT, PSEUDO_CONTEXT_PROMPT
    from .query import Query
except ImportError:
    from common import (
        MemoryItem, SearchResult, MemoryStatus,
        get_embedding_client, get_neo4j_client, get_milvus_client, get_llm_client
    )
    from configs.search import SearchConfig
    from templates.search_prompts import DTR_DECISION_PROMPT, PSEUDO_CONTEXT_PROMPT
    from search.query import Query


class DTRRetriever:
    """
    DTR（Decide-Then-Retrieve）自适应检索器
    先决策、再检索：当系统判断外部信息确实能带来收益时才触发检索
    """
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.embedding = get_embedding_client()
        self.neo4j = get_neo4j_client()
        self.milvus = get_milvus_client()
        self.llm = get_llm_client()
    
    def search(self, query: Query) -> SearchResult:
        """
        执行DTR检索
        
        Args:
            query: 检索查询
            
        Returns:
            检索结果
        """
        # 1. 决策阶段：判断是否需要检索
        should_retrieve, uncertainty = self._decide_retrieval(query)
        
        if self.config.verbose:
            print(f"[DTRRetriever] Uncertainty: {uncertainty:.3f}, Should retrieve: {should_retrieve}")
        
        if not should_retrieve:
            return SearchResult(
                memories=[],
                query=query.text,
                total_found=0
            )
        
        # 2. 双路径检索
        results = self._dual_path_search(query)
        
        # 3. 自适应信息选择（AIS）
        selected = self._adaptive_selection(query, results)
        
        return SearchResult(
            memories=selected,
            query=query.text,
            total_found=len(selected)
        )
    
    def _decide_retrieval(self, query: Query) -> Tuple[bool, float]:
        """
        决策是否需要检索
        基于生成不确定性判断
        
        Returns:
            (是否检索, 不确定性分数)
        """
        # 使用LLM生成草稿答案并估计不确定性
        prompt = DTR_DECISION_PROMPT.format(query=query.text)
        
        response = self.llm.chat(
            system_prompt="You are a knowledge assessment expert.",
            user_prompt=prompt
        )
        
        result = self.llm.parse_json(response)
        if not result:
            # 解析失败，默认检索
            return True, 0.5
        
        confidence = result.get("confidence", 0.5)
        uncertainty = 1.0 - confidence
        
        # 不确定性超过阈值则检索
        should_retrieve = uncertainty > self.config.uncertainty_threshold
        
        return should_retrieve, uncertainty
    
    def _dual_path_search(self, query: Query) -> List[Tuple[MemoryItem, float, str]]:
        """
        双路径检索：原始查询 + 伪上下文
        
        Returns:
            (记忆, 分数, 来源路径) 列表
        """
        results = []
        
        # 路径A：基于原始查询
        query_emb = self.embedding.embed(query.text)
        path_a_results = self.milvus.search(query_emb, top_k=self.config.top_k)
        
        for memory_id, score in path_a_results:
            memory = self.neo4j.get_memory(memory_id)
            if memory and memory.status == MemoryStatus.ACTIVATED.value:
                results.append((memory, score, "path_a"))
        
        # 路径B：生成伪上下文并检索
        pseudo_context = self._generate_pseudo_context(query)
        if pseudo_context:
            pseudo_emb = self.embedding.embed(pseudo_context)
            path_b_results = self.milvus.search(pseudo_emb, top_k=self.config.top_k)
            
            for memory_id, score in path_b_results:
                memory = self.neo4j.get_memory(memory_id)
                if memory and memory.status == MemoryStatus.ACTIVATED.value:
                    results.append((memory, score, "path_b"))
        
        return results
    
    def _generate_pseudo_context(self, query: Query) -> str:
        """生成伪上下文：补全用户意图"""
        prompt = PSEUDO_CONTEXT_PROMPT.format(query=query.text)
        
        response = self.llm.chat(
            system_prompt="You are a query expansion expert.",
            user_prompt=prompt
        )
        
        return response.strip()
    
    def _adaptive_selection(self, query: Query, 
                           results: List[Tuple[MemoryItem, float, str]]) -> List[Tuple[MemoryItem, float]]:
        """
        自适应信息选择（AIS）
        为每个候选计算综合分数
        """
        query_emb = self.embedding.embed(query.text)
        
        scored = []
        seen_ids = set()
        now = datetime.now()
        
        for memory, vec_score, path in results:
            if memory.id in seen_ids:
                continue
            seen_ids.add(memory.id)
            
            # 计算与查询的相似度
            if memory.embedding:
                s1 = self.embedding.similarity(query_emb, memory.embedding)
            else:
                mem_emb = self.embedding.embed(f"{memory.key} {memory.value}")
                s1 = self.embedding.similarity(query_emb, mem_emb)
            
            # 时间新鲜度
            time_score = 0.5
            if self.config.use_time_decay:
                try:
                    mem_time = datetime.fromisoformat(memory.updated_at.replace('Z', ''))
                    days = max((now - mem_time).days, 0)
                    time_score = math.exp(-self.config.time_decay_lambda * days)
                except:
                    pass
            
            # 综合分数
            final_score = 0.6 * s1 + 0.2 * vec_score + 0.2 * time_score
            
            if final_score >= self.config.similarity_threshold:
                scored.append((memory, final_score))
        
        # 排序并返回
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.config.top_k]

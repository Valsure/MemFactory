# =============================================================================
# 记忆检索主类 - Memory Searcher
# =============================================================================

"""
记忆检索器：整合所有检索策略的主入口
"""

from typing import Tuple

try:
    from ..common import SearchResult
    from ..configs.search import SearchConfig
    from .query import Query
    from .passive_retriever import PassiveRetriever
    from .active_retriever import ActiveRetriever
    from .dtr_retriever import DTRRetriever
    from .context_injector import ContextInjector
except ImportError:
    from common import SearchResult
    from configs.search import SearchConfig
    from search.query import Query
    from search.passive_retriever import PassiveRetriever
    from search.active_retriever import ActiveRetriever
    from search.dtr_retriever import DTRRetriever
    from search.context_injector import ContextInjector


class MemorySearcher:
    """
    记忆检索器：整合所有检索策略的主入口
    
    使用方式：
        searcher = MemorySearcher(config)
        result = searcher.run(query)
    """
    
    def __init__(self, config: SearchConfig = None):
        self.config = config or SearchConfig()
        
        # 根据策略选择检索器
        if self.config.strategy == "active":
            self._retriever = ActiveRetriever(self.config)
        elif self.config.strategy == "dtr":
            self._retriever = DTRRetriever(self.config)
        else:
            self._retriever = PassiveRetriever(self.config)
        
        # 上下文注入器
        self.injector = ContextInjector(self.config)
        
        if self.config.verbose:
            print(f"[MemorySearcher] 初始化完成，策略: {self.config.strategy}")
    
    def run(self, query: str, context: str = None) -> SearchResult:
        """
        执行记忆检索
        
        Args:
            query: 查询文本
            context: 上下文（可选）
            
        Returns:
            检索结果
        """
        q = Query(text=query, context=context)
        return self._retriever.search(q)
    
    def search_and_inject(self, query: str, context: str = None) -> Tuple[SearchResult, str]:
        """
        检索并注入上下文
        
        Args:
            query: 查询文本
            context: 上下文（可选）
            
        Returns:
            (检索结果, 注入后的上下文)
        """
        result = self.run(query, context)
        injected = self.injector.inject(query, result.memories)
        return result, injected


def search_memories(
    query: str,
    strategy: str = "passive",
    top_k: int = 10,
    verbose: bool = False,
    user_id: str = "default_user"
) -> SearchResult:
    """
    便捷函数：检索记忆
    
    Args:
        query: 查询文本
        strategy: 检索策略 ("passive", "active", "dtr")
        top_k: 返回数量
        verbose: 是否打印调试信息
        user_id: 用户ID
        
    Returns:
        检索结果
    """
    config = SearchConfig(
        strategy=strategy,
        top_k=top_k,
        verbose=verbose,
        user_id=user_id
    )
    searcher = MemorySearcher(config)
    return searcher.run(query)

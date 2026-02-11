# =============================================================================
# 记忆抽取主类 - Memory Extractor
# =============================================================================

"""
记忆抽取器：整合所有抽取策略的主入口
"""

from typing import List

try:
    from ..common import (
        MemoryItem, ConversationMessage, ExtractionResult,
        get_neo4j_client, get_milvus_client, get_embedding_client
    )
    from ..configs.extraction import ExtractionConfig
    from .simple_extractor import SimpleExtractor
    from .react_extractor import ReActExtractor
except ImportError:
    from common import (
        MemoryItem, ConversationMessage, ExtractionResult,
        get_neo4j_client, get_milvus_client, get_embedding_client
    )
    from configs.extraction import ExtractionConfig
    from extraction.simple_extractor import SimpleExtractor
    from extraction.react_extractor import ReActExtractor


class MemoryExtractor:
    """
    记忆抽取器：整合所有抽取策略的主入口
    
    使用方式：
        extractor = MemoryExtractor(config)
        result = extractor.run(messages)
    """
    
    def __init__(self, config: ExtractionConfig = None):
        self.config = config or ExtractionConfig()
        # 根据策略选择抽取器
        if self.config.strategy == "react":
            self._extractor = ReActExtractor(self.config)
        else:
            self._extractor = SimpleExtractor(self.config)
        
        # 数据库客户端
        self.neo4j = get_neo4j_client()
        self.milvus = get_milvus_client()
        self.embedding = get_embedding_client()
        
        if self.config.verbose:
            print(f"[MemoryExtractor] 初始化完成，策略: {self.config.strategy}")
    
    def run(self, messages: List[ConversationMessage]) -> ExtractionResult:
        """
        执行记忆抽取
        
        Args:
            messages: 对话消息列表
            
        Returns:
            抽取结果
        """
        # 1. 抽取记忆
        result = self._extractor.extract(messages)
        
        # 2. 自动保存
        if self.config.auto_save and result.status == "SUCCESS":
            self._save_memories(result.memory_list)
        
        return result
    
    def _save_memories(self, memories: List[MemoryItem]):
        """保存记忆到数据库"""
        for memory in memories:
            # 生成embedding
            if self.config.auto_embed:
                text = f"{memory.key} {memory.value}"
                memory.embedding = self.embedding.embed(text)
                
                # 保存到Milvus
                self.milvus.insert(memory.id, memory.embedding)
            
            # 保存到Neo4j
            self.neo4j.save_memory(memory)
            
            if self.config.verbose:
                print(f"[MemoryExtractor] 保存记忆: {memory.id} - {memory.key}")


def extract_memories(
    messages: List[ConversationMessage],
    strategy: str = "simple",
    auto_save: bool = True,
    verbose: bool = False,
    user_id: str = "default_user"
) -> ExtractionResult:
    """
    便捷函数：从对话中抽取记忆
    
    Args:
        messages: 对话消息列表
        strategy: 抽取策略 ("simple" 或 "react")
        auto_save: 是否自动保存
        verbose: 是否打印调试信息
        user_id: 用户ID
        
    Returns:
        抽取结果
    """
    config = ExtractionConfig(
        strategy=strategy,
        auto_save=auto_save,
        verbose=verbose,
        user_id=user_id
    )
    extractor = MemoryExtractor(config)
    return extractor.run(messages)

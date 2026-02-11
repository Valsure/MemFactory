# =============================================================================
# Memory Engineering - 记忆工程模块
# =============================================================================

"""
Memory Engineering 核心模块

包含以下子模块：
- common: 公共配置和工具（LLM、Embedding、Neo4j、Milvus客户端）
- configs: 各模块配置类
- templates: Prompt模板
- extraction: 记忆抽取
- search: 记忆检索
- organization: 记忆组织
- update: 记忆更新
"""

# =============================================================================
# 公共模块
# =============================================================================
from .common import (
    # 数据结构
    MemoryItem,
    ConversationMessage,
    ExtractionResult,
    SearchResult,
    Edge,
    
    # 枚举类型
    MemoryType,
    MemoryStatus,
    UpdateAction,
    RelationType,
    
    # 客户端
    LLMClient,
    EmbeddingClient,
    Neo4jClient,
    MilvusClient,
    MemoryStore,
    
    # 单例获取函数
    get_llm_client,
    get_embedding_client,
    get_neo4j_client,
    get_milvus_client,
    get_memory_store,
    
    # 工具函数
    generate_id,
    current_timestamp,
    format_conversation,
)

# =============================================================================
# 配置模块 - 从新的 configs 目录导入
# =============================================================================
from .configs import (
    ExtractionConfig,
    OrganizationConfig,
    SearchConfig,
    UpdateConfig,
)

# =============================================================================
# 记忆抽取模块 - 从新的 extraction 目录导入
# =============================================================================
from .extraction import (
    MemoryExtractor,
    SimpleExtractor,
    ReActExtractor,
    MemoryBuffer,
    extract_memories,
    detect_lang,
)

# =============================================================================
# 记忆检索模块 - 从新的 search 目录导入
# =============================================================================
from .search import (
    Query,
    MemorySearcher,
    PassiveRetriever,
    ActiveRetriever,
    DTRRetriever,
    ContextInjector,
    search_memories,
)

# =============================================================================
# 记忆组织模块 - 从新的 organization 目录导入
# =============================================================================
from .organization import (
    Session,
    Phase,
    EventUnit,
    AbstractionNode,
    MemoryGraph,
    MemoryOrganizer,
    TemporalStructureBuilder,
    EventStructureBuilder,
    SemanticStructureBuilder,
    HierarchyBuilder,
    organize_memories,
)

# =============================================================================
# 记忆更新模块 - 从新的 update 目录导入
# =============================================================================
from .update import (
    ConflictType,
    ConflictRecord,
    UpdateResult,
    MemoryUpdater,
    MemoryOverwriter,
    MemoryMerger,
    MemoryVersionManager,
    ConflictResolver,
    MemoryForgetter,
    update_memory,
)


__version__ = "0.1.0"
__author__ = "Memory Engineering Team"

__all__ = [
    # 数据结构
    "MemoryItem",
    "ConversationMessage",
    "ExtractionResult",
    "SearchResult",
    "Edge",
    "Session",
    "Phase",
    "EventUnit",
    "AbstractionNode",
    "MemoryGraph",
    "Query",
    "ConflictRecord",
    "UpdateResult",
    
    # 枚举类型
    "MemoryType",
    "MemoryStatus",
    "UpdateAction",
    "RelationType",
    "ConflictType",
    
    # 配置类
    "ExtractionConfig",
    "SearchConfig",
    "OrganizationConfig",
    "UpdateConfig",
    
    # 主要类
    "MemoryExtractor",
    "MemorySearcher",
    "MemoryOrganizer",
    "MemoryUpdater",
    
    # 抽取器
    "SimpleExtractor",
    "ReActExtractor",
    "MemoryBuffer",
    
    # 检索器
    "PassiveRetriever",
    "ActiveRetriever",
    "DTRRetriever",
    "ContextInjector",
    
    # 组织器
    "TemporalStructureBuilder",
    "EventStructureBuilder",
    "SemanticStructureBuilder",
    "HierarchyBuilder",
    
    # 更新器
    "MemoryOverwriter",
    "MemoryMerger",
    "MemoryVersionManager",
    "ConflictResolver",
    "MemoryForgetter",
    
    # 客户端
    "LLMClient",
    "EmbeddingClient",
    "Neo4jClient",
    "MilvusClient",
    "MemoryStore",
    "get_llm_client",
    "get_embedding_client",
    "get_neo4j_client",
    "get_milvus_client",
    "get_memory_store",
    
    # 便捷函数
    "extract_memories",
    "search_memories",
    "organize_memories",
    "update_memory",
    "generate_id",
    "current_timestamp",
    "format_conversation",
    "detect_lang",
]

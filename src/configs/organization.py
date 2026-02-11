# =============================================================================
# 记忆组织配置 - Organization Config
# =============================================================================

from dataclasses import dataclass


@dataclass
class OrganizationConfig:
    """记忆组织配置"""
    # 组织策略
    use_temporal: bool = True       # 使用时间结构
    use_event: bool = True          # 使用事件结构
    use_semantic: bool = True       # 使用语义结构
    use_hierarchy: bool = True      # 使用层级抽象
    
    # 时间结构参数
    time_threshold_minutes: int = 30      # 会话切分的时间阈值
    topic_drift_threshold: float = 0.3    # 主题漂移阈值
    
    # 语义结构参数
    similarity_threshold: float = 0.4     # 相似度阈值
    max_candidates: int = 50              # 最大候选数
    
    # 层级抽象参数
    min_cluster_size: int = 2             # 最小聚类大小
    abstraction_threshold: float = 0.5    # 抽象阈值
    
    # 存储配置
    auto_save_edges: bool = True          # 是否自动保存边
    
    # 调试参数
    verbose: bool = False

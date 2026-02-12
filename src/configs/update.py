# =============================================================================
# 记忆更新配置 - Update Config
# =============================================================================

from dataclasses import dataclass
from typing import Literal


@dataclass
class UpdateConfig:
    """记忆更新配置"""
    # 更新策略
    strategy: Literal["overwrite", "merge", "version", "auto"] = "auto"
    
    # 衰减参数
    decay_rate: float = 0.1              # 基础衰减率
    decay_interval_hours: int = 24       # 衰减计算间隔
    min_decay_score: float = 0.1         # 最小衰减分数
    
    # 遗忘阈值
    archive_threshold: float = 0.3       # 归档阈值
    delete_threshold: float = 0.1        # 删除阈值
    
    # 合并参数 - 调整后的阈值（针对 bge-m3 等模型的特性）
    similarity_threshold: float = 0.7    # 相似度阈值（原0.7，降低以适应实际分数分布）
    merge_confidence_boost: float = 0.1  # 合并后置信度提升
    
    # 更新决策阈值（用于 decide_action）
    high_similarity_threshold: float = 0.5   # 高度相似阈值（原0.85）
    medium_similarity_threshold: float = 0.2  # 中等相似阈值（原0.6）
    llm_judge_threshold: float = 0.05         # 触发LLM判定的最低阈值
    
    # 是否启用 LLM 语义判定
    use_llm_semantic_judge: bool = False  # 当相似度在阈值边界时，使用LLM辅助判断
    
    # 冲突消解参数
    time_priority_weight: float = 0.4    # 时效性权重
    source_priority_weight: float = 0.4  # 来源可信度权重
    
    # 存储配置
    auto_save: bool = True               # 是否自动保存
    
    # 调试参数
    verbose: bool = False

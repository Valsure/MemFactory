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
    
    # 合并参数
    similarity_threshold: float = 0.7    # 相似度阈值
    merge_confidence_boost: float = 0.1  # 合并后置信度提升
    
    # 冲突消解参数
    time_priority_weight: float = 0.4    # 时效性权重
    source_priority_weight: float = 0.4  # 来源可信度权重
    
    # 存储配置
    auto_save: bool = True               # 是否自动保存
    
    # 调试参数
    verbose: bool = False

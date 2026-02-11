# =============================================================================
# 记忆检索配置 - Search Config
# =============================================================================

from dataclasses import dataclass
from typing import Literal


@dataclass
class SearchConfig:
    """记忆检索配置"""
    # 检索策略
    strategy: Literal["passive", "active", "dtr"] = "passive"
    
    # 检索参数
    top_k: int = 10                      # 返回数量
    similarity_threshold: float = 0.3    # 相似度阈值
    
    # DTR参数
    uncertainty_threshold: float = 0.5   # 不确定性阈值
    
    # 时间衰减参数
    use_time_decay: bool = True          # 是否使用时间衰减
    time_decay_lambda: float = 0.04      # 衰减系数（每天）
    
    # 注入策略
    injection_mode: Literal["prefix", "suffix", "partition"] = "partition"
    injection_format: Literal["summary", "slots", "constraints"] = "summary"
    max_inject_chars: int = 2000         # 最大注入字符数
    
    # 用户信息
    user_id: str = "default_user"
    
    # 调试参数
    verbose: bool = False

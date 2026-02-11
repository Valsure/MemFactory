# =============================================================================
# 记忆抽取配置 - Extraction Config
# =============================================================================

from dataclasses import dataclass
from typing import Literal


@dataclass
class ExtractionConfig:
    """记忆抽取配置"""
    # 抽取策略
    strategy: Literal["simple", "react"] = "simple"
    
    # ReAct Agent参数
    max_steps: int = 3  # 最大思考步数
    
    # 存储配置
    auto_save: bool = True  # 是否自动保存到数据库
    auto_embed: bool = True  # 是否自动生成embedding
    
    # 调试参数
    verbose: bool = False
    
    # 用户信息
    user_id: str = "default_user"
    session_id: str = "default_session"

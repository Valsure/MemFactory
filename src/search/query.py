# =============================================================================
# 检索查询 - Query
# =============================================================================

"""
检索查询数据结构
"""

from dataclasses import dataclass
from typing import Optional

try:
    from ..common import current_timestamp
except ImportError:
    from common import current_timestamp


@dataclass
class Query:
    """检索查询"""
    text: str                            # 查询文本
    context: Optional[str] = None        # 上下文
    timestamp: str = ""                  # 时间戳
    intent: Optional[str] = None         # 意图（可选）
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = current_timestamp()

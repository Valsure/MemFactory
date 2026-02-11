# =============================================================================
# 记忆更新数据结构 - Update Data Structures
# =============================================================================

"""
记忆更新相关的数据结构定义
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

try:
    from ..common import current_timestamp
except ImportError:
    from common import current_timestamp


class ConflictType(Enum):
    """冲突类型"""
    FACTUAL = "factual"          # 事实冲突
    PREFERENCE = "preference"    # 偏好冲突
    TEMPORAL = "temporal"        # 时间冲突


@dataclass
class ConflictRecord:
    """冲突记录"""
    conflict_id: str
    memory_id_a: str
    memory_id_b: str
    conflict_type: ConflictType
    description: str
    resolution: Optional[str] = None
    resolved: bool = False
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = current_timestamp()


@dataclass
class UpdateResult:
    """更新结果"""
    action: str                          # 执行的操作
    success: bool                        # 是否成功
    memory_id: str                       # 结果记忆ID
    original_ids: List[str] = field(default_factory=list)  # 原始记忆ID
    conflicts: List[ConflictRecord] = field(default_factory=list)  # 冲突记录
    message: str = ""                    # 消息

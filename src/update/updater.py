# =============================================================================
# 记忆更新主类 - Memory Updater
# =============================================================================

"""
记忆更新器：整合所有更新模块的主入口
"""

from typing import List, Dict, Any

try:
    from ..common import (
        MemoryItem,
        get_neo4j_client, get_milvus_client, get_embedding_client
    )
    from ..configs.update import UpdateConfig
    from .data_structures import ConflictType, UpdateResult
    from .overwriter import MemoryOverwriter
    from .merger import MemoryMerger
    from .version_manager import MemoryVersionManager
    from .conflict_resolver import ConflictResolver
    from .forgetter import MemoryForgetter
except ImportError:
    from common import (
        MemoryItem,
        get_neo4j_client, get_milvus_client, get_embedding_client
    )
    from configs.update import UpdateConfig
    from update.data_structures import ConflictType, UpdateResult
    from update.overwriter import MemoryOverwriter
    from update.merger import MemoryMerger
    from update.version_manager import MemoryVersionManager
    from update.conflict_resolver import ConflictResolver
    from update.forgetter import MemoryForgetter


class MemoryUpdater:
    """
    记忆更新器：整合所有更新模块的主入口
    
    使用方式：
        updater = MemoryUpdater(config)
        result = updater.run(memory, action="merge", related_memories=[...])
    """
    
    def __init__(self, config: UpdateConfig = None):
        self.config = config or UpdateConfig()
        
        # 子模块
        self.overwriter = MemoryOverwriter(self.config)
        self.merger = MemoryMerger(self.config)
        self.version_manager = MemoryVersionManager(self.config)
        self.conflict_resolver = ConflictResolver(self.config)
        self.forgetter = MemoryForgetter(self.config)
        
        # 数据库客户端
        self.neo4j = get_neo4j_client()
        self.milvus = get_milvus_client()
        self.embedding = get_embedding_client()
        
        if self.config.verbose:
            print(f"[MemoryUpdater] 初始化完成，策略: {self.config.strategy}")
    
    def run(self, memory: MemoryItem,
            action: str = None,
            new_content: str = None,
            related_memories: List[MemoryItem] = None) -> UpdateResult:
        """
        执行记忆更新
        
        Args:
            memory: 待更新的记忆
            action: 操作类型 (overwrite/merge/version/auto)
            new_content: 新内容（用于覆盖/版本化）
            related_memories: 相关记忆（用于合并/冲突检测）
            
        Returns:
            更新结果
        """
        action = action or self.config.strategy
        
        # 自动选择策略
        if action == "auto":
            action = self._auto_select_action(memory, related_memories or [])
        
        if action == "overwrite" and new_content:
            return self._do_overwrite(memory, new_content)
        elif action == "merge" and related_memories:
            return self._do_merge([memory] + related_memories)
        elif action == "version" and new_content:
            return self._do_version(memory, new_content)
        else:
            return UpdateResult(
                action="none",
                success=False,
                memory_id=memory.id,
                message="无效的操作或参数"
            )
    
    def _auto_select_action(self, memory: MemoryItem,
                            related_memories: List[MemoryItem]) -> str:
        """自动选择更新策略"""
        if not related_memories:
            return "overwrite"
        
        # 检查是否有可合并的记忆
        for related in related_memories:
            should_merge, similarity = self.merger.should_merge(memory, related)
            if should_merge:
                return "merge"
        
        # 检查是否有冲突
        conflicts = self.conflict_resolver.detect_conflict(memory, related_memories)
        if conflicts:
            return "version"  # 有冲突时使用版本化
        
        return "overwrite"
    
    def decide_action(self, new_memory: MemoryItem,
                      related_memories: List[MemoryItem] = None) -> Dict[str, Any]:
        """
        决定对新记忆的操作（用于Pipeline调用）
        
        这是Pipeline流程中的核心决策函数：
        根据新抽取的记忆和已有相关记忆，决定应该执行什么操作
        
        Args:
            new_memory: 新抽取的候选记忆
            related_memories: 查询到的相关已有记忆
            
        Returns:
            决策结果字典:
            {
                "action": "ADD" | "MERGE" | "OVERWRITE" | "VERSION" | "IGNORE",
                "final_memory": MemoryItem,  # 最终要保存的记忆
                "deprecated_ids": List[str],  # 需要废弃的旧记忆ID
                "reason": str  # 决策原因
            }
        """
        related_memories = related_memories or []
        
        result = {
            "action": "ADD",
            "final_memory": new_memory,
            "deprecated_ids": [],
            "reason": ""
        }
        
        # 没有相关记忆，直接新增
        if not related_memories:
            result["action"] = "ADD"
            result["reason"] = "无相关已有记忆，直接新增"
            if self.config.verbose:
                print(f"[MemoryUpdater] 决策: ADD - {result['reason']}")
            return result
        
        # 检查是否有高度相似的记忆（可能是重复）
        highest_similarity = 0.0
        most_similar_memory = None
        
        for related in related_memories:
            should_merge, similarity = self.merger.should_merge(new_memory, related)
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_memory = related
        
        # 高度相似（>0.85）：可能是重复，忽略或合并
        if highest_similarity > 0.85:
            # 检查内容是否完全相同
            if new_memory.value.strip() == most_similar_memory.value.strip():
                result["action"] = "IGNORE"
                result["reason"] = f"与已有记忆完全相同 (相似度: {highest_similarity:.2f})"
            else:
                # 内容有差异，执行合并
                merged = self.merger.merge_two(most_similar_memory, new_memory)
                result["action"] = "MERGE"
                result["final_memory"] = merged
                result["deprecated_ids"] = [most_similar_memory.id]
                result["reason"] = f"与已有记忆高度相似 (相似度: {highest_similarity:.2f})，执行合并"
        
        # 中等相似（0.6-0.85）：检查是否有冲突
        elif highest_similarity > 0.6:
            conflicts = self.conflict_resolver.detect_conflict(new_memory, [most_similar_memory])
            
            if conflicts:
                # 有冲突，使用版本化或覆盖
                conflict_type = conflicts[0].conflict_type
                
                if conflict_type == ConflictType.FACTUAL:
                    # 事实冲突：保留新信息（用户最新输入）
                    result["action"] = "OVERWRITE"
                    result["final_memory"] = new_memory
                    result["deprecated_ids"] = [most_similar_memory.id]
                    result["reason"] = f"检测到事实冲突，以新信息覆盖"
                else:
                    # 偏好/时间冲突：版本化
                    _, new_version = self.version_manager.create_version(
                        most_similar_memory, new_memory.value, "信息更新"
                    )
                    result["action"] = "VERSION"
                    result["final_memory"] = new_version
                    result["deprecated_ids"] = [most_similar_memory.id]
                    result["reason"] = f"检测到{conflict_type.value}冲突，创建新版本"
            else:
                # 无冲突，合并补充信息
                merged = self.merger.merge_two(most_similar_memory, new_memory)
                result["action"] = "MERGE"
                result["final_memory"] = merged
                result["deprecated_ids"] = [most_similar_memory.id]
                result["reason"] = f"信息互补 (相似度: {highest_similarity:.2f})，执行合并"
        
        # 低相似（<0.6）：作为新记忆添加
        else:
            result["action"] = "ADD"
            result["reason"] = f"与已有记忆相似度较低 ({highest_similarity:.2f})，作为新记忆添加"
        
        if self.config.verbose:
            print(f"[MemoryUpdater] 决策: {result['action']} - {result['reason']}")
        
        return result
    
    def _do_overwrite(self, memory: MemoryItem, new_content: str) -> UpdateResult:
        """执行覆盖"""
        updated = self.overwriter.overwrite(memory, new_content)
        
        if self.config.auto_save:
            self._save_memory(updated)
        
        return UpdateResult(
            action="overwrite",
            success=True,
            memory_id=updated.id,
            original_ids=[memory.id],
            message="覆盖成功"
        )
    
    def _do_merge(self, memories: List[MemoryItem]) -> UpdateResult:
        """执行合并"""
        merged = self.merger.merge_batch(memories)
        
        if self.config.auto_save:
            # 废弃原记忆
            for mem in memories:
                deprecated = self.overwriter.deprecate(mem, "已合并")
                self.neo4j.save_memory(deprecated)
            
            # 保存新记忆
            self._save_memory(merged)
        
        return UpdateResult(
            action="merge",
            success=True,
            memory_id=merged.id,
            original_ids=[m.id for m in memories],
            message=f"合并了 {len(memories)} 条记忆"
        )
    
    def _do_version(self, memory: MemoryItem, new_content: str) -> UpdateResult:
        """执行版本化"""
        archived, new_version = self.version_manager.create_version(
            memory, new_content, "内容更新"
        )
        
        if self.config.auto_save:
            self.neo4j.save_memory(archived)
            self._save_memory(new_version)
        
        return UpdateResult(
            action="version",
            success=True,
            memory_id=new_version.id,
            original_ids=[memory.id],
            message=f"创建新版本 v{new_version.version}"
        )
    
    def _save_memory(self, memory: MemoryItem):
        """保存记忆到数据库"""
        # 生成embedding
        text = f"{memory.key} {memory.value}"
        memory.embedding = self.embedding.embed(text)
        
        # 保存到Milvus
        self.milvus.insert(memory.id, memory.embedding)
        
        # 保存到Neo4j
        self.neo4j.save_memory(memory)
    
    def cleanup(self, memories: List[MemoryItem]) -> Dict[str, List[MemoryItem]]:
        """执行清理任务"""
        result = self.forgetter.auto_cleanup(memories)
        
        if self.config.auto_save:
            for mem in result["archived"]:
                self.neo4j.save_memory(mem)
        
        return result
    
    def reinforce(self, memory: MemoryItem) -> MemoryItem:
        """强化记忆"""
        reinforced = self.forgetter.reinforce(memory)
        
        if self.config.auto_save:
            self.neo4j.save_memory(reinforced)
        
        return reinforced


def update_memory(
    memory: MemoryItem,
    action: str = "auto",
    new_content: str = None,
    related_memories: List[MemoryItem] = None,
    verbose: bool = False
) -> UpdateResult:
    """
    便捷函数：更新记忆
    
    Args:
        memory: 待更新的记忆
        action: 操作类型
        new_content: 新内容
        related_memories: 相关记忆
        verbose: 是否打印调试信息
        
    Returns:
        更新结果
    """
    config = UpdateConfig(
        strategy=action,
        verbose=verbose
    )
    updater = MemoryUpdater(config)
    return updater.run(memory, action, new_content, related_memories)

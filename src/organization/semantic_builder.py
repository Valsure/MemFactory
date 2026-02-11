# =============================================================================
# 语义结构构建器 - Semantic Structure Builder
# =============================================================================

"""
语义结构构建器：负责对记忆进行语义结构化和关系建模
"""

from typing import List

try:
    from ..common import MemoryItem, Edge, RelationType, get_embedding_client
    from ..configs.organization import OrganizationConfig
except ImportError:
    from common import MemoryItem, Edge, RelationType, get_embedding_client
    from configs.organization import OrganizationConfig


class SemanticStructureBuilder:
    """
    语义结构构建器
    负责对记忆进行语义结构化和关系建模
    """
    
    def __init__(self, config: OrganizationConfig):
        self.config = config
        self.embedding = get_embedding_client()
    
    def build(self, memories: List[MemoryItem]) -> List[Edge]:
        """
        构建语义结构
        
        Returns:
            语义关系边列表
        """
        edges = []
        
        # 为每个记忆生成embedding（如果没有）
        for mem in memories:
            if not mem.embedding:
                text = f"{mem.key} {mem.value}"
                mem.embedding = self.embedding.embed(text)
        
        # 计算两两相似度并建立边
        for i, mem_a in enumerate(memories):
            for j in range(i + 1, len(memories)):
                mem_b = memories[j]
                
                # 计算相似度
                similarity = self.embedding.similarity(mem_a.embedding, mem_b.embedding)
                
                if similarity >= self.config.similarity_threshold:
                    # 判断关系类型
                    relation_type = self._classify_relation(mem_a, mem_b, similarity)
                    
                    edge = Edge(
                        source_id=mem_a.id,
                        target_id=mem_b.id,
                        relation_type=relation_type,
                        weight=similarity,
                        metadata={"type": "semantic"}
                    )
                    edges.append(edge)
        
        if self.config.verbose:
            print(f"[SemanticStructureBuilder] 边: {len(edges)}")
        
        return edges
    
    def _classify_relation(self, mem_a: MemoryItem, mem_b: MemoryItem, 
                          similarity: float) -> str:
        """分类关系类型"""
        # 检查tag重叠
        tags_a = set(mem_a.tags)
        tags_b = set(mem_b.tags)
        tag_overlap = len(tags_a & tags_b) / max(len(tags_a | tags_b), 1)
        
        if similarity >= 0.8 and tag_overlap >= 0.5:
            return RelationType.SAME_TOPIC.value
        
        return RelationType.RELATED_TO.value

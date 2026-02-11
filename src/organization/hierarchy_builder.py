# =============================================================================
# 层级抽象构建器 - Hierarchy Builder
# =============================================================================

"""
层级抽象构建器：负责构建索引树和生成抽象节点
"""

from typing import List, Dict, Tuple, Optional

try:
    from ..common import MemoryItem, Edge, RelationType, get_llm_client, get_embedding_client, generate_id
    from ..configs.organization import OrganizationConfig
    from ..templates.organization_prompts import ABSTRACTION_PROMPT
    from .data_structures import EventUnit, AbstractionNode
except ImportError:
    from common import MemoryItem, Edge, RelationType, get_llm_client, get_embedding_client, generate_id
    from configs.organization import OrganizationConfig
    from templates.organization_prompts import ABSTRACTION_PROMPT
    from organization.data_structures import EventUnit, AbstractionNode


class HierarchyBuilder:
    """
    层级抽象构建器
    负责构建索引树和生成抽象节点
    """
    
    def __init__(self, config: OrganizationConfig):
        self.config = config
        self.llm = get_llm_client()
        self.embedding = get_embedding_client()
        self.abstractions: Dict[str, AbstractionNode] = {}
    
    def build(self, memories: List[MemoryItem], 
              events: List[EventUnit] = None) -> Tuple[Dict[str, AbstractionNode], List[Edge]]:
        """
        构建层级抽象
        
        Returns:
            (抽象节点字典, 抽象关系边列表)
        """
        # 1. 聚合相似记忆
        clusters = self._aggregate_similar(memories)
        
        # 2. 生成抽象
        for cluster in clusters:
            if len(cluster) >= self.config.min_cluster_size:
                abstraction = self._generate_abstraction(cluster, events or [])
                if abstraction:
                    self.abstractions[abstraction.node_id] = abstraction
        
        # 3. 生成抽象关系边
        edges = self._get_abstraction_edges()
        
        if self.config.verbose:
            print(f"[HierarchyBuilder] 抽象: {len(self.abstractions)}, 边: {len(edges)}")
        
        return self.abstractions, edges
    
    def _aggregate_similar(self, memories: List[MemoryItem]) -> List[List[MemoryItem]]:
        """聚合相似记忆"""
        clusters = []
        used = set()
        
        for i, mem in enumerate(memories):
            if mem.id in used:
                continue
            
            cluster = [mem]
            used.add(mem.id)
            
            for j in range(i + 1, len(memories)):
                other = memories[j]
                if other.id in used:
                    continue
                
                # 计算相似度
                if mem.embedding and other.embedding:
                    sim = self.embedding.similarity(mem.embedding, other.embedding)
                    if sim >= self.config.abstraction_threshold:
                        cluster.append(other)
                        used.add(other.id)
            
            if len(cluster) >= self.config.min_cluster_size:
                clusters.append(cluster)
        
        return clusters
    
    def _generate_abstraction(self, memories: List[MemoryItem], 
                              events: List[EventUnit]) -> Optional[AbstractionNode]:
        """从记忆簇生成抽象模式"""
        # 收集记忆内容
        contents = [m.value for m in memories]
        memory_contents = "\n".join(f"- {c}" for c in contents)
        
        prompt = ABSTRACTION_PROMPT.format(memory_contents=memory_contents)
        
        response = self.llm.chat(
            system_prompt="You are a pattern induction expert.",
            user_prompt=prompt
        )
        
        result = self.llm.parse_json(response)
        
        if not result:
            return None
        
        return AbstractionNode(
            node_id=generate_id(),
            label=result.get("label", "Unknown pattern"),
            condition=result.get("condition", ""),
            solution=result.get("solution", ""),
            verification=result.get("verification", ""),
            support_ids=[m.id for m in memories]
        )
    
    def _get_abstraction_edges(self) -> List[Edge]:
        """生成抽象关系边"""
        edges = []
        
        for abs_id, abstraction in self.abstractions.items():
            for support_id in abstraction.support_ids:
                edge = Edge(
                    source_id=abs_id,
                    target_id=support_id,
                    relation_type=RelationType.CONTAINS.value,
                    weight=abstraction.confidence,
                    metadata={"type": "abstraction", "pattern_label": abstraction.label}
                )
                edges.append(edge)
        
        return edges

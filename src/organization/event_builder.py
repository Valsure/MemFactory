# =============================================================================
# 事件结构构建器 - Event Structure Builder
# =============================================================================

"""
事件结构构建器：负责从记忆中抽取事件要素并建立事件间关系
"""

from typing import List, Dict, Tuple, Optional

try:
    from ..common import MemoryItem, Edge, RelationType, get_llm_client, generate_id
    from ..configs.organization import OrganizationConfig
    from ..templates.organization_prompts import EVENT_EXTRACTION_PROMPT
    from .data_structures import EventUnit
except ImportError:
    from common import MemoryItem, Edge, RelationType, get_llm_client, generate_id
    from configs.organization import OrganizationConfig
    from templates.organization_prompts import EVENT_EXTRACTION_PROMPT
    from organization.data_structures import EventUnit


class EventStructureBuilder:
    """
    事件结构构建器
    负责从记忆中抽取事件要素并建立事件间关系
    """
    
    def __init__(self, config: OrganizationConfig):
        self.config = config
        self.llm = get_llm_client()
        self.events: Dict[str, EventUnit] = {}
        self.memory_to_event: Dict[str, str] = {}
    
    def build(self, memories: List[MemoryItem]) -> Tuple[List[EventUnit], List[Edge]]:
        """
        构建事件结构
        
        Returns:
            (事件列表, 事件关系边列表)
        """
        # 1. 抽取事件
        events = self._extract_events(memories)
        
        # 2. 推断事件关系
        edges = self._infer_event_relations(events)
        
        if self.config.verbose:
            print(f"[EventStructureBuilder] 事件: {len(events)}, 边: {len(edges)}")
        
        return events, edges
    
    def _extract_events(self, memories: List[MemoryItem]) -> List[EventUnit]:
        """从记忆中抽取事件要素"""
        events = []
        
        for memory in memories:
            event = self._extract_single_event(memory)
            if event:
                events.append(event)
                self.events[event.event_id] = event
                self.memory_to_event[memory.id] = event.event_id
        
        return events
    
    def _extract_single_event(self, memory: MemoryItem) -> Optional[EventUnit]:
        """使用LLM抽取单个事件"""
        prompt = EVENT_EXTRACTION_PROMPT.format(memory_value=memory.value)
        
        response = self.llm.chat(
            system_prompt="You are an event extraction expert.",
            user_prompt=prompt
        )
        
        result = self.llm.parse_json(response)
        
        if not result or result.get("valid") == False:
            # 使用简单规则抽取
            return self._rule_based_extract(memory)
        
        return EventUnit(
            event_id=generate_id(),
            agent=result.get("agent"),
            action=result.get("action", "record"),
            object=result.get("object"),
            outcome=result.get("outcome"),
            context=result.get("context"),
            timestamp=memory.created_at,
            source_memory_id=memory.id
        )
    
    def _rule_based_extract(self, memory: MemoryItem) -> EventUnit:
        """基于规则的简单事件抽取"""
        text = memory.value
        
        # 简单规则
        agent = "用户"
        action = "记录"
        
        keywords = {
            "喜欢": ("用户", "表达偏好"),
            "不喜欢": ("用户", "表达偏好"),
            "完成": ("用户", "完成任务"),
            "计划": ("用户", "制定计划"),
            "工作": ("用户", "工作相关"),
        }
        
        for keyword, (ag, act) in keywords.items():
            if keyword in text:
                agent, action = ag, act
                break
        
        return EventUnit(
            event_id=generate_id(),
            agent=agent,
            action=action,
            object=None,
            outcome=None,
            context=None,
            timestamp=memory.created_at,
            source_memory_id=memory.id
        )
    
    def _infer_event_relations(self, events: List[EventUnit]) -> List[Edge]:
        """推断事件之间的关系"""
        edges = []
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        for i, event_a in enumerate(sorted_events):
            for j in range(i + 1, min(i + 5, len(sorted_events))):
                event_b = sorted_events[j]
                
                # 推断关系
                relation_type, confidence = self._infer_relation(event_a, event_b)
                
                if confidence >= 0.5:
                    edge = Edge(
                        source_id=event_a.source_memory_id,
                        target_id=event_b.source_memory_id,
                        relation_type=relation_type,
                        weight=confidence,
                        metadata={"type": "event"}
                    )
                    edges.append(edge)
        
        return edges
    
    def _infer_relation(self, event_a: EventUnit, event_b: EventUnit) -> Tuple[str, float]:
        """推断两个事件之间的关系"""
        action_a = event_a.action.lower() if event_a.action else ""
        action_b = event_b.action.lower() if event_b.action else ""
        
        # 简单规则推断
        if "计划" in action_a and ("完成" in action_b or "执行" in action_b):
            return RelationType.CAUSES.value, 0.8
        
        if "问题" in action_a and "解决" in action_b:
            return RelationType.RESOLVES.value, 0.85
        
        return RelationType.FOLLOWS.value, 0.6

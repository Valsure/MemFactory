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
        """使用 LLM 推断两个事件之间的关系"""
        return self._llm_infer_relation(event_a, event_b)
    
    def _llm_infer_relation(self, event_a: EventUnit, event_b: EventUnit) -> Tuple[str, float]:
        """使用 LLM 推断事件关系"""
        # 构建事件描述
        event_a_msg = self._format_event_message(event_a)
        event_b_msg = self._format_event_message(event_b)
        
        # 获取所有可用的关系类型
        valid_relation_types = [rt.value for rt in RelationType]
        
        prompt = f"""Given two events in chronological order, determine if there is a semantic relationship between them.

Event A:
- {event_a_msg}

Event B:
- {event_b_msg}

Please determine:
1. Did Event A cause Event B? (causes - causal relationship)
2. Does Event B resolve the problem described in Event A? (resolves - resolution relationship)
3. Does Event B depend on Event A? (depends_on - dependency relationship)
4. Do both belong to the same topic? (same_topic - topic relationship)
5. Does Event A contain Event B? (contains - containment relationship)
6. Is there only a general association? (related_to - association relationship)
7. Is there only temporal sequence without semantic association? (follows - temporal relationship)

Output JSON format:
{{"relation": "causes/resolves/depends_on/same_topic/contains/related_to/follows", "confidence": 0.0-1.0, "reasoning": "brief explanation of judgment basis"}}

Output JSON only."""

        response = self.llm.chat(
            system_prompt="You are an event relationship analysis expert, skilled at judging causal, temporal, and dependency relationships between events.",
            user_prompt=prompt
        )
        
        result = self.llm.parse_json(response)
        
        if not result:
            # LLM 解析失败，回退到规则方法
            return self._rule_based_infer_relation(event_a, event_b)
        
        relation_type = result.get("relation", RelationType.RELATED_TO.value)
        confidence = result.get("confidence", 0.5)
        
        # 验证关系类型是否有效
        if relation_type not in valid_relation_types:
            relation_type = RelationType.RELATED_TO.value
        
        # 确保 confidence 在合理范围内
        confidence = max(0.0, min(1.0, float(confidence)))
        
        return relation_type, confidence
    
    def _format_event_message(self, event: EventUnit) -> str:
        """格式化事件消息"""
        parts = []
        
        # 构建主干描述
        if event.agent and event.action:
            if event.object:
                parts.append(f"{event.agent} {event.action} {event.object}")
            else:
                parts.append(f"{event.agent} {event.action}")
        elif event.action:
            parts.append(event.action)
        
        # 添加结果
        if event.outcome:
            parts.append(f"Result: {event.outcome}")
        
        # 添加上下文
        if event.context:
            parts.append(f"Context: {event.context}")
        
        return ", ".join(parts) if parts else "Unknown event"
    
    def _rule_based_infer_relation(self, event_a: EventUnit, event_b: EventUnit) -> Tuple[str, float]:
        """基于规则的关系推断（作为 LLM 失败时的回退方案）"""
        action_a = event_a.action.lower() if event_a.action else ""
        action_b = event_b.action.lower() if event_b.action else ""
        
        # 因果关系
        if "计划" in action_a and ("完成" in action_b or "执行" in action_b):
            return RelationType.CAUSES.value, 0.8
        
        # 解决关系
        if "问题" in action_a and "解决" in action_b:
            return RelationType.RESOLVES.value, 0.85
        
        # 依赖关系
        if "需要" in action_b or "依赖" in action_b:
            return RelationType.DEPENDS_ON.value, 0.7
        
        # 默认时序关系
        return RelationType.FOLLOWS.value, 0.6

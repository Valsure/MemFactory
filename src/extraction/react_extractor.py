# =============================================================================
# ReAct Agent 记忆抽取器 - ReAct Extractor
# =============================================================================

"""
基于ReAct范式的记忆抽取Agent
可以在生成最终记忆之前，自主决定是否需要查阅历史数据库以补充上下文
"""

from typing import List, Dict, Optional

try:
    from ..common import (
        MemoryItem, ConversationMessage, ExtractionResult,
        get_llm_client, get_embedding_client, get_neo4j_client, get_milvus_client,
        generate_id, format_conversation
    )
    from ..configs.extraction import ExtractionConfig
    from ..templates.extraction_prompts import REACT_SYSTEM_PROMPT_EN
except ImportError:
    from common import (
        MemoryItem, ConversationMessage, ExtractionResult,
        get_llm_client, get_embedding_client, get_neo4j_client, get_milvus_client,
        generate_id, format_conversation
    )
    from configs.extraction import ExtractionConfig
    from templates.extraction_prompts import REACT_SYSTEM_PROMPT_EN


class MemoryBuffer:
    """记忆缓冲区：用于存储跨轮次的碎片信息"""
    
    def __init__(self):
        self._buffer: List[str] = []
    
    def append(self, fragment: str):
        """添加片段"""
        self._buffer.append(fragment)
    
    def get_contents(self) -> List[str]:
        """获取所有内容"""
        return self._buffer.copy()
    
    def clear(self):
        """清空缓冲区"""
        self._buffer.clear()
    
    def is_empty(self) -> bool:
        """是否为空"""
        return len(self._buffer) == 0


class ReActExtractor:
    """
    基于ReAct范式的记忆抽取Agent
    可以在生成最终记忆之前，自主决定是否需要查阅历史数据库以补充上下文
    """
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.llm = get_llm_client()
        self.embedding = get_embedding_client()
        self.neo4j = get_neo4j_client()
        self.milvus = get_milvus_client()
        self.buffer = MemoryBuffer()
    
    def extract(self, messages: List[ConversationMessage]) -> ExtractionResult:
        """
        ReAct主循环：Observation -> Thought -> Action -> Observation
        
        Args:
            messages: 对话消息列表
            
        Returns:
            抽取结果
        """
        # 初始化上下文
        conversation = format_conversation(messages)
        context = {
            "current": conversation,
            "history": messages[-5:] if len(messages) > 5 else messages,
            "buffer": self.buffer.get_contents(),
            "rag_context": []
        }
        
        thought_trace = []  # 记录思考轨迹
        
        for step in range(self.config.max_steps):
            if self.config.verbose:
                print(f"\n[ReActExtractor] Step {step + 1}/{self.config.max_steps}")
            
            # 1. 思考阶段 (Reasoning)
            agent_decision = self._think(context, thought_trace)
            
            if not agent_decision:
                continue
            
            # 2. 行动阶段 (Acting)
            action = agent_decision.get("action", "Ignore")
            reasoning = agent_decision.get("thought", "")
            params = agent_decision.get("action_params", {})
            
            thought_trace.append(f"Thought: {reasoning}")
            
            if self.config.verbose:
                print(f"  Thought: {reasoning[:100]}...")
                print(f"  Action: {action}")
            
            # --- 工具分发逻辑 ---
            
            # [Action 1] 检索增强：指代不清或需要验证事实时触发
            if action == "SearchContext":
                query = params.get("query", "")
                search_results = self._search_context(query)
                context["rag_context"] = search_results
                thought_trace.append(f"Observation: Found {len(search_results)} related memories")
                continue
            
            # [Action 2] 缓冲挂起：信息不全，等待后续补充
            elif action == "UpdateBuffer":
                fragment = params.get("fragment", "")
                self.buffer.append(fragment)
                return ExtractionResult(
                    memory_list=[],
                    summary="信息不完整，已暂存到缓冲区",
                    status="BUFFERED"
                )
            
            # [Action 3] 提交记忆：信息完整且有价值
            elif action == "CommitMemory":
                memory_list = self._parse_memories(params)
                summary = params.get("summary", "")
                self.buffer.clear()
                return ExtractionResult(
                    memory_list=memory_list,
                    summary=summary,
                    status="SUCCESS"
                )
            
            # [Action 4] 忽略：判定为闲聊
            elif action == "Ignore":
                return ExtractionResult(
                    memory_list=[],
                    summary="判定为闲聊，无需抽取记忆",
                    status="IGNORED"
                )
        
        # 兜底：超过最大步数，强制结束
        return ExtractionResult(
            memory_list=[],
            summary="超时",
            status="TIMEOUT"
        )
    
    def _think(self, context: Dict, trace: List[str]) -> Optional[Dict]:
        """Agent思考：根据当前上下文决定下一步行动"""
        user_prompt = f"""current conversation:
{context['current']}

buffer content: {context['buffer']}

history retrieval results: {context['rag_context']}

reasoning trace: {trace}

Please decide next move."""
        
        response = self.llm.chat(
            system_prompt=REACT_SYSTEM_PROMPT_EN,
            user_prompt=user_prompt
        )
        
        return self.llm.parse_json(response)
    
    def _search_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索历史记忆"""
        # 生成查询向量
        query_emb = self.embedding.embed(query)
        
        # 向量检索
        results = self.milvus.search(query_emb, top_k=top_k)
        
        # 获取记忆详情
        memories = []
        for memory_id, score in results:
            mem = self.neo4j.get_memory(memory_id)
            if mem:
                memories.append({
                    "key": mem.key,
                    "value": mem.value,
                    "score": score
                })
        
        return memories
    
    def _parse_memories(self, params: Dict) -> List[MemoryItem]:
        """解析记忆参数为MemoryItem列表"""
        memory_list = []
        for item in params.get("memory_list", []):
            memory = MemoryItem(
                id=generate_id(),
                key=item.get("key", ""),
                value=item.get("value", ""),
                memory_type=item.get("memory_type", "UserMemory"),
                tags=item.get("tags", []),
                user_id=self.config.user_id,
                session_id=self.config.session_id
            )
            memory_list.append(memory)
        return memory_list

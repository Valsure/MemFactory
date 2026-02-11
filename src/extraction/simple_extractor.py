# =============================================================================
# 简单记忆抽取器 - Simple Extractor
# =============================================================================

"""
简单记忆抽取器：直接使用LLM进行一次性抽取
"""

from typing import List

try:
    from ..common import (
        MemoryItem, ConversationMessage, ExtractionResult,
        get_llm_client, generate_id, format_conversation
    )
    from ..configs.extraction import ExtractionConfig
    from ..templates.extraction_prompts import EXTRACTION_PROMPT_EN, EXTRACTION_PROMPT_ZH
    from .utils import detect_lang
except ImportError:
    from common import (
        MemoryItem, ConversationMessage, ExtractionResult,
        get_llm_client, generate_id, format_conversation
    )
    from configs.extraction import ExtractionConfig
    from templates.extraction_prompts import EXTRACTION_PROMPT_EN, EXTRACTION_PROMPT_ZH
    from extraction.utils import detect_lang


class SimpleExtractor:
    """
    简单记忆抽取器
    直接使用LLM进行一次性抽取
    """
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.llm = get_llm_client()
    
    def extract(self, messages: List[ConversationMessage]) -> ExtractionResult:
        """
        从对话中抽取记忆
        
        Args:
            messages: 对话消息列表
            
        Returns:
            抽取结果
        """
        # 格式化对话
        conversation = format_conversation(messages)
        
        # 检测语言并选择对应的prompt模板
        lang = detect_lang(conversation)
        template = EXTRACTION_PROMPT_ZH if lang == "zh" else EXTRACTION_PROMPT_EN
        
        # 构建prompt
        prompt = template.replace("{conversation}", conversation)
        
        # 调用LLM
        response = self.llm.chat(
            system_prompt="You are an expert of extracting memories, and you output JSON only",
            user_prompt=prompt
        )
        
        # 解析结果
        result = self.llm.parse_json(response)
        
        if not result:
            # 解析失败，返回空结果
            return ExtractionResult(
                memory_list=[],
                summary="抽取失败",
                status="FAILED"
            )
        
        # 转换为MemoryItem列表
        memory_list = []
        for item in result.get("memory_list", []):
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
        
        if self.config.verbose:
            print(f"[SimpleExtractor] 抽取了 {len(memory_list)} 条记忆")
        
        return ExtractionResult(
            memory_list=memory_list,
            summary=result.get("summary", ""),
            status="SUCCESS"
        )

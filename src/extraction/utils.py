# =============================================================================
# 记忆抽取工具函数 - Extraction Utils
# =============================================================================

"""
记忆抽取相关的工具函数
"""

import re


def detect_lang(text: str) -> str:
    """
    检测文本语言（中文或英文）
    
    Args:
        text: 待分析的文本
        
    Returns:
        "zh" 表示中文, "en" 表示英文（默认）
    """
    try:
        if not text or not isinstance(text, str):
            return "en"
        cleaned_text = text
        # 移除角色和时间戳
        cleaned_text = re.sub(
            r"\b(user|assistant|query|answer)\s*:", "", cleaned_text, flags=re.IGNORECASE
        )
        cleaned_text = re.sub(r"\[[\d\-:\s]+\]", "", cleaned_text)

        # 提取中文字符
        chinese_pattern = r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf\uf900-\ufaff]"
        chinese_chars = re.findall(chinese_pattern, cleaned_text)
        text_without_special = re.sub(r"[\s\d\W]", "", cleaned_text)
        if text_without_special and len(chinese_chars) / len(text_without_special) > 0.3:
            return "zh"
        return "en"
    except Exception:
        return "en"

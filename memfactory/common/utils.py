import json
import os
import sys
import torch
import json
from typing import List, Optional, Union, Any, Dict
from concurrent.futures import ThreadPoolExecutor

# =============================================================================
# LLM 服务
# =============================================================================
from openai import OpenAI
try:
    from dotenv import load_dotenv
    # 尝试从多个位置加载 .env
    for env_path in ['.env', '../.env', '../../.env']:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            break
except ImportError:
    print("警告：无法加载环境变量，无法使用 OpenAI 等服务")
    pass  

# OpenAI LLM API 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-nano")
class LLMClient:
    """
    LLM客户端：封装OpenAI API调用
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        self.model = LLM_MODEL
        self._initialized = True
        print(f"[LLMClient] 已初始化，模型: {self.model}")
    
    def chat(self, system_prompt: str, user_prompt: str, 
             temperature: float = 0.3) -> str:
        """
        调用LLM进行对话
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户输入
            temperature: 温度参数
            
        Returns:
            LLM响应文本
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLMClient] API调用失败: {e}")
            return ""
    
    def parse_json(self, response: str) -> Optional[Dict]:
        """解析JSON响应"""
        try:
            # 尝试提取JSON块
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            # 清理空白字符
            response = response.strip()
            # 处理思维链
            if response.startswith("<think>"):
                response = response.split("</think>")[-1]
                response = response.strip()
            # 尝试提取JSON对象（处理可能存在的<think>标签或其他前缀）
            if not response.startswith("{"):
                print("[LLMClient-parse_json] 回答不是 { 开头无法解析", response[:100])

            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"[LLMClient] JSON解析失败: {e}")
            return None


TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

TEMPLATE_FINAL_BOXED = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""

JUDGE_PROMPT = """Please judge whether the predicted answer is correct based on the standard answer.

Question: {question}
Standard Answer: {answer}
Predicted Answer: {prediction}

Is the predicted answer consistent with the standard answer? Please output only "True" or "False".
"""

def extract_boxed_content(text):
    """
    Extracts the content inside the last \boxed{...} in the text.
    Handles nested braces correctly.
    """
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    
    i = idx + 6
    while i < len(text) and text[i].isspace():
        i += 1
        
    if i >= len(text) or text[i] != '{':
        return None
    
    start_brace = i
    balance = 0
    content_start = start_brace + 1
    
    for j in range(start_brace, len(text)):
        if text[j] == '{':
            balance += 1
        elif text[j] == '}':
            balance -= 1
            if balance == 0:
                return text[content_start:j]
                
    return None

def evaluate_memory_agent(response, ground_truth, question="", llm_client=None):
    try:
        boxed_content = extract_boxed_content(response)
        if boxed_content is None:
            return 0.0
            
        assert llm_client is not None, "llm_client 不能为空"
        llm = llm_client
        judge_prompt = JUDGE_PROMPT.format(
            question=question, 
            answer=ground_truth, 
            prediction=boxed_content
        )
        
        judge_result = llm.chat("You are an impartial judge.", judge_prompt)
        
        if "<think>" in judge_result:
            if "</think>" in judge_result:
                judge_result = judge_result.split("</think>")[-1].strip()
            else:
                judge_result = judge_result[-100:].strip()
                
        if "True" in judge_result:
            return 1.0
        return 0.0
    except Exception as e:
        print(f"Evaluation Error: {e}")
        return 0.0

def evaluate_memory_agent_batch(responses, ground_truths, questions, max_workers=16, llm_client=None):
    if llm_client is None:
        assert False, "llm_client (最好）不能为空"
        # llm_client = LLMClient() optional
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(len(responses)):
            futures.append(
                executor.submit(
                    evaluate_memory_agent, 
                    responses[i], 
                    ground_truths[i] if isinstance(ground_truths, list) else ground_truths,
                    questions[i] if isinstance(questions, list) else questions,
                    llm_client
                )
            )
        scores = [f.result() for f in futures]
    return scores


def parse_json_from_text(response: str) -> Optional[Dict]:
        """解析JSON响应"""
        try:
            # 尝试提取JSON块
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            # 清理空白字符
            response = response.strip()
            # 处理思维链
            if response.startswith("<think>"):
                response = response.split("</think>")[-1]
                response = response.strip()
            # 尝试提取JSON对象（处理可能存在的<think>标签或其他前缀）
            if not response.startswith("{"):
                print("extract 结果不是 { 开头无法解析", response[:100])

            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"extract 结果 JSON 解析失败: {e}")
            return {}
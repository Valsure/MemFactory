import json
import os
import sys
import torch
from typing import List, Optional, Union, Any, Dict
from concurrent.futures import ThreadPoolExecutor

# Import LLMClient from src.common if available
try:
    # Attempt to import from project root structure
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.common import LLMClient
except ImportError:
    # Fallback or mock if src.common is not found (for standalone usage)
    class LLMClient:
        def chat(self, system, user):
            return "True" # Mock for now

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
            
        llm = llm_client if llm_client else LLMClient()
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
        llm_client = LLMClient()
        
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

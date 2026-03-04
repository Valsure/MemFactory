import json
import re
import sys
import os

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.common import LLMClient
except ImportError:
    # Fallback if running from a different location
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from src.common import LLMClient

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
    # Find the last occurrence of \boxed
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    
    i = idx + 6
    
    # Skip whitespace if any (though typically \boxed is immediately followed by {)
    while i < len(text) and text[i].isspace():
        i += 1
        
    if i >= len(text) or text[i] != '{':
        # Malformed or different format (e.g. \boxed 123)
        # For our purpose, we strictly require \boxed{...}
        return None
    
    # Now track braces to find the matching closing brace
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
                
    # If we reach here, braces were not balanced
    return None

from concurrent.futures import ThreadPoolExecutor

def evaluate_memory_agent(response, ground_truth, question="", llm_client=None):
    try:
        # 1. Extract boxed content
        boxed_content = extract_boxed_content(response)
        
        # 2. If no boxed content found, return 0.0
        if boxed_content is None:
            # print(f"Format Error: No \\boxed{{}} found in response: {response[-100:]}") # Debug log
            return 0.0
            
        # 3. Use LLM to judge
        llm = llm_client if llm_client else LLMClient()
        # We focus on comparing the extracted answer with the ground truth
        # The question is provided for context but the core task is equivalence check
        judge_prompt = JUDGE_PROMPT.format(
            question=question, 
            answer=ground_truth, 
            prediction=boxed_content
        )
        
        judge_result = llm.chat("You are an impartial judge.", judge_prompt)
        
        # Clean up thinking process if present
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
    """
    Parallel evaluation of memory agents using ThreadPoolExecutor.
    """
    if llm_client is None:
        llm_client = LLMClient()
        
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
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
            
        # Collect results in order
        scores = [f.result() for f in futures]
    return scores

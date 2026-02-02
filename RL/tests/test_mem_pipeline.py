
import json
import sys
import os
from typing import List

# Add project root to sys.path to allow imports from src and RL
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.common import (
    MemoryItem, 
    ConversationMessage, 
    get_llm_client, 
    get_memory_store,
    format_conversation
)
from RL.mem_utils import (
    construct_extraction_prompt, 
    construct_update_prompt, 
    MemoryEvaluator
)

def load_first_data_item(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data[0]

def main():
    print("=" * 60)
    print("Memory Pipeline Test Script")
    print("=" * 60)

    # 1. Load Data
    # Path relative to script execution or absolute. 
    # Assuming running from Memory-CookBook root:
    data_path = "/home/guozl/project/MemRL/Memory-CookBook/scripts/processed_locomo_copy.json"
    item = load_first_data_item(data_path)
    
     

    # 2. Parse Components
    # Parse Context Memory (M)
    memory = item["M"]
    context_memory = item["M"]
    print(f"Context Memories loaded: {len(context_memory)}")
    
    # Parse Conversation (f) -> Fact
    msgs = []
    # Hardcode timestamp to match the dataset's ground truth context (May 2023)
    # The gold answer "7 May 2023" implies the conversation "yesterday" was May 7, so conversation is May 8.
    conversation_date = "2023-05-08T10:00:00"
    
    msgs = item["f"]
    
    # Query and Answer
    query = item["q"]
    answer = item["a"]
    print(f"Query: {query}")
    print(f"Gold Answer: {answer}")
    
    # Initialize LLM
    llm = get_llm_client()
    
    # 3. Step 1: Memory Extraction
    print("\n[Step 1] Extraction")
    # construct_extraction_prompt expects List[ConversationMessage] (despite type hint saying str in some places)
    
    extraction_prompt = construct_extraction_prompt(msgs)
    # print(f"Extraction Prompt Preview:\n{extraction_prompt[:200]}...\n")
    
    print("Calling LLM for extraction...")
    extraction_output = llm.chat("You are a helpful assistant.", extraction_prompt)
    print(f"Extraction Output:\n{extraction_output}")
    
    # 4. Step 2: Update Planning
    print("\n[Step 2] Update Planning")
    update_prompt = construct_update_prompt(context_memory, extraction_output)
    # print(f"Update Prompt Preview:\n{update_prompt[:200]}...\n")
    
    print("Calling LLM for update plan...")
    update_plan_output = llm.chat("You are a helpful assistant.", update_prompt)
    print(f"Update Plan Output:\n{update_plan_output}")
    
    # 5. Step 3: Evaluation
    print("\n[Step 3] Evaluation")
    evaluator = MemoryEvaluator()
    
    # Fact string for evaluator
    
    print("Starting evaluation (this includes resetting memory, applying updates, retrieval, QA, and judging)...")
    reward = evaluator.evaluate(
        memory=memory,
        fact=msgs,
        query=query,
        answer=answer,
        context_memory=context_memory,
        extraction_output=extraction_output,
        update_plan_output=update_plan_output
    )
    
    print("=" * 60)
    print(f"Final Reward: {reward}")
    print("=" * 60)

if __name__ == "__main__":
    main()

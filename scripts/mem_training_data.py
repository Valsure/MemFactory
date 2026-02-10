
import json
import os
import sys
import math
import random
from tqdm import tqdm
from typing import List, Dict, Any

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.common import (
    MemoryItem, 
    get_llm_client, 
    get_memory_store,
    format_conversation,
    ConversationMessage
)
from RL.mem_utils import (
    construct_extraction_prompt,
    parse_json_from_text
)

def process_data(source_path: str, target_path: str, k: int = 8, max_items: int = 512):
    print(f"Processing data from {source_path} to {target_path}")
    
    # Initialize components
    llm = get_llm_client()
    store = get_memory_store()
    
    # Load source data
    with open(source_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    processed_ids = set()
    if os.path.exists(target_path):
        with open(target_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        sample_id = item.get("sample_id", "")
                        trigger_id = item.get("trigger_id", "")
                        q = item.get("q", "")
                        unique_id = f"{sample_id}||{trigger_id}||{q}"
                        processed_ids.add(unique_id)
                    except json.JSONDecodeError:
                        continue
    
    print(f"Found {len(processed_ids)} already processed items.")
    
    count = 0
    with open(target_path, 'a', encoding='utf-8') as f_out:
        for item in tqdm(source_data, desc="Processing"):
            if count >= max_items:
                break
                
            sample_id = item.get("sample_id", "")
            trigger_id = item.get("trigger_id", "")
            q = item.get("q", "")
            unique_id = f"{sample_id}||{trigger_id}||{q}"
            if unique_id in processed_ids:
                continue
            
            print(f"Processing item {sample_id}...")
            
            try:
                store.use_mock = True
                store.from_list(item["M"])
                facts = item["f"]
                if not facts:
                    print(f"Skipping {sample_id}: No facts.")
                    continue
                    
                recent_facts = facts[-4:]
                extraction_prompt = construct_extraction_prompt(recent_facts)
                extraction_output = llm.chat("You are a memory extraction expert.", extraction_prompt)
                ext_json = parse_json_from_text(extraction_output)
                
                extracted_memories = []
                if ext_json and "memory_list" in ext_json:
                    extracted_memories = ext_json["memory_list"]
                
                retrieved_context = []
                
                if extracted_memories:
                    # 1. Semantic Search
                    # Each extracted memory participates in search, top 3 for each
                    semantic_memories_map = {} # id -> MemoryItem
                    
                    for mem_dict in extracted_memories:
                        query_text = f"{mem_dict.get('key', '')}: {mem_dict.get('value', '')}"
                        # top_k=3 for each candidate
                        results = store.search_similar(query_text, top_k=3)
                        for mem_item, score in results:
                            semantic_memories_map[mem_item.id] = mem_item
                            
                    # 2. Temporal Search
                    # Get latest 10 memories, pick random 5
                    all_memories = store.get_all()
                    
                    temporal_candidates = all_memories[-10:]
                    temporal_selection = []
                    if temporal_candidates:
                        k_temporal = min(5, len(temporal_candidates))
                        temporal_selection = random.sample(temporal_candidates, k_temporal)
                        
                    temporal_memories_map = {m.id: m for m in temporal_selection}
                    
                    # 3. Combine and Truncate
                    # Priority: Temporal > Semantic
                    # Limit: 13
                    
                    final_context_map = {}
                    
                    # Add all temporal memories first
                    for m_id, m_item in temporal_memories_map.items():
                        final_context_map[m_id] = m_item
                        
                    # Calculate remaining slots
                    remaining_slots = 13 - len(final_context_map)
                    
                    if remaining_slots > 0:
                        # Filter semantic memories that are not already in final_context (temporal)
                        semantic_only = [m for m in semantic_memories_map.values() if m.id not in final_context_map]
                        
                        # Randomly select to fill slots
                        count_to_add = min(remaining_slots, len(semantic_only))
                        if count_to_add > 0:
                            selected_semantic = random.sample(semantic_only, count_to_add)
                            for m in selected_semantic:
                                final_context_map[m.id] = m
                                
                    retrieved_context = [m.to_dict() for m in final_context_map.values()]
                
                if not retrieved_context:
                    print(f"Skipping {sample_id}: empty context_memory.")
                    continue
                
                item["context_memory"] = retrieved_context
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                f_out.flush()
                
                count += 1
                print(f"Processed {sample_id}. Context memory size: {len(retrieved_context)}")
            except Exception as e:
                print(f"Error processing item {sample_id}: {e}")
                continue

def split_dataset(source_file: str, train_file: str, test_file: str, split_ratio: float = 0.9):
    """Split dataset into train and test sets."""
    print(f"Splitting data from {source_file}...")
    if not os.path.exists(source_file):
        print(f"Source file {source_file} not found.")
        return

    data = []
    with open(source_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    if not data:
        print("No data found to split.")
        return

    random.shuffle(data)
    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    
    print(f"Writing {len(train_data)} items to {train_file}")
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Writing {len(test_data)} items to {test_file}")
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # Configuration
    SOURCE_FILE = "/home/guozl/project/MemRL/Memory-CookBook/scripts/processed_locomo.json"
    TARGET_FILE = "/home/guozl/project/MemRL/Memory-CookBook/scripts/temp.jsonl"
    K = 9
    MAX_ITEMS = 567 # Process first 10 for testing as per user request "N items"
    SPLIT_DATA = True # Switch to enable data splitting
    
    process_data(SOURCE_FILE, TARGET_FILE, K, MAX_ITEMS)
    
    if SPLIT_DATA:
        TRAIN_FILE = "../datas/train.jsonl"
        TEST_FILE = "../datas/test.jsonl"
        # Resolve relative paths relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_path = os.path.join(script_dir, TRAIN_FILE)
        test_path = os.path.join(script_dir, TEST_FILE)
        
        split_dataset(TARGET_FILE, train_path, test_path)

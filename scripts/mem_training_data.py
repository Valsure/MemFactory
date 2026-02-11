
import json
import os
import sys
import math
import random
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Optional

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

# Configuration for Memory Allocation
TOTAL_CONTEXT_SIZE = 16
TEMPORAL_SLOTS = 6 

def process_single_item(item: Dict) -> Optional[Dict]:
    """
    Process a single data item to retrieve context memory.
    This function is designed to be run in a parallel worker.
    """
    sample_id = item.get("sample_id", "unknown")
    
    try:
        # Initialize components within the process (safe for multiprocessing)
        llm = get_llm_client()
        store = get_memory_store()
        
        # Use mock store as per original requirement
        store.use_mock = True
        store.from_list(item["M"])
        
        facts = item.get("f", [])
        if not facts:
            # print(f"Skipping {sample_id}: No facts.")
            return None
            
        recent_facts = facts[-4:]
        extraction_prompt = construct_extraction_prompt(recent_facts)
        extraction_output = llm.chat("You are a memory extraction expert.", extraction_prompt)
        ext_json = parse_json_from_text(extraction_output)
        
        extracted_memories = []
        if ext_json and "memory_list" in ext_json:
            extracted_memories = ext_json["memory_list"]
        
        # If no extracted memories, we might skip or just use temporal. 
        # Original logic implies skipping if extraction fails or returns nothing relevant to search.
        if not extracted_memories:
            return None

        # 1. Semantic Search
        semantic_memories_map = {} # id -> MemoryItem
        
        for mem_dict in extracted_memories:
            query_text = f"{mem_dict.get('key', '')}: {mem_dict.get('value', '')}"
            # top_k=3 for each candidate
            results = store.search_similar(query_text, top_k=3)
            for mem_item, score in results:
                semantic_memories_map[mem_item.id] = mem_item
                
        # 2. Temporal Search
        # Allocation Strategy: 
        # We reserve TEMPORAL_SLOTS (6) for recent memories to ensure conversation continuity.
        all_memories = store.get_all()
        
        # Look back slightly more than the quota (12) to allow for some sampling or just take the tail
        temporal_candidates = all_memories[-12:]
        
        temporal_selection = []
        if temporal_candidates:
            k_temporal = min(TEMPORAL_SLOTS, len(temporal_candidates))
            temporal_selection = random.sample(temporal_candidates, k_temporal)
            
        temporal_memories_map = {m.id: m for m in temporal_selection}
        
        # 3. Combine and Truncate
        final_context_map = {}
        
        # Priority 1: Temporal Memories (Recent Context)
        for m_id, m_item in temporal_memories_map.items():
            final_context_map[m_id] = m_item
            
        # Priority 2: Semantic Memories (Relevant Context)
        remaining_slots = TOTAL_CONTEXT_SIZE - len(final_context_map)
        
        if remaining_slots > 0:
            # Filter semantic memories that are not already in final_context
            semantic_only = [m for m in semantic_memories_map.values() if m.id not in final_context_map]
            
            # Fill remaining slots with semantic memories
            count_to_add = min(remaining_slots, len(semantic_only))
            if count_to_add > 0:
                selected_semantic = random.sample(semantic_only, count_to_add)
                for m in selected_semantic:
                    final_context_map[m.id] = m
                    
        retrieved_context = [m.to_dict() for m in final_context_map.values()]
        
        if not retrieved_context:
            return None
        
        item["context_memory"] = retrieved_context
        return item

    except Exception as e:
        # In parallel processing, it's better not to print too much to stdout, 
        # or use a lock for printing. Ignoring errors for now to keep it clean.
        # print(f"Error processing item {sample_id}: {e}")
        return None

def process_data(source_path: str, target_path: str, k: int = 8, max_items: int = 512, num_workers: int = 1):
    print(f"Processing data from {source_path} to {target_path}")
    print(f"Parallel Workers: {num_workers}")
    print(f"Context Window Config: Total={TOTAL_CONTEXT_SIZE}, Temporal={TEMPORAL_SLOTS}")
    
    # Load source data
    if not os.path.exists(source_path):
        print(f"Source file {source_path} does not exist.")
        return

    with open(source_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    # Check already processed items
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
    
    # Filter items to process
    items_to_process = []
    for item in source_data:
        sample_id = item.get("sample_id", "")
        trigger_id = item.get("trigger_id", "")
        q = item.get("q", "")
        unique_id = f"{sample_id}||{trigger_id}||{q}"
        
        if unique_id not in processed_ids:
            items_to_process.append(item)
            
    # Apply max_items limit
    if max_items > 0:
        items_to_process = items_to_process[:max_items]
        
    print(f"Items to process: {len(items_to_process)}")
    
    if not items_to_process:
        print("No items to process.")
        return

    # Processing
    with open(target_path, 'a', encoding='utf-8') as f_out:
        if num_workers > 1:
            with Pool(num_workers) as pool:
                # Use imap to iterate over results as they complete (in order)
                # chunksize can be tuned, default is 1.
                for result in tqdm(pool.imap(process_single_item, items_to_process), total=len(items_to_process), desc="Processing (Parallel)"):
                    if result:
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f_out.flush()
        else:
            for item in tqdm(items_to_process, desc="Processing (Single Process)"):
                result = process_single_item(item)
                if result:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()

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
    parser = argparse.ArgumentParser(description="Process memory training data.")
    parser.add_argument("--source", default="/home/guozl/project/MemRL/Memory-CookBook/scripts/processed_locomo.json", help="Source JSON file")
    parser.add_argument("--target", default="/home/guozl/project/MemRL/Memory-CookBook/scripts/temp.jsonl", help="Target JSONL file")
    parser.add_argument("--k", type=int, default=9, help="K (unused in current logic, but kept for compat)")
    parser.add_argument("--limit", type=int, default=567, help="Max items to process (0 for all)")
    parser.add_argument("--workers", type=int, default=12, help="Number of parallel workers")
    parser.add_argument("--split", action="store_true", default=True, help="Split data into train/test")
    
    args = parser.parse_args()
    
    # Run Processing
    process_data(args.source, args.target, args.k, args.limit, args.workers)
    
    # Run Split
    if args.split:
        TRAIN_FILE = "../datas/train.jsonl"
        TEST_FILE = "../datas/test.jsonl"
        # Resolve relative paths relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_path = os.path.join(script_dir, TRAIN_FILE)
        test_path = os.path.join(script_dir, TEST_FILE)
        
        split_dataset(args.target, train_path, test_path)

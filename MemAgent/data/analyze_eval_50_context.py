import json
from transformers import AutoTokenizer
from collections import defaultdict
import os

# Configuration
JSON_FILE = "/home/guozl/project/MemRL/Memory-CookBook/MemAgent/data/eval_50.json"
MODEL_PATH = "/home/models/Qwen2.5-3B"
BASE_THRESHOLD = 8000
STEP_SIZE = 2000

def analyze_lengths():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Loading json file from {JSON_FILE}...")
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading json file: {e}")
        return

    print(f"Analyzing {len(data)} items...")
    
    # Statistics storage
    stats = defaultdict(int)
    
    max_len = 0
    min_len = float('inf')
    total_len = 0
    lengths = []

    for index, item in enumerate(data):
        context = item.get('context', "")
        if not isinstance(context, str):
            context = str(context) if context is not None else ""
        
        # Tokenize and get length
        tokens = tokenizer(context, add_special_tokens=False)['input_ids']
        length = len(tokens)
        lengths.append(length)
        
        max_len = max(max_len, length)
        min_len = min(min_len, length)
        total_len += length

        if length < BASE_THRESHOLD:
            stats["< 8k"] += 1
        else:
            # Calculate bucket index for >= 8k
            bucket_idx = (length - BASE_THRESHOLD) // STEP_SIZE
            start = BASE_THRESHOLD + bucket_idx * STEP_SIZE
            end = start + STEP_SIZE
            bucket_key = f"{start//1000}k - {end//1000}k"
            stats[bucket_key] += 1
            
        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1} items...", end='\r')

    print(f"\n\nAnalysis Complete.")
    print(f"Total items: {len(data)}")
    if len(data) > 0:
        print(f"Min length: {min_len}")
        print(f"Max length: {max_len}")
        print(f"Avg length: {total_len / len(data):.2f}")
    
    print("\nLength Distribution:")
    
    # Sort keys for display
    def sort_key(k):
        if k == "< 8k":
            return -1
        try:
            # Extract the starting number: "8k - 10k" -> 8
            return int(k.split('k')[0])
        except:
            return 999999

    sorted_keys = sorted(stats.keys(), key=sort_key)
    
    for key in sorted_keys:
        count = stats[key]
        percentage = (count / len(data)) * 100
        print(f"{key}: {count} ({percentage:.2f}%)")

if __name__ == "__main__":
    analyze_lengths()

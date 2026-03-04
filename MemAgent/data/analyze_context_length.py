import pandas as pd
from transformers import AutoTokenizer
import math
from collections import defaultdict

# Configuration
PARQUET_FILE = "/home/guozl/project/MemRL/Memory-CookBook/MemAgent/data/train_1k.parquet"
MODEL_PATH = "/home/models/Qwen2.5-3B"
BASE_THRESHOLD = 24000
STEP_SIZE = 4000

def analyze_lengths():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Loading parquet file from {PARQUET_FILE}...")
    try:
        df = pd.read_parquet(PARQUET_FILE)
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return

    if 'context' not in df.columns:
        print("Error: 'context' column not found in the parquet file.")
        return

    print(f"Analyzing {len(df)} rows...")
    
    # Statistics storage
    stats = defaultdict(int)
    # Ensure specific keys exist for reporting even if 0
    stats["< 24k"] = 0
    
    max_len = 0
    min_len = float('inf')
    total_len = 0

    for index, row in df.iterrows():
        context = row['context']
        if not isinstance(context, str):
            context = str(context) if context is not None else ""
        
        # Tokenize and get length
        # Using simple encoding without special tokens might be slightly different depending on usage, 
        # but usually len(tokenizer.encode(text)) is standard.
        # To be faster, we can use tokenizer(text, add_special_tokens=False)['input_ids']
        tokens = tokenizer(context, add_special_tokens=False)['input_ids']
        length = len(tokens)
        
        max_len = max(max_len, length)
        min_len = min(min_len, length)
        total_len += length

        if length < BASE_THRESHOLD:
            stats["< 24k"] += 1
        else:
            # Calculate bucket index for >= 24k
            # 24000-27999 -> bucket 0 (24k-28k)
            # 28000-31999 -> bucket 1 (28k-32k)
            bucket_idx = (length - BASE_THRESHOLD) // STEP_SIZE
            start = BASE_THRESHOLD + bucket_idx * STEP_SIZE
            end = start + STEP_SIZE
            bucket_key = f"{start//1000}k - {end//1000}k"
            stats[bucket_key] += 1
            
        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1} rows...", end='\r')

    print(f"\n\nAnalysis Complete.")
    print(f"Total rows: {len(df)}")
    print(f"Min length: {min_len}")
    print(f"Max length: {max_len}")
    print(f"Avg length: {total_len / len(df):.2f}")
    print("\nLength Distribution:")
    
    # Sort keys for display
    # We need a custom sort key
    def sort_key(k):
        if k == "< 24k":
            return -1
        # Extract the starting number: "24k - 28k" -> 24
        try:
            return int(k.split('k')[0])
        except:
            return 999999

    sorted_keys = sorted(stats.keys(), key=sort_key)
    
    for key in sorted_keys:
        count = stats[key]
        percentage = (count / len(df)) * 100
        print(f"{key}: {count} ({percentage:.2f}%)")

if __name__ == "__main__":
    analyze_lengths()

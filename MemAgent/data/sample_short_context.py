import pandas as pd
from transformers import AutoTokenizer
import random
import os

# Configuration
INPUT_FILE = "/home/guozl/project/MemRL/Memory-CookBook/MemAgent/data/hotpotqa_train.parquet"
OUTPUT_FILE = "/home/guozl/project/MemRL/Memory-CookBook/MemAgent/data/hotpotqa_train_short_1k.parquet"
MODEL_PATH = "/home/models/Qwen2.5-3B"

def sample_short_context():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Loading parquet file from {INPUT_FILE}...")
    try:
        df = pd.read_parquet(INPUT_FILE)
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return

    if 'context' not in df.columns:
        print("Error: 'context' column not found in the parquet file.")
        return

    print(f"Total rows loaded: {len(df)}")
    
    # Calculate lengths
    print("Calculating context lengths...")
    lengths = []
    for index, row in df.iterrows():
        context = row['context']
        if not isinstance(context, str):
            context = str(context) if context is not None else ""
        
        # Tokenize and get length
        tokens = tokenizer(context, add_special_tokens=False)['input_ids']
        lengths.append(len(tokens))
        
        if (index + 1) % 1000 == 0:
            print(f"Processed {index + 1} rows...", end='\r')
            
    df['context_length'] = lengths
    print(f"\nLength calculation complete.")

    # Sort by length and take shortest 2000
    print("Sorting and selecting shortest 2000...")
    df_sorted = df.sort_values(by='context_length', ascending=True)
    shortest_2k = df_sorted.head(2000)
    
    print(f"Shortest 2000 context length range: {shortest_2k['context_length'].min()} - {shortest_2k['context_length'].max()}")

    # Randomly sample 1000 from the shortest 2000
    print("Randomly sampling 1000 from the shortest 2000...")
    final_sample = shortest_2k.sample(n=1000, random_state=42) # Set random_state for reproducibility

    # Save to new parquet file
    print(f"Saving to {OUTPUT_FILE}...")
    # Optionally drop the helper column 'context_length' if you want the schema to be identical to source
    # final_sample = final_sample.drop(columns=['context_length'])
    final_sample.to_parquet(OUTPUT_FILE)
    
    print("Done!")
    print(f"Final sample size: {len(final_sample)}")
    print(f"Final sample context length range: {final_sample['context_length'].min()} - {final_sample['context_length'].max()}")

if __name__ == "__main__":
    sample_short_context()

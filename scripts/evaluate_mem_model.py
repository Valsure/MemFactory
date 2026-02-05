
import os
import sys
import json
import argparse
import torch
import gc
import time
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Memory-CookBook
rl_dir = os.path.join(project_root, "RL")
sys.path.append(project_root)
sys.path.append(rl_dir)

# Import from RL and src
try:
    import mem_utils
    from src.common import MemoryItem
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

class ModelInference:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        print(f"Loading model from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            "device_map": "auto",
            "trust_remote_code": True
        }
        
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 2048) -> str:
        return self.generate_batch([prompt], max_new_tokens)[0]

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 2048) -> List[str]:
        if not prompts:
            return []
            
        messages_list = [[{"role": "user", "content": p}] for p in prompts]
        texts = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages_list]
        
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, padding_side="left").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0, # Deterministic for eval
                do_sample=False
            )
            
        # Decode only the new tokens
        input_len = inputs.input_ids.shape[1]
        generated_texts = []
        for output in outputs:
            generated_ids = output[input_len:]
            generated_texts.append(self.tokenizer.decode(generated_ids, skip_special_tokens=True))
            
        return generated_texts

    def extract_memories(self, fact: List[Dict]) -> str:
        prompt = mem_utils.construct_extraction_prompt(fact)
        return self.generate(prompt)

    def update_memories(self, context_memory: List[Dict], extraction_output: str) -> str:
        prompt = mem_utils.construct_update_prompt(context_memory, extraction_output)
        return self.generate(prompt)

def load_dataset(file_path: str, max_samples: int = None) -> List[Dict]:
    data = []
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found.")
        return []

    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                if isinstance(raw_data, list):
                    data = raw_data
                else:
                    print("JSON content is not a list.")
        elif file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

    processed_data = []
    for item in data:
        # Normalize keys
        processed_item = {
            "memory": item.get("M") or item.get("memory") or [],
            "fact": item.get("f") or item.get("fact") or [],
            "query": item.get("q") or item.get("query") or "",
            "answer": item.get("a") or item.get("answer") or "",
            "context_memory": item.get("context_memory") # Optional
        }
        
        # Validate essential fields
        if not processed_item["fact"] or not processed_item["query"]:
            continue
            
        processed_data.append(processed_item)
    
    if max_samples and max_samples > 0:
        processed_data = processed_data[:max_samples]
        
    print(f"Loaded {len(processed_data)} samples from {file_path}")
    return processed_data

def evaluate_benchmark(model: ModelInference, benchmark_name: str, data_path: str, evaluator: mem_utils.MemoryEvaluator, batch_size: int = 4, max_samples: int = None) -> Dict:
    dataset = load_dataset(data_path, max_samples)
    if not dataset:
        return {"accuracy": 0.0, "count": 0, "correct": 0}

    correct_count = 0
    total_count = 0
    
    print(f"Starting evaluation for {benchmark_name} ({len(dataset)} samples)...")
    
    # Create batches
    batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]
    
    # Using tqdm with more details
    pbar = tqdm(batches, desc=f"Evaluating {benchmark_name}")
    
    for batch in pbar:
        try:
            # 1. Extraction Batch
            facts = [s["fact"] for s in batch]
            ext_prompts = [mem_utils.construct_extraction_prompt(f) for f in facts]
            
            ext_outputs = model.generate_batch(ext_prompts)
            
            # 2. Context Retrieval & Update Prompt Preparation
            upd_prompts = []
            
            # Note: Retrieval is sequential because it depends on evaluator state or needs sequential calls
            # However, for prompt construction we just need the context memory
            
            for i, sample in enumerate(batch):
                memory = sample["memory"]
                context_memory = sample["context_memory"]
                extraction_output = ext_outputs[i]
                
                # Retrieve if needed
                if context_memory is None:
                    # Initialize store with current memory state (Mock Store reset is fast)
                    evaluator.reset_memory(memory)
                    
                    # Parse extraction to get search queries
                    candidates = []
                    ext_json = mem_utils.parse_json_from_text(extraction_output)
                    if ext_json and "memory_list" in ext_json:
                        for m in ext_json["memory_list"]:
                            if isinstance(m, dict):
                                candidates.append(m)
                    
                    # Retrieve relevant existing memories
                    retrieved_items = []
                    retrieved_ids = set()
                    
                    # Search using candidates
                    for cand in candidates:
                        # Use key and value for search query
                        query_text = f"{cand.get('key', '')} {cand.get('value', '')}"
                        results = evaluator.store.search_similar(query_text, top_k=3)
                        
                        for item, _ in results:
                            # Stop if we reached the limit
                            if len(retrieved_items) >= 12:
                                break
                            
                            # Skip if already added
                            if item.id in retrieved_ids:
                                continue
                                
                            retrieved_items.append(item)
                            retrieved_ids.add(item.id)
                        
                        if len(retrieved_items) >= 12:
                            break
                    
                    # Convert back to dicts
                    context_memory = [item.to_dict() for item in retrieved_items]
                    # Update the sample so we can use it in evaluation step
                    sample["context_memory"] = context_memory
                
                # Construct Update Prompt
                upd_prompts.append(mem_utils.construct_update_prompt(context_memory, extraction_output))
            
            # 3. Update Batch
            upd_outputs = model.generate_batch(upd_prompts)
            
            # 4. Evaluation (Sequential)
            for i, sample in enumerate(batch):
                memory = sample["memory"]
                fact = sample["fact"]
                query = sample["query"]
                answer = sample["answer"]
                context_memory = sample["context_memory"] # This is now populated
                extraction_output = ext_outputs[i]
                update_plan_output = upd_outputs[i]
                
                score = evaluator.evaluate(
                    memory=memory,
                    fact=fact,
                    query=query,
                    answer=answer,
                    context_memory=context_memory,
                    extraction_output=extraction_output,
                    update_plan_output=update_plan_output
                )
                
                if score >= 1.0:
                    correct_count += 1
                total_count += 1
            
            # Update progress bar description with current accuracy
            current_acc = correct_count / total_count if total_count > 0 else 0.0
            pbar.set_postfix({"Acc": f"{current_acc:.2%}"})

        except Exception as e:
            print(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            continue

    accuracy = correct_count / total_count if total_count > 0 else 0.0
    return {
        "accuracy": accuracy,
        "count": total_count,
        "correct": correct_count
    }

def main():
    # --- Configuration ---
    # TODO: 这里正在测试，如果现在的实现没有问题的话，后面的修改应该包括（1）测试用时（2）批量处理吃满显存。
    models_config = {
        "Qwen3-1.7B": "/home/models/Qwen3-1.7B",
        "Qwen3-4B": "/home/models/qwen3-4b",
        "4B-100step": "../output/mem_grpo/checkpoint_100",
        "4B-200step": "../output/mem_grpo/checkpoint_200",
        "4B-300step": "../output/mem_grpo/checkpoint_300",
        "4B-400step": "../output/mem_grpo/checkpoint_400",
    }
    
    # Define your benchmarks here
    benchmarks_config = {
        "TrainData": "../scripts/training_data_with_context.jsonl",
        "TestData": "../datas/test.jsonl",
        # "LocomoNo10": os.path.join(os.path.dirname(current_dir), "scripts", "processed_locomo.json"),
    }
    
    # Argument parser for CLI overrides
    parser = argparse.ArgumentParser(description="Evaluate Memory Models")
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--data_path", type=str, help="Path to a specific benchmark file")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate per benchmark")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    args = parser.parse_args()

    # Override config if CLI args provided
    if args.model_name and args.model_path:
        models_config = {args.model_name: args.model_path}
    if args.data_path:
        benchmarks_config = {"CLI_Benchmark": args.data_path}

    if not models_config:
        print("No models configured. Please edit the script or provide --model_name and --model_path.")
        return

    if not benchmarks_config:
        print("No benchmarks configured. Please edit the script or provide --data_path.")
        return

    # Initialize Evaluator (Shared environment)
    print("Initializing MemoryEvaluator (Environment)...")
    try:
        evaluator = mem_utils.MemoryEvaluator()
    except Exception as e:
        print(f"Failed to initialize MemoryEvaluator: {e}")
        print("Ensure LLMClient and EmbeddingClient are configured correctly.")
        return

    # --- Main Loop ---
    results_summary = {}

    for model_name, model_path in models_config.items():
        print(f"\n{'='*20}\nEvaluating Model: {model_name}\n{'='*20}")
        
        # Load Model
        try:
            model_inference = ModelInference(model_path)
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            continue

        model_results = {}
        
        for bench_name, bench_path in benchmarks_config.items():
            print(f"\n--- Benchmark: {bench_name} ---")
            # Pass max_samples and batch_size
            res = evaluate_benchmark(
                model_inference, 
                bench_name, 
                bench_path, 
                evaluator, 
                batch_size=args.batch_size, 
                max_samples=args.max_samples
            )
            print(f"Result: {res}")
            model_results[bench_name] = res
        
        results_summary[model_name] = model_results

        # Cleanup Model to free GPU
        del model_inference
        gc.collect()
        torch.cuda.empty_cache()

    # --- Final Report ---
    print("\n\n" + "="*40)
    print("FINAL EVALUATION REPORT")
    print("="*40)
    for model_name, res in results_summary.items():
        print(f"\nModel: {model_name}")
        for bench_name, metrics in res.items():
            print(f"  Benchmark: {bench_name}")
            print(f"    Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['count']})")
    print("="*40)

if __name__ == "__main__":
    main()

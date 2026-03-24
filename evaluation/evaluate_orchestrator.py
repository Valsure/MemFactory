import os
import sys
import subprocess
import argparse
import json
from multiprocessing import Process
from typing import List, Dict

def is_task_completed(task: Dict[str, str]) -> bool:
    """Check if the task has already been successfully completed."""
    out_file = task['out_file']
    if not os.path.exists(out_file):
        return False
    
    try:
        with open(out_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        summary = data.get('summary')
        if not summary:
            return False
            
        # Check model and dataset match
        if summary.get('model') != task['model']:
            return False
        if summary.get('dataset') != task['dataset']:
            return False
            
        # Check current_accuracy is valid number
        acc = summary.get('current_accuracy')
        if not isinstance(acc, (int, float)):
            return False
            
        # Check processed == total
        processed = summary.get('processed')
        total = summary.get('total')
        if processed is None or total is None or processed != total:
            return False
            
        return True
    except Exception:
        return False

def run_tasks_on_gpu(gpu_id: str, tasks: List[Dict[str, str]]):
    """
    Worker process function to run a queue of tasks sequentially on a specific GPU.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    for t in tasks:
        print(f"\n[GPU {gpu_id}] Starting task:")
        print(f"  Model:   {t['model']}")
        print(f"  Dataset: {t['dataset']}")
        
        worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate_worker.py")
        cmd = [
            sys.executable, worker_script,
            "--model_path", t['model'],
            "--dataset_path", t['dataset'],
            "--output_file", t['out_file']
        ]
        
        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"[GPU {gpu_id}] Finished task: {t['out_file']}")
        except subprocess.CalledProcessError as e:
            print(f"[GPU {gpu_id}] Task failed with error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Orchestrator for Evaluation")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7", help="Comma-separated list of GPU IDs to use (e.g. '0,1,2,3')")
    # parser.add_argument("--base_model", type=str, required=True, help="Path or name of the base model")
    args = parser.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    num_gpus = len(gpus)

    # Models to evaluate (Base model + 3 Checkpoints)
    models = [
        "/home/models/Qwen3-1.7B",
        "/home/models/Qwen3-4B-Instruct",
        "/home/guozl/project/MemRL/Memory-CookBook/output/mem_factory_qwen3_1.7B/checkpoint_250",
        "/home/guozl/project/MemRL/Memory-CookBook/output/mem_factory_qwen3_1.7B-noshuffle/checkpoint_250",
        "/home/guozl/project/MemRL/Memory-CookBook/output/mem_factory_qwen3_4bI_noshuffle/checkpoint_250"
    ]

    # Datasets to evaluate
    datasets = [
        "/home/guozl/project/guozl_project/MemAgentDatas/eval_50.json",
        "/home/guozl/project/guozl_project/MemAgentDatas/eval_100.json",
        "/home/guozl/project/guozl_project/MemAgentDatas/eval_fwe_16384.json",
        # "/home/guozl/project/guozl_project/MemAgentDatas/eval_fwe_32768.json"
    ]

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")
    os.makedirs(output_dir, exist_ok=True)

    # Generate all tasks
    all_tasks = []
    tasks = []
    for m in models:
        for d in datasets:
            # Generate a readable name for the model
            if "checkpoint" in m:
                parts = m.rstrip("/").split("/")
                m_name = f"{parts[-2]}_{parts[-1]}"
            else: # 也只保留最后两项
                parts = m.rstrip("/").split("/")
                m_name = f"{parts[-2]}_{parts[-1]}"
                
            d_name = os.path.basename(d).replace(".json", "")
            out_file = os.path.join(output_dir, f"{m_name}_{d_name}.json")
            
            task_info = {
                "model": m,
                "dataset": d,
                "out_file": out_file
            }
            all_tasks.append(task_info)
            
            if is_task_completed(task_info):
                print(f"Skipping completed task: {out_file}")
            else:
                tasks.append(task_info)

    # Distribute tasks across GPUs
    gpu_queues = {g: [] for g in gpus}
    for i, t in enumerate(tasks):
        gpu = gpus[i % num_gpus]
        gpu_queues[gpu].append(t)

    print(f"Total tasks: {len(tasks)}")
    print(f"Total GPUs: {num_gpus}")
    for gpu, q in gpu_queues.items():
        print(f"  GPU {gpu}: {len(q)} tasks")

    # Spawn processes
    processes = []
    for gpu, q in gpu_queues.items():
        if not q:
            continue
        p = Process(target=run_tasks_on_gpu, args=(gpu, q))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("\nAll evaluations finished! Results are saved in the 'eval_results' directory.")
    
    # Optional: summarize results
    print("\n--- Summary ---")
    for t in all_tasks:
        if os.path.exists(t['out_file']):
            try:
                with open(t['out_file'], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    acc = data.get('summary', {}).get('current_accuracy', 0.0)
                    print(f"{t['out_file']}: Accuracy = {acc:.4f}")
            except Exception as e:
                print(f"Failed to read {t['out_file']}: {e}")

if __name__ == "__main__":
    main()

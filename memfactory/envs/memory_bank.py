import json
import os
import torch
import numpy as np
from typing import List, Dict, Any
from ..common.registry import ENV_REGISTRY
from .base import BaseEnv
from ..common.utils import parse_json_from_text, LLMClient

# Import src.common dependencies
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.common import MemoryItem, generate_id, get_memory_store
    from src.common import format_conversation, ConversationMessage
except ImportError:
    pass # Assume setup in other files

QA_PROMPT = """Based on the following memory information, answer the user's question. Please answer directly and accurately. If the memory contains explicit information, use it.

Context Memories:
{context}

User Question: {question}
Answer:"""

JUDGE_PROMPT = """Please judge whether the predicted answer is correct based on the standard answer.

Question: {question}
Standard Answer: {answer}
Predicted Answer: {prediction}

Is the predicted answer consistent with the standard answer? Please output only "True" or "False".
"""

@ENV_REGISTRY.register("memory_bank")
class MemoryBankEnv(BaseEnv):
    def __init__(self, data_path, tokenizer, **kwargs):
        super().__init__(data_path, tokenizer)
        self.data = []
        self._load_data()
        
        # Initialize Evaluator Components
        # We create a new store instance or reuse global?
        # Ideally, for training we want a transient/mock store.
        self.store = get_memory_store() 
        # Make sure we use mock store for training to avoid disk IO and persistence
        self.store.use_mock = True 
        
        self.llm_client = LLMClient()

    def _load_data(self):
        raw_data = []
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
            except json.JSONDecodeError:
                with open(self.data_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                raw_data.append(json.loads(line))
                            except: pass
        
        for item in raw_data:
            # Helper to safely get list or default
            def get_list(d, keys):
                for k in keys:
                    if d.get(k) is not None: return d[k]
                return []
            
            def get_str(d, keys):
                for k in keys:
                    if d.get(k) is not None: return d[k]
                return ""

            processed_item = {
                "memory": get_list(item, ["M", "memory"]),
                "fact": get_list(item, ["f", "fact"]),
                "query": get_str(item, ["q", "query"]),
                "answer": get_str(item, ["a", "answer"]),
                "context_memory": get_list(item, ["context_memory"])
            }
            self.data.append(processed_item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def prepare_memory_lists(self, context_memory, extraction_output):
        # Need to reconstruct ID mapping for evaluation
        # Copied logic from mem_utils
        context_memory_objs = [ MemoryItem.from_dict(mem) for mem in context_memory ]
        id_counter = 1
        id_map = {}
        
        for mem in context_memory_objs:
            temp_id = id_counter
            id_counter += 1
            id_map[temp_id] = ("context", mem)
            
        ext_json = parse_json_from_text(extraction_output)
        if ext_json and "memory_list" in ext_json and isinstance(ext_json["memory_list"], list):
            for item in ext_json["memory_list"]:
                if not isinstance(item, dict): continue
                temp_id = id_counter
                id_counter += 1
                id_map[temp_id] = ("candidate", item)
        
        return id_map

    def compute_reward(self, predictions: Dict[str, List[str]], ground_truths: Dict[str, Any], num_generations: int, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute rewards for Extraction and Update.
        predictions: {'extraction': [texts], 'update': [texts]}
        ground_truths: Batch data dict
        """
        ext_texts = predictions['extraction']
        upd_texts = predictions['update']
        
        bs = len(ground_truths['fact'])
        
        all_rewards_ext = []
        all_rewards_upd = []
        
        # Parallelize this if slow
        for i in range(bs):
            memory = ground_truths['memory'][i]
            # fact = ground_truths['fact'][i] # Not needed for eval logic here
            query = ground_truths['query'][i]
            answer = ground_truths['answer'][i]
            ctx_mem = ground_truths['context_memory'][i]
            
            for j in range(num_generations):
                global_idx = i * num_generations + j
                ext_out = ext_texts[global_idx]
                upd_out = upd_texts[global_idx]
                
                # Logic from MemoryEvaluator.evaluate
                
                # 1. Extraction Format Reward
                ext_json = parse_json_from_text(ext_out)
                ext_reward = 0.5 if (ext_json != {} and "memory_list" in ext_json) else 0.0
                
                # 2. Update Format Reward
                upd_json = parse_json_from_text(upd_out)
                upd_reward = 0.5 if (upd_json != {} and "operations" in upd_json) else 0.0
                
                accuracy_reward = 0.0
                
                # 3. Accuracy Reward (QA)
                # Only if formats are valid
                if ext_reward > 0 and upd_reward > 0:
                    try:
                        # Reset Store
                        if self.store.use_mock:
                             self.store.from_list(memory)
                        
                        # Apply Update
                        # Need to parse update plan and apply to store
                        # Logic copied from mem_utils.apply_update_plan
                        
                        id_map = self.prepare_memory_lists(ctx_mem, ext_out)
                        update_err = False
                        
                        if "operations" in upd_json:
                            for op in upd_json["operations"]:
                                temp_id = op.get("id")
                                action = op.get("op", "NONE").upper()
                                if temp_id not in id_map: 
                                    update_err = True; break
                                origin_type, origin_obj = id_map[temp_id]
                                
                                if origin_type == "context":
                                    if action == "DEL": self.store.delete(origin_obj.id)
                                elif origin_type == "candidate":
                                    if action in ["ADD", "UPDATE"]:
                                        key = op.get("key", origin_obj.get("key"))
                                        value = op.get("value", origin_obj.get("value"))
                                        new_mem = MemoryItem(id=generate_id(), key=key, value=value)
                                        self.store.save(new_mem)
                        else:
                            update_err = True
                            
                        if not update_err:
                            # Retrieve
                            results = self.store.search_similar(query, top_k=3) # Use small k for speed
                            retrieved_docs = [m for m, s in results]
                            context_str = "\n".join([f"- {m.key}: {m.value}" for m in retrieved_docs])
                            
                            # QA
                            qa_prompt = QA_PROMPT.format(context=context_str, question=query)
                            pred_answer = self.llm_client.chat("You are a helpful assistant.", qa_prompt)
                            
                            # Judge
                            judge_prompt = JUDGE_PROMPT.format(question=query, answer=answer, prediction=pred_answer)
                            judge_result = self.llm_client.chat("You are an impartial judge.", judge_prompt)
                            
                            if "True" in judge_result:
                                accuracy_reward = 1.0
                    except Exception as e:
                        # print(f"Eval Error: {e}")
                        pass
                
                # Combine Rewards
                # We add accuracy to both? Or just update?
                # Reference code logic implies separate rewards.
                # Let's add accuracy to both if formats are valid.
                
                final_ext = ext_reward
                final_upd = upd_reward
                
                if accuracy_reward > 0:
                    final_ext += accuracy_reward
                    final_upd += accuracy_reward
                    
                all_rewards_ext.append(final_ext)
                all_rewards_upd.append(final_upd)
                
        return {
            "extraction": torch.tensor(all_rewards_ext, dtype=torch.float32, device="cuda"), # Assuming cuda
            "update": torch.tensor(all_rewards_upd, dtype=torch.float32, device="cuda")
        }

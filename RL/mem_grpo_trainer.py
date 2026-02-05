
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional, Union, Any, Dict, Tuple
from copy import deepcopy
import json
import os
import swanlab
import random
import sys
from tqdm import tqdm

# Import mem_utils and src.common
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import mem_utils
# Ensure we can import src.common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import MemoryItem, generate_id

@dataclass
class Samples:
    prompt_response_ids: torch.Tensor
    response_ids: torch.Tensor
    prompt: Any # Can be string or list of strings
    answer: Any
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    response_length: int
    step_type: str # 'extraction' or 'update'
    rewards: Optional[torch.Tensor] = None

@dataclass
class MemGRPOArguments:
    output_dir: str = "./output"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr: float = 5e-7
    save_steps: int = 500
    epoch: int = 1
    num_generations: int = 4 # Group size
    max_prompt_length: int = 4096
    max_generate_length: int = 2048
    beta: float = 0.1 # KL penalty
    clip_eps: float = 0.2
    gradient_accumulation_steps: int = 1
    num_iterations: int = 1
    batch_size: int = 1
    train_extraction: bool = True
    train_update: bool = True
    gradient_checkpointing: bool = True

class MemoryDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = []
        raw_data = []
        if os.path.exists(data_path):
            try:
                # Try loading as a JSON array first
                with open(data_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
            except json.JSONDecodeError:
                # Fallback to JSONL
                with open(data_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                raw_data.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
        else:
            print(f"Warning: {data_path} not found.")
            assert False, f"{data_path} not found."

        # Process and Sanitize
        for item in raw_data:
            # Helper to safely get list or default
            def get_list(d, keys):
                for k in keys:
                    if d.get(k) is not None:
                        return d[k]
                assert False, f"Memory field not found in item: {item}"
                return []
            
            # Helper to safely get str or default
            def get_str(d, keys):
                for k in keys:
                    if d.get(k) is not None:
                        return d[k]
                assert False, f"Query field not found in item: {item}"
                return ""

            processed_item = {
                "memory": get_list(item, ["M", "memory"]),
                "fact": get_list(item, ["f", "fact"]),
                "query": get_str(item, ["q", "query"]),
                "answer": get_str(item, ["a", "answer"]),
                "context_memory": get_list(item, ["context_memory"])
            }
            self.data.append(processed_item)
        
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class MemGRPOTrainer:
    def __init__(self,
                 model,
                 args: MemGRPOArguments,
                 train_dataset: Dataset,
                 tokenizer,
                 ref_model=None):
        self.args = args
        self.model = model.to(self.args.device)
        
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        self.tokenizer = self.get_tokenizer(tokenizer)
        
        self.ref_model = ref_model
        if self.ref_model is None and self.args.beta != 0.0:
            self.ref_model = deepcopy(model)
            self.ref_model.eval()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.train_dataset = train_dataset
        self.input_buffer = [None] * self.args.gradient_accumulation_steps
        self.update_steps = 0
        self.global_steps = 0
        # BFloat16 does not need GradScaler
        self.scaler = torch.amp.GradScaler() if (self.args.device == 'cuda' and self.model.dtype != torch.bfloat16) else None
        
        # Initialize Evaluator
        self.evaluator = mem_utils.MemoryEvaluator()

    def get_tokenizer(self, tokenizer):
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def downstream_evaluate(self, memory, fact, query, answer, context_memory, extraction_output, update_plan):
        """
        Delegate to mem_utils.MemoryEvaluator
        """
        return self.evaluator.evaluate(
            memory=memory,
            fact=fact,
            query=query,
            answer=answer,
            context_memory=context_memory,
            extraction_output=extraction_output,
            update_plan_output=update_plan
        )

    def construct_extraction_prompt(self, fact):
        return mem_utils.construct_extraction_prompt(fact)

    def construct_update_prompt(self, context_memory, extraction_output):
        return mem_utils.construct_update_prompt(context_memory, extraction_output)

    def generate_samples(self, batch_data):
        samples_list = []
        self.model.eval()
        
        bs = len(batch_data['fact'])
        num_generations = self.args.num_generations
        
        # --- Step 1: Extraction ---
        prompts_ext = [self.construct_extraction_prompt(fact) for fact in batch_data['fact']]
        msgs_ext_list = [[{"role": "user", "content": p}] for p in prompts_ext]
        text_ext_list = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs_ext_list]
        
        # Repeat prompts for num_generations
        # Shape: [p1, p1, p1, p1, p2, p2, p2, p2, ...]
        text_ext_batch = []
        for t in text_ext_list:
            text_ext_batch.extend([t] * num_generations)
            
        tokenized_ext = self.tokenizer(text_ext_batch, 
                                     padding='longest', 
                                     max_length=self.args.max_prompt_length, 
                                     truncation=True, 
                                     return_tensors='pt').to(self.args.device)
        
        with torch.no_grad():
            ext_outputs = self.model.generate(**tokenized_ext, 
                                            max_new_tokens=self.args.max_generate_length,
                                            temperature=1.0)
            
        prompt_len_ext = tokenized_ext['input_ids'].size(1)
        resp_ids_ext = ext_outputs[:, prompt_len_ext:]
        resp_texts_ext = self.tokenizer.batch_decode(resp_ids_ext, skip_special_tokens=True)

        # Organize Extraction Samples
        if self.args.train_extraction:
            attention_mask_ext = (ext_outputs.ne(self.tokenizer.pad_token_id)).long()
            action_mask_ext = (resp_ids_ext.ne(self.tokenizer.eos_token_id) & 
                              resp_ids_ext.ne(self.tokenizer.pad_token_id)).long()
            
            # Create samples per original batch item
            for i in range(bs):
                start_idx = i * num_generations
                end_idx = start_idx + num_generations
                
                samples_ext = Samples(
                    prompt_response_ids=ext_outputs[start_idx:end_idx],
                    response_ids=resp_ids_ext[start_idx:end_idx],
                    prompt=prompts_ext[i],
                    answer=batch_data['answer'][i],
                    attention_mask=attention_mask_ext[start_idx:end_idx],
                    action_mask=action_mask_ext[start_idx:end_idx],
                    num_actions=action_mask_ext[start_idx:end_idx].size(1),
                    response_length=action_mask_ext[start_idx:end_idx].float().sum(dim=-1),
                    step_type='extraction'
                )
                samples_list.append(samples_ext)

        # --- Step 2: Update ---
        prompts_upd = []
        # Construct update prompts for all generated extractions
        for i in range(bs):
            ctx_mem = batch_data['context_memory'][i]
            for j in range(num_generations):
                global_idx = i * num_generations + j
                prompts_upd.append(self.construct_update_prompt(ctx_mem, resp_texts_ext[global_idx]))
        
        msgs_upd_list = [[{"role": "user", "content": p}] for p in prompts_upd]
        text_upd_list = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs_upd_list]
        
        tokenized_upd = self.tokenizer(text_upd_list,
                                     padding='longest',
                                     max_length=self.args.max_prompt_length,
                                     truncation=True,
                                     return_tensors='pt').to(self.args.device)
        
        with torch.no_grad():
            upd_outputs = self.model.generate(**tokenized_upd,
                                            max_new_tokens=self.args.max_generate_length,
                                            temperature=1.0)
        
        prompt_len_upd = tokenized_upd['input_ids'].size(1)
        resp_ids_upd = upd_outputs[:, prompt_len_upd:]
        resp_texts_upd = self.tokenizer.batch_decode(resp_ids_upd, skip_special_tokens=True)
        
        # Calculate Rewards (requires iterating through batch and generations)
        all_rewards = []
        for i in range(bs):
            rewards_i = []
            memory = batch_data['memory'][i]
            fact = batch_data['fact'][i]
            query = batch_data['query'][i]
            answer = batch_data['answer'][i]
            ctx_mem = batch_data['context_memory'][i]
            
            for j in range(num_generations):
                global_idx = i * num_generations + j
                r = self.downstream_evaluate(
                    memory, fact, query, answer, ctx_mem, 
                    resp_texts_ext[global_idx], 
                    resp_texts_upd[global_idx]
                )
                rewards_i.append(r)
            all_rewards.append(torch.tensor(rewards_i, device=self.args.device, dtype=torch.float32))

        # Organize Update Samples and Attach Rewards
        for i in range(bs):
            # Attach rewards to extraction samples if they exist
            if self.args.train_extraction:
                # Find the extraction sample corresponding to this batch item
                # samples_list already contains bs extraction samples in order
                samples_list[i].rewards = all_rewards[i]
            
            if self.args.train_update:
                start_idx = i * num_generations
                end_idx = start_idx + num_generations
                
                attention_mask_upd = (upd_outputs.ne(self.tokenizer.pad_token_id)).long()
                action_mask_upd = (resp_ids_upd.ne(self.tokenizer.eos_token_id) & 
                                  resp_ids_upd.ne(self.tokenizer.pad_token_id)).long()
                
                samples_upd = Samples(
                    prompt_response_ids=upd_outputs[start_idx:end_idx],
                    response_ids=resp_ids_upd[start_idx:end_idx],
                    prompt=prompts_upd[start_idx:end_idx],
                    answer=batch_data['answer'][i],
                    attention_mask=attention_mask_upd[start_idx:end_idx],
                    action_mask=action_mask_upd[start_idx:end_idx],
                    num_actions=action_mask_upd[start_idx:end_idx].size(1),
                    response_length=action_mask_upd[start_idx:end_idx].float().sum(dim=-1),
                    step_type='update',
                    rewards=all_rewards[i]
                )
                samples_list.append(samples_upd)
                
        return samples_list

    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        output = model(input_ids, attention_mask=attention_mask, use_cache=False)
        logits = output.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
        action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
        return action_log_probs

    def generate_experiences(self, batch_data):
        self.model.eval()
        samples_list = self.generate_samples(batch_data)
        
        batch_exp_ext = {
            "prompt_response_ids": [],
            "attention_mask": [],
            "action_mask": [],
            "advantages": [],
            "old_action_log_probs": [],
            "ref_action_log_probs": []
        }
        
        batch_exp_upd = {
            "prompt_response_ids": [],
            "attention_mask": [],
            "action_mask": [],
            "advantages": [],
            "old_action_log_probs": [],
            "ref_action_log_probs": []
        }
        
        for samples in samples_list:
            rewards = samples.rewards
            mean_reward = rewards.mean()
            std_reward = rewards.std()
            advantages = (rewards - mean_reward) / (std_reward + 1e-8)

            swanlab.log({
                            f"reward_mean/{samples.step_type}": mean_reward.item(),
                            f"reward_std/{samples.step_type}": std_reward.item(),
                            "reward_mean": mean_reward.item(), # 全局平均
                            "reward_std": std_reward.item()    # 全局标准差
                        })

            with torch.no_grad():
                old_log_probs = self.get_action_log_probs(self.model, samples.prompt_response_ids, samples.attention_mask, samples.num_actions)
                ref_log_probs = None
                if self.ref_model:
                    ref_log_probs = self.get_action_log_probs(self.ref_model, samples.prompt_response_ids, samples.attention_mask, samples.num_actions)

            # Assign to correct batch dictionary based on step_type
            if samples.step_type == 'extraction':
                target_dict = batch_exp_ext
            elif samples.step_type == 'update':
                target_dict = batch_exp_upd
            else:
                continue

            target_dict["prompt_response_ids"].append(samples.prompt_response_ids)
            target_dict["attention_mask"].append(samples.attention_mask)
            target_dict["action_mask"].append(samples.action_mask)
            target_dict["advantages"].append(advantages)
            target_dict["old_action_log_probs"].append(old_log_probs)
            if ref_log_probs is not None:
                target_dict["ref_action_log_probs"].append(ref_log_probs)

        # Helper to collate a single experience dict
        def collate_exp(exp_dict):
            if not exp_dict["prompt_response_ids"]:
                return None
            return {
                "prompt_response_ids": torch.cat(exp_dict["prompt_response_ids"], dim=0),
                "attention_mask": torch.cat(exp_dict["attention_mask"], dim=0),
                "action_mask": torch.cat(exp_dict["action_mask"], dim=0),
                "advantages": torch.cat(exp_dict["advantages"], dim=0),
                "old_action_log_probs": torch.cat(exp_dict["old_action_log_probs"], dim=0),
                "ref_action_log_probs": torch.cat(exp_dict["ref_action_log_probs"], dim=0) if self.ref_model else None
            }
        return {
            "extraction": collate_exp(batch_exp_ext),
            "update": collate_exp(batch_exp_upd)
        }

    def compute_loss(self, model, inputs):
        prompt_response_ids = inputs['prompt_response_ids']
        attention_mask = inputs['attention_mask']
        action_mask = inputs['action_mask']
        num_actions = action_mask.size(1)
        
        action_log_probs = self.get_action_log_probs(model, prompt_response_ids, attention_mask, num_actions)
        
        k3 = None
        if self.args.beta != 0.0 and inputs.get('ref_action_log_probs') is not None:
            ref_action_log_probs = inputs['ref_action_log_probs']
            log_ratio = ref_action_log_probs - action_log_probs
            log_ratio = log_ratio * action_mask
            k3 = log_ratio.exp() - 1 - log_ratio
            
        advantages = inputs['advantages']
        old_action_log_probs = inputs['old_action_log_probs']
        
        coef_1 = torch.exp(action_log_probs - old_action_log_probs)
        coef_2 = torch.clamp(coef_1, 1 - self.args.clip_eps, 1 + self.args.clip_eps)
        
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss * action_mask
        
        if k3 is not None:
            per_token_loss = per_token_loss + self.args.beta * k3
            
        loss = per_token_loss.sum(dim=1) / (action_mask.sum(dim=1) + 1e-8)
        return loss.mean()

    def train_step(self, model, inputs, optimizer, step):
        if inputs is None: return
        model.train()
        
        if self.scaler:
             with torch.amp.autocast(device_type='cuda'):
                 loss = self.compute_loss(model, inputs)
                 self.scaler.scale(loss / self.args.gradient_accumulation_steps).backward()
        else:
             loss = self.compute_loss(model, inputs)
             (loss / self.args.gradient_accumulation_steps).backward()
             
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            if self.scaler:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

            swanlab.log({
                            "train/loss": loss.item(),
                            "step": self.update_steps
                        })
            
            if self.update_steps % 10 == 0:
                print(f"Step {self.update_steps}: Loss {loss.item():.4f}")

    def train(self):
        self.optimizer.zero_grad()
        self.global_steps = self.args.num_iterations * self.args.epoch * len(self.train_dataset) // (self.args.batch_size * self.args.gradient_accumulation_steps)
        
        # Custom collate function to handle variable length lists (don't stack them)
        def collate_fn(batch):
            keys = batch[0].keys()
            return {key: [d[key] for d in batch] for key in keys}

        for epoch in range(self.args.epoch):
            
            dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=collate_fn)
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{self.args.epoch}")
            for idx, batch in pbar:
                experiences = self.generate_experiences(batch)
                
                pbar.set_postfix(step=idx, global_step=self.update_steps)
                
                if experiences:
                    # Inner Loop for GRPO/PPO
                    for _ in range(self.args.num_iterations):
                        # Train Extraction
                        if self.args.train_extraction and experiences["extraction"] is not None:
                            self.train_step(self.model, experiences["extraction"], self.optimizer, idx)
                        
                        # Train Update
                        if self.args.train_update and experiences["update"] is not None:
                            self.train_step(self.model, experiences["update"], self.optimizer, idx)
                    
                    self.update_steps += 1
                    
                    if self.update_steps % self.args.save_steps == 0:
                        self.save_model(f"checkpoint_{self.update_steps}")
                
                torch.cuda.empty_cache()

    def save_model(self, name):
        path = os.path.join(self.args.output_dir, name)
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/home/models/qwen3-4b", help="Path to the model")
    parser.add_argument("--data_path", type=str, default="./scripts/training_data_with_context.jsonl", help="Path to the training data")
    parser.add_argument("--output_dir", type=str, default="./output/mem_grpo", help="Output directory")
    
    # Parse known args to allow passing other args if needed
    args, unknown = parser.parse_known_args()
    
    print(f"Loading model from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        "device_map": "auto",
        "trust_remote_code": True
    }
    
    try:
        import flash_attn
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")
    except ImportError:
        print("Flash Attention 2 not found, using default attention")
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        **model_kwargs
    )
    
    # Configure Training Arguments
    grpo_args = MemGRPOArguments(
        output_dir=args.output_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4, # Group size
        save_steps=100,
        epoch=1,
        max_prompt_length=2048,
        max_generate_length=4096,
        train_extraction=True,
        train_update=True
    )
    os.environ["SWANLAB_API_KEY"] = "Zkrggz0kWlnEuNRu5r4dz"
    swanlab.init(
            project="MemFactory",
            config=vars(grpo_args)
        )
    
    print(f"Loading data from {args.data_path}...")
    # Initialize Dataset
    dataset = MemoryDataset(args.data_path, tokenizer)
    
    if len(dataset) == 0:
        print("Error: Dataset is empty or file not found. Please run 'python scripts/process_locomo.py' first.")
        assert False
        
    print("Initializing Trainer...")
    trainer = MemGRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    
    print("Starting Training...")
    trainer.train()

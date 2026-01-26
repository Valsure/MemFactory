
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
    max_prompt_length: int = 1024
    max_generate_length: int = 512
    beta: float = 0.1 # KL penalty
    clip_eps: float = 0.2
    gradient_accumulation_steps: int = 1
    num_iterations: int = 1
    batch_size: int = 1
    train_extraction: bool = True
    train_update: bool = True

class MemoryDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = []
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        # Parse context_memory to MemoryItem objects
                        if "context_memory" in item and isinstance(item["context_memory"], list):
                            # Assuming items in list are dicts matching MemoryItem fields
                            item["context_memory"] = [MemoryItem(**m) for m in item["context_memory"]]
                        self.data.append(item)
        else:
            print(f"Warning: {data_path} not found. Using dummy data.")
            for i in range(2):
                self.data.append({
                    "memory": "Initial memory state",
                    "fact": f"User visited location {i}",
                    "query": f"Where did user visit?",
                    "answer": f"Location {i}",
                    "context_memory": [
                        MemoryItem(id=f"init_{i}", key="history", value="User likes travel", memory_type="UserMemory", tags=[])
                    ]
                })
        
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
        self.scaler = torch.amp.GradScaler() if self.args.device == 'cuda' else None
        
        # Initialize Evaluator
        self.evaluator = mem_utils.MemoryEvaluator()

    def get_tokenizer(self, tokenizer):
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def downstream_evaluate(self, fact, query, answer, context_memory, extraction_output, update_plan):
        """
        Delegate to mem_utils.MemoryEvaluator
        """
        return self.evaluator.evaluate(
            fact=fact,
            query=query,
            answer=answer,
            context_memory=context_memory,
            extraction_output=extraction_output,
            update_plan_output=update_plan
        )

    def construct_extraction_prompt(self, fact, context_memory):
        # Format conversation or fact for extraction
        # context_memory might be used to simulate conversation history
        conversation_str = f"User: {fact}\n" 
        # Add some context if available? 
        # For this task, we treat 'fact' as the user input to extract from.
        return mem_utils.construct_extraction_prompt(conversation_str)

    def construct_update_prompt(self, fact, context_memory, extraction_output):
        # 1. Parse extraction output to get facts
        ext_json = mem_utils.parse_json_from_text(extraction_output)
        facts = []
        if ext_json and "memory_list" in ext_json:
            for m in ext_json["memory_list"]:
                facts.append(m.get("value", ""))
        
        if not facts:
            facts = [fact] # Fallback
            
        # 2. Convert context_memory (List[MemoryItem]) to List[Dict] for prompt
        old_memory = mem_utils.memory_to_dict(context_memory)
        
        return mem_utils.construct_update_prompt(facts, old_memory)

    def generate_samples(self, batch_data):
        samples_list = []
        self.model.eval()
        
        bs = len(batch_data['fact'])
        
        for i in range(bs):
            fact = batch_data['fact'][i]
            query = batch_data['query'][i]
            answer = batch_data['answer'][i]
            ctx_mem = batch_data['context_memory'][i] # List[MemoryItem]
            
            # --- Step 1: Extraction ---
            prompt_ext = self.construct_extraction_prompt(fact, ctx_mem)
            
            msgs_ext = [{"role": "user", "content": prompt_ext}]
            text_ext = self.tokenizer.apply_chat_template(msgs_ext, add_generation_prompt=True, tokenize=False)
            
            tokenized_ext = self.tokenizer([text_ext] * self.args.num_generations, 
                                         padding='max_length', 
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
            
            # Create Extraction Samples
            if self.args.train_extraction:
                attention_mask_ext = (ext_outputs.ne(self.tokenizer.pad_token_id)).long()
                action_mask_ext = (resp_ids_ext.ne(self.tokenizer.eos_token_id) & 
                                  resp_ids_ext.ne(self.tokenizer.pad_token_id)).long()
                
                samples_ext = Samples(
                    prompt_response_ids=ext_outputs,
                    response_ids=resp_ids_ext,
                    prompt=prompt_ext,
                    answer=answer,
                    attention_mask=attention_mask_ext,
                    action_mask=action_mask_ext,
                    num_actions=action_mask_ext.size(1),
                    response_length=action_mask_ext.float().sum(dim=-1),
                    step_type='extraction'
                )
                samples_list.append(samples_ext)
            else:
                samples_ext = None # Placeholder if not training extraction

            # --- Step 2: Update ---
            prompts_upd = []
            for r_text in resp_texts_ext:
                prompts_upd.append(self.construct_update_prompt(fact, ctx_mem, r_text))
            
            # Note: prompts_upd might be different lengths, but tokenizer handles padding
            msgs_upd_list = [[{"role": "user", "content": p}] for p in prompts_upd]
            texts_upd = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs_upd_list]
            
            tokenized_upd = self.tokenizer(texts_upd,
                                         padding='max_length',
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
            
            # Calculate Rewards
            rewards = []
            for j in range(self.args.num_generations):
                # Pass original memory info. For simulation, fact + ctx_mem is enough.
                r = self.downstream_evaluate(fact, query, answer, ctx_mem, resp_texts_ext[j], resp_texts_upd[j])
                rewards.append(r)
            
            rewards_tensor = torch.tensor(rewards, device=self.args.device, dtype=torch.float32)
            
            # Attach rewards
            if samples_ext:
                samples_ext.rewards = rewards_tensor
            
            if self.args.train_update:
                attention_mask_upd = (upd_outputs.ne(self.tokenizer.pad_token_id)).long()
                action_mask_upd = (resp_ids_upd.ne(self.tokenizer.eos_token_id) & 
                                  resp_ids_upd.ne(self.tokenizer.pad_token_id)).long()
                
                samples_upd = Samples(
                    prompt_response_ids=upd_outputs,
                    response_ids=resp_ids_upd,
                    prompt=prompts_upd,
                    answer=answer,
                    attention_mask=attention_mask_upd,
                    action_mask=action_mask_upd,
                    num_actions=action_mask_upd.size(1),
                    response_length=action_mask_upd.float().sum(dim=-1),
                    step_type='update',
                    rewards=rewards_tensor
                )
                samples_list.append(samples_upd)
                
        return samples_list

    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
        action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
        return action_log_probs

    def generate_experiences(self, batch_data):
        self.model.eval()
        samples_list = self.generate_samples(batch_data)
        
        batch_exp = {
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
            
            with torch.no_grad():
                old_log_probs = self.get_action_log_probs(self.model, samples.prompt_response_ids, samples.attention_mask, samples.num_actions)
                ref_log_probs = None
                if self.ref_model:
                    ref_log_probs = self.get_action_log_probs(self.ref_model, samples.prompt_response_ids, samples.attention_mask, samples.num_actions)

            batch_exp["prompt_response_ids"].append(samples.prompt_response_ids)
            batch_exp["attention_mask"].append(samples.attention_mask)
            batch_exp["action_mask"].append(samples.action_mask)
            batch_exp["advantages"].append(advantages)
            batch_exp["old_action_log_probs"].append(old_log_probs)
            if ref_log_probs is not None:
                batch_exp["ref_action_log_probs"].append(ref_log_probs)

        if not batch_exp["prompt_response_ids"]:
            return None

        return {
            "prompt_response_ids": torch.cat(batch_exp["prompt_response_ids"], dim=0),
            "attention_mask": torch.cat(batch_exp["attention_mask"], dim=0),
            "action_mask": torch.cat(batch_exp["action_mask"], dim=0),
            "advantages": torch.cat(batch_exp["advantages"], dim=0),
            "old_action_log_probs": torch.cat(batch_exp["old_action_log_probs"], dim=0),
            "ref_action_log_probs": torch.cat(batch_exp["ref_action_log_probs"], dim=0) if self.ref_model else None
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
            
            if self.update_steps % 10 == 0:
                print(f"Step {self.update_steps}: Loss {loss.item():.4f}")

    def train(self):
        self.optimizer.zero_grad()
        self.global_steps = self.args.num_iterations * self.args.epoch * len(self.train_dataset) // (self.args.batch_size * self.args.gradient_accumulation_steps)
        
        for epoch in range(self.args.epoch):
            dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
            for idx, batch in enumerate(dataloader):
                experiences = self.generate_experiences(batch)
                
                if experiences:
                    self.train_step(self.model, experiences, self.optimizer, idx)
                    self.update_steps += 1
                    
                    if self.update_steps % self.args.save_steps == 0:
                        self.save_model(f"checkpoint_{self.update_steps}")

    def save_model(self, name):
        path = os.path.join(self.args.output_dir, name)
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

if __name__ == "__main__":
    args = MemGRPOArguments()
    pass

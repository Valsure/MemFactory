from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
import os
from copy import deepcopy
from typing import Optional, Dict, Any
from tqdm import tqdm
import swanlab
from torch.utils.data import DataLoader

from ..common.registry import TRAINER_REGISTRY, ENV_REGISTRY, AGENT_REGISTRY
from ..common.utils import LLMClient
from ..modules.base import Samples

@dataclass
class MemGRPOArguments:
    output_dir: str = "./output"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr: float = 5e-7
    save_steps: int = 500
    epoch: int = 1
    beta: float = 0.1
    clip_eps: float = 0.2
    gradient_accumulation_steps: int = 1
    num_iterations: int = 1
    batch_size: int = 1
    gradient_checkpointing: bool = True
    
    # MemFactory specific
    agent_type: str = "naive"
    env_type: str = "longcontext"
    max_chunk_number: int = 5
    num_generations: int = 4
    max_generate_length: int = 2048
    
    # Training control
    train_extraction: bool = True
    train_update: bool = True

@TRAINER_REGISTRY.register("mem_grpo")
class MemGRPOTrainer:
    def __init__(self, model, args: MemGRPOArguments, tokenizer, ref_model=None):
        self.args = args
        self.model = model.to(self.args.device)
        self.tokenizer = tokenizer
        
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        self.ref_model = ref_model
        if self.ref_model is None and self.args.beta != 0.0:
            self.ref_model = deepcopy(model)
            self.ref_model.eval()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scaler = torch.amp.GradScaler() if (self.args.device == 'cuda' and self.model.dtype != torch.bfloat16) else None
        
        self.update_steps = 0
        self.llm_client = LLMClient() # For reward computation

    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        output = model(input_ids, attention_mask=attention_mask, use_cache=False)
        logits = output.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        
        per_token_nll = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        log_probs_labels = -per_token_nll.view(shift_labels.size())
        
        return log_probs_labels[:, -num_actions:]

    def compute_loss(self, model, inputs):
        prompt_response_ids = inputs['prompt_response_ids']
        attention_mask = inputs['attention_mask']
        action_mask = inputs['action_mask']
        num_actions = inputs['old_action_log_probs'].size(1)
        
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

    def train_step(self, inputs, step):
        self.model.train()
        training_batch_size = 4
        total_samples = inputs['prompt_response_ids'].size(0)
        total_loss = 0.0
        
        for i in range(0, total_samples, training_batch_size):
            end_i = min(i + training_batch_size, total_samples)
            mini_inputs = {k: v[i:end_i] if v is not None else None for k, v in inputs.items()}
            
            loss = self.compute_loss(self.model, mini_inputs)
            
            mini_batch_size = end_i - i
            scale_factor = mini_batch_size / total_samples
            scaled_loss = loss * scale_factor
            
            if self.scaler:
                 with torch.amp.autocast(device_type='cuda'):
                     self.scaler.scale(scaled_loss / self.args.gradient_accumulation_steps).backward()
            else:
                 (scaled_loss / self.args.gradient_accumulation_steps).backward()
            
            total_loss += scaled_loss.item()
             
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            self.optimizer.zero_grad()

            swanlab.log({"train/loss": total_loss, "step": self.update_steps})
            if self.update_steps % 10 == 0:
                print(f"Step {self.update_steps}: Loss {total_loss:.4f}")

    def save_model(self, name):
        path = os.path.join(self.args.output_dir, name)
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
    def _prepare_train_inputs(self, samples):
        with torch.no_grad():
            old_lp = self.get_action_log_probs(self.model, samples.prompt_response_ids, samples.attention_mask, samples.num_actions)
            ref_lp = self.get_action_log_probs(self.ref_model, samples.prompt_response_ids, samples.attention_mask, samples.num_actions) if self.ref_model else None
        
        return {
            "prompt_response_ids": samples.prompt_response_ids,
            "attention_mask": samples.attention_mask,
            "action_mask": samples.action_mask,
            "advantages": samples.rewards,
            "old_action_log_probs": old_lp,
            "ref_action_log_probs": ref_lp
        }

    def train(self, data_path):
        # 1. Initialize Env
        EnvClass = ENV_REGISTRY.get(self.args.env_type)
        env = EnvClass(data_path, self.tokenizer)
        
        # 2. Initialize Agent
        AgentClass = AGENT_REGISTRY.get(self.args.agent_type)
        agent = AgentClass(
            self.tokenizer, 
            device=self.args.device,
            chunk_size=2048, 
            max_chunk_number=self.args.max_chunk_number,
            num_generations=self.args.num_generations,
            max_generate_length=self.args.max_generate_length
        )

        dataloader = DataLoader(env, batch_size=self.args.batch_size, shuffle=False, collate_fn=env.collate_fn)
        
        # Define reward function wrapper
        # For MemoryBankEnv, it uses: predictions(Dict), ground_truths, num_generations
        # For LongContextMemoryEnv, it uses: predictions(List), ground_truths, questions
        # We need to unify or adapt
        def reward_fn_wrapper(*args, **kwargs):
            return env.compute_reward(*args, **kwargs, llm_client=self.llm_client)

        for epoch in range(self.args.epoch):
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{self.args.epoch}")
            for idx, batch in pbar:
                # Rollout
                self.model.eval()
                samples_output = agent.rollout(self.model, batch, reward_fn=reward_fn_wrapper)
                
                if samples_output:
                    # Normalize output to Dict[str, Samples]
                    if isinstance(samples_output, Samples):
                        samples_dict = {"default": samples_output}
                    elif isinstance(samples_output, dict):
                        samples_dict = samples_output
                    else:
                        continue
                        
                    # Inner Loop
                    for _ in range(self.args.num_iterations):
                        # Iterate over all types of samples returned by rollout
                        for step_type, samples in samples_dict.items():
                            # Check if we should train this step type
                            # Logic: 
                            # If step_type is 'extraction' -> check args.train_extraction
                            # If step_type is 'update' -> check args.train_update
                            # If step_type is 'default' (Naive Agent) -> Always train
                            
                            should_train = True
                            if step_type == 'extraction' and not self.args.train_extraction:
                                should_train = False
                            if step_type == 'update' and not self.args.train_update:
                                should_train = False
                                
                            if should_train and samples.rewards is not None:
                                train_inputs = self._prepare_train_inputs(samples)
                                self.train_step(train_inputs, idx)
                    
                    self.update_steps += 1
                    if self.update_steps % self.args.save_steps == 0:
                        self.save_model(f"checkpoint_{self.update_steps}")
                
                torch.cuda.empty_cache()

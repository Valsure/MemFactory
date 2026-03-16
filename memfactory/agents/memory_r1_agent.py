import torch
import json
from typing import Dict, Any, List, Optional
from ..common.registry import AGENT_REGISTRY
from .base import BaseAgent
from ..modules.base import Samples
from ..modules.memory_extractor import NaiveExtractor
from ..modules.memory_updater import NaiveUpdater
from ..modules.memory_retriever import NaiveRetriever

@AGENT_REGISTRY.register("memory_r1_agent")
class MemoryR1Agent(BaseAgent):
    """
    Composite Agent that orchestrates Extractor, Updater, and Retriever modules.
    """
    def __init__(self, tokenizer, device="cuda", **kwargs):
        super().__init__(tokenizer, device)
        # Initialize sub-modules
        # Note: Modules are initialized with the same tokenizer/device
        self.extractor = NaiveExtractor(tokenizer, device, **kwargs)
        self.updater = NaiveUpdater(tokenizer, device, **kwargs)
        # self.retriever = NaiveRetriever(tokenizer, device, **kwargs)
        
        self.num_generations = kwargs.get("num_generations", 4)
        
    def process_samples(self, prompts: List[str], responses: List[str], rewards: torch.Tensor, step_type: str) -> Samples:
        # Tokenize and Pad logic adapted from MemoryAgent
        prompts_ids = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        responses_ids = [self.tokenizer.encode(r, add_special_tokens=False) + [self.tokenizer.eos_token_id] for r in responses]
        
        # Pad Prompts (Left)
        max_p_len = max(len(ids) for ids in prompts_ids)
        padded_prompts = []
        prompt_masks = []
        for p_ids in prompts_ids:
            pad_len = max_p_len - len(p_ids)
            padded_prompts.append([self.tokenizer.pad_token_id] * pad_len + p_ids)
            prompt_masks.append([0] * pad_len + [1] * len(p_ids))

        # Pad Responses (Right)
        max_r_len = max(len(ids) for ids in responses_ids)
        padded_responses = []
        response_masks = []
        response_att_masks = []
        for r_ids in responses_ids:
            pad_len = max_r_len - len(r_ids)
            padded_responses.append(r_ids + [self.tokenizer.pad_token_id] * pad_len)
            response_masks.append([1] * len(r_ids) + [0] * pad_len)
            response_att_masks.append([1] * len(r_ids) + [0] * pad_len)

        # Concat
        input_ids = torch.tensor([p + r for p, r in zip(padded_prompts, padded_responses)], device=self.device, dtype=torch.long)
        attention_mask = torch.tensor([p + r for p, r in zip(prompt_masks, response_att_masks)], device=self.device, dtype=torch.long)
        action_mask = torch.tensor(response_masks, device=self.device, dtype=torch.bool)
        
        # Process rewards/advantages
        if rewards is not None:
             # Calculate advantages
             bs = len(prompts) // self.num_generations # Assuming structure
             # rewards passed here are already [BS * NumGen]
             # But if bs=0 or mismatch, handle gracefully
             if bs > 0:
                 reshaped_rewards = rewards.view(bs, self.num_generations)
                 mean = reshaped_rewards.mean(dim=1, keepdim=True)
                 std = reshaped_rewards.std(dim=1, keepdim=True)
                 adv = (reshaped_rewards - mean) / (std + 1e-8)
                 advantages_tensor = adv.flatten().to(self.device)
             else:
                 advantages_tensor = rewards.to(self.device)
        else:
            advantages_tensor = None

        samples = Samples(
            prompt_response_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=max_r_len,
            rewards=advantages_tensor,
            step_type=step_type,
            response_length=action_mask.float().sum(dim=-1)
        )
        return samples

    def rollout(self, model: Any, batch_data: Dict[str, Any], **kwargs) -> Dict[str, Samples]:
        """
        Orchestrate the rollout process:
        1. Extract memories from 'fact'.
        2. Update memory bank with 'context_memory' and extracted memories.
        3. Compute rewards using the Environment.
        """
        # 1. Extraction
        facts = batch_data['fact']
        ext_prompts, ext_texts = self.extractor.generate(model, facts, self.num_generations)
        
        # 2. Update & Reward Calculation (delegated to Updater)
        reward_fn = kwargs.get('reward_fn')
        upd_prompts, upd_texts, scores = self.updater.rollout(
            model, 
            batch_data, 
            ext_texts, 
            reward_fn=reward_fn
        )
        if not scores or scores['extraction'] is None:
            return None
            
        # 3. Process into Samples
        # (tokenize, pad, compute_adv)
        ext_samples = self.process_samples(ext_prompts, ext_texts, scores['extraction'], 'extraction')
        upd_samples = self.process_samples(upd_prompts, upd_texts, scores['update'], 'update')
        
        return {
            "extraction": ext_samples,
            "update": upd_samples
        }

    def inference(self, batch_data, **kwargs):
        # Use vllm server to generate extraction and update samples for given batch data.
        # we encourage you to implement your own inference logic.
        pass

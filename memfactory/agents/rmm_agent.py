import torch
import json
from typing import Dict, Any, List, Optional
from ..common.registry import AGENT_REGISTRY
from .base import BaseAgent
from ..modules.base import Samples
from ..modules.memory_extractor import NaiveExtractor
from ..modules.memory_updater import NaiveUpdater
from ..modules.memory_retriever import RerankRetriever
from ..common.utils import LLMClient
from ..envs.memory_bank_utils import get_memory_store

@AGENT_REGISTRY.register("memory_rmm_agent")
class MemoryRMMAgent(BaseAgent):
    """
    Composite Agent that orchestrates Extractor, Updater, and Retriever modules.
    Focused on training the Retriever (Reranker).
    """
    def __init__(self, tokenizer, device="cuda", **kwargs):
        super().__init__(tokenizer, device)
        # Initialize sub-modules
        # Note: Modules are initialized with the same tokenizer/device
        self.extractor = NaiveExtractor(tokenizer, device, **kwargs)
        self.updater = NaiveUpdater(tokenizer, device, **kwargs)
        self.retriever = RerankRetriever(tokenizer, device, **kwargs)
        
        self.llm_client = LLMClient()
        self.store = get_memory_store()
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
        Orchestrate the rollout process for RMM:
        1. Extract memories (Inference mode, num_gen=1).
        2. Update memory bank (Inference mode, num_gen=1).
        3. Train Retriever (Rerank) with Extracted/Updated context.
        """
        # 1. Extraction (Inference)
        bs = len(batch_data['fact'])
        ext_texts = self.extractor.inference(self.llm_client, batch_data, num_generations=1)
        assert len(ext_texts) == bs, "extraction texts must match batch size"
        # 2. Update (Inference)
        upd_texts = self.updater.inference(self.llm_client, batch_data, ext_texts, num_generations=1)
        assert len(upd_texts) == bs, "update texts must match batch size"
        
        # 3. Retriever Rollout (Training)
        ret_prompts, ret_responses, scores = self.retriever.rollout(
            model, 
            batch_data, 
            extraction_texts=ext_texts,
            update_texts=upd_texts,
            store=self.store,
            reward_fn=kwargs.get('reward_fn'),
            num_generations=self.num_generations
        )
        if not scores:
            return None

        # 4. Process into Samples
        ret_samples = self.process_samples(ret_prompts, ret_responses, scores['rerank'], 'rerank')
        
        return {
            "default": ret_samples
        }

    def inference(self, batch_data, **kwargs):
        # Use vllm server to generate extraction and update samples for given batch data.
        # we encourage you to implement your own inference logic.
        pass


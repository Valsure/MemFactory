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
        self.retriever = NaiveRetriever(tokenizer, device, **kwargs)
        
        self.num_generations = kwargs.get("num_generations", 4)
        
    def rollout(self, model: Any, batch_data: Dict[str, Any], **kwargs) -> Dict[str, Samples]:
        """
        Orchestrate the rollout process:
        1. Extract memories from 'fact'.
        2. Update memory bank with 'context_memory' and extracted memories.
        3. Compute rewards using the Environment.
        """
        # 1. Extraction
        facts = batch_data['fact']
        ext_texts, ext_samples = self.extractor.generate(model, facts, self.num_generations)
        
        # 2. Update
        context_memories = batch_data['context_memory']
        # We need to adapt the arguments passed to updater.generate
        upd_texts, upd_samples = self.updater.generate(model, context_memories, ext_texts)
        
        # 3. Reward Calculation
        # We need the Environment's compute_reward function, passed via kwargs
        reward_fn = kwargs.get('reward_fn')
        if not reward_fn:
             # Fallback: Zero rewards
             return None

        # Call reward_fn from Environment
        scores = reward_fn(
            predictions={'extraction': ext_texts, 'update': upd_texts}, 
            ground_truths=batch_data,
            num_generations=self.num_generations
        )
        
        # scores should be a dict: {'extraction': Tensor, 'update': Tensor}
        ext_rewards = scores['extraction']
        upd_rewards = scores['update']
        
        # Helper to compute advantages
        def compute_advantages(rewards):
            # rewards: [BS * NumGen]
            # Reshape to [BS, NumGen]
            bs = len(batch_data['fact'])
            rewards = rewards.view(bs, self.num_generations)
            mean = rewards.mean(dim=1, keepdim=True)
            std = rewards.std(dim=1, keepdim=True)
            adv = (rewards - mean) / (std + 1e-8)
            return adv.flatten()

        ext_samples.rewards = compute_advantages(ext_rewards)
        upd_samples.rewards = compute_advantages(upd_rewards)
        
        return {
            "extraction": ext_samples,
            "update": upd_samples
        }

    def inference(self, batch_data, **kwargs):
        pass

import torch
from typing import Dict, Any, List, Optional
from ..common.registry import AGENT_REGISTRY
from .base import BaseAgent
from ..modules.base import Samples
from ..modules.memory_agent import RecurrentMemoryModule

@AGENT_REGISTRY.register("memagent")
class MemoryAgent(BaseAgent):
    """
    The 'Naive' Memory Agent implementation.
    Wraps the RecurrentMemoryModule.
    """
    def __init__(self, tokenizer, device="cuda", **kwargs):
        super().__init__(tokenizer, device)
        # Initialize the core module
        self.module = RecurrentMemoryModule(tokenizer, device, **kwargs)
        self.tokenizer = self.module.tokenizer # Ensure tokenizer sync

    def rollout(self, model: Any, batch_data: Dict[str, Any], **kwargs) -> Optional[Samples]:
        # Delegate rollout logic to the module
        results = self.module.rollout(model, batch_data, **kwargs)
        if not results:
            return None

        # Process results into Samples object (Tokenize, Pad, etc.)
        # This logic was previously in MemoryAgent.rollout
        
        # 2. Tokenize and Pad
        prompts_ids = [self.tokenizer.encode(p, add_special_tokens=False) for p, r, a in results]
        responses_ids = [self.tokenizer.encode(r, add_special_tokens=False) + [self.tokenizer.eos_token_id] for p, r, a in results]
        advantages = [a for p, r, a in results]

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
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        assert max_r_len == action_mask.size(1), "Action mask length must match response length"
        samples = Samples(
            prompt_response_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=max_r_len,
            rewards=advantages_tensor,
            prompt_length=torch.tensor([max_p_len]*len(results), device=self.device),
            response_length=action_mask.sum(dim=1)
        )
        return samples

    def inference(self, batch_data: Dict[str, Any], **kwargs) -> List[str]:
        return self.module.inference(batch_data, **kwargs)

import torch
from typing import Dict, Any, List, Optional
from ..common.registry import MODULE_REGISTRY
from .base import BaseModule, Samples
from ..common.utils import TEMPLATE

# We need to define the specific prompts for Extraction
# Copied from reference code
EXTRACTION_PROMPT_EN = """You are a memory extraction expert.
Your task is to extract memories from the user's perspective based on the conversation between the user and the assistant. This means identifying information the user might remember—including the user's own experiences, thoughts, plans, or statements and actions made by others (such as the assistant) that affect the user or are acknowledged by the user.

Please perform the following operations:
1. Identify information reflecting the user's experiences, beliefs, concerns, decisions, plans, or responses—including meaningful information from the assistant acknowledged or responded to by the user.

2. Clearly parse all references to time, people, and events:
   - If possible, convert relative time expressions (e.g., "yesterday", "next Friday") to absolute dates using message timestamps.
   - Clearly distinguish between event time and message time.
   - If specific locations are mentioned, include them.
   - Resolve all pronouns, aliases, and vague references to full names or clear identities.

3. Always write in the third person perspective, using "User" to refer to the user, rather than the first person.

4. Do not omit any information the user might remember.
   - Include all key experiences, thoughts, emotional reactions, and plans.
   - Prioritize completeness and fidelity over brevity.

Return a valid JSON object with the following structure:

{{
  "memory_list": [
    {{
      "key": "<string, unique and concise memory title>",
      "memory_type": "<string, 'LongTermMemory' or 'UserMemory'>",
      "value": "<detailed, independent, and unambiguous memory statement>",
      "tags": ["<list of relevant topic keywords>"]
    }}
  ],
  "summary": "<paragraph naturally summarizing the above memories from the user's perspective, 120-200 words>"
}}

Conversation:
{conversation}

Your output:"""

@MODULE_REGISTRY.register("naive_extractor")
class NaiveExtractor(BaseModule):
    def __init__(self, tokenizer, device="cuda", **kwargs):
        super().__init__(tokenizer, device)
        self.max_prompt_length = kwargs.get("max_prompt_length", 4096)
        self.max_generate_length = kwargs.get("max_generate_length", 2048)

    def _generate_with_pytorch(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_generate_length,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
        return outputs

    def rollout(self, model, batch_data, **kwargs):
        # This module is usually called by a parent Orchestrator, but can be standalone.
        # If standalone, it needs 'fact' in batch_data.
        # Returns Samples object.
        pass # Not implemented as standalone for now, used by MemoryR1Agent

    def generate(self, model, facts: List[str], num_generations: int = 1) -> Tuple[List[str], Samples]:
        """
        Generate extraction samples.
        Returns:
            generated_texts: List[str] (flattened)
            samples: Samples object (without rewards)
        """
        prompts = [EXTRACTION_PROMPT_EN.format(conversation=fact) for fact in facts]
        
        # Duplicate for num_generations
        batch_prompts = []
        for p in prompts:
            batch_prompts.extend([p] * num_generations)
            
        msgs_list = [[{"role": "user", "content": p}] for p in batch_prompts]
        formatted_prompts = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs_list]
        
        tokenized = self.tokenizer(formatted_prompts, padding=True, return_tensors='pt').to(self.device)
        outputs = self._generate_with_pytorch(model, tokenized['input_ids'], tokenized['attention_mask'])
        
        input_len = tokenized['input_ids'].size(1)
        generated_ids = outputs[:, input_len:]
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Create Samples object
        attention_mask = (outputs.ne(self.tokenizer.pad_token_id)).long()
        action_mask = (generated_ids.ne(self.tokenizer.eos_token_id) & 
                       generated_ids.ne(self.tokenizer.pad_token_id)).long()
        
        samples = Samples(
            prompt_response_ids=outputs,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=action_mask.size(1),
            rewards=None, # To be filled later
            step_type='extraction',
            response_length=action_mask.float().sum(dim=-1)
        )
        
        return generated_texts, samples

    def inference(self, batch_data, **kwargs):
        return []

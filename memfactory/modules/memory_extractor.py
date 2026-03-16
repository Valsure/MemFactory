import torch
from typing import Dict, Any, List, Optional
from ..common.registry import MODULE_REGISTRY
from .base import BaseModule, Samples
from ..common.utils import TEMPLATE
from ..envs.memory_bank_utils import format_conversation, ConversationMessage

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
        self.max_prompt_length = kwargs.get("max_prompt_length", 3072)
        self.max_generate_length = kwargs.get("max_generate_length", 2048)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
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
        # 解释，rollout 要求返回含有奖励的对象，这需要新的架构和算法
        # 目前并没有收录可以单独训练抽取（抽取之后就能获取奖励）的算法，我们鼓励您进行这方面的扩展。
        pass # Not implemented as standalone for now, used by MemoryR1Agent

    def generate(self, model, facts: List[List[Dict]], num_generations: int = 1) -> tuple[List[str], List[str]]:
        prompts = []
        for fact in facts:
            # fact is a list of dicts: [{"role": "user", "content": "...", "timestamp": "..."}]
            conversation_msg_list = []
            for msg in fact:
                msg_fmt = ConversationMessage(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    timestamp=msg.get("timestamp", "")
                )
                conversation_msg_list.append(msg_fmt)
            conversation_str = format_conversation(conversation_msg_list)
            prompts.append(EXTRACTION_PROMPT_EN.format(conversation=conversation_str))
        
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
        
        return formatted_prompts, generated_texts

    def inference(self, llm_client, batch_data, num_generations: int = 1, **kwargs):
        assert num_generations == 1, "we recommend num_generations=1"
        # it's ok because bs=1 most of the time
        facts = batch_data['fact']
        prompts = []
        for fact in facts:
            # fact is a list of dicts: [{"role": "user", "content": "...", "timestamp": "..."}]
            conversation_msg_list = []
            for msg in fact:
                msg_fmt = ConversationMessage(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    timestamp=msg.get("timestamp", "")
                )
                conversation_msg_list.append(msg_fmt)
            conversation_str = format_conversation(conversation_msg_list)
            prompts.append(EXTRACTION_PROMPT_EN.format(conversation=conversation_str))
        
        generated_texts = []
        for prompt in prompts:
            response = llm_client.chat("You are a memory extraction expert.", prompt)
            generated_texts.append(response)
            
        return generated_texts

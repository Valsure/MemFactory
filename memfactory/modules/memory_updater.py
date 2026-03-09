import torch
import json
from typing import Dict, Any, List, Optional, Tuple
from ..common.registry import MODULE_REGISTRY
from .base import BaseModule, Samples

UPDATE_MEMORY_PROMPT = """You are a smart memory manager.
You have two lists of memories:
1. **Existing Memories** (from the database).
2. **New Candidate Memories** (extracted from the latest conversation).

Your goal is to decide how to update the memory database.

**Operations Allowed:**

For **Existing Memories**:
- `NONE`: Keep as is.
- `DEL`: Delete this memory (e.g., if it is contradicted by new info, or merged into a new memory).

For **New Candidate Memories**:
- `ADD`: Add this memory to the database.
- `NONE`: Ignore this memory (e.g., if it's redundant or already covered by existing memories).
- `UPDATE`: Modify this memory before adding (e.g., to merge information from an old memory).

**Merging Strategy:**
If a New Candidate (ID: Y) contains updated information for an Existing Memory (ID: X):
1. Mark Existing Memory X as `DEL`.
2. Mark New Candidate Y as `UPDATE` and provide the merged content.

**Output Format:**
Return a JSON object with a list of operations.
You MUST include an operation for **EVERY** memory item (both Existing and Candidate) in the input lists. Do not skip any IDs.

Format:
```json
{{
  "operations": [
    {{ "id": <id>, "op": "NONE" }},
    {{ "id": <id>, "op": "DEL" }},
    {{ "id": <id>, "op": "ADD" }},
    {{ "id": <id>, "op": "UPDATE", "key": "...", "value": "..." }}
  ]
}}
```

**Task:**

Existing Memories:
{context_memory}

New Candidate Memories:
{candidate_memory}

Output:"""

@MODULE_REGISTRY.register("naive_updater")
class NaiveUpdater(BaseModule):
    def __init__(self, tokenizer, device="cuda", **kwargs):
        super().__init__(tokenizer, device)
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
        pass

    def prepare_memory_lists(self, context_memory, extraction_output):
        # Simplified version of mem_utils.prepare_memory_lists logic for prompt construction
        # We assume context_memory is list of dicts (or MemoryItems)
        # We parse extraction_output (JSON string)
        
        ctx_list = []
        id_counter = 1
        for mem in context_memory:
            # mem could be dict or MemoryItem object. Handle both.
            key = mem.get('key') if isinstance(mem, dict) else mem.key
            val = mem.get('value') if isinstance(mem, dict) else mem.value
            ctx_list.append({"id": id_counter, "key": key, "value": val})
            id_counter += 1
            
        cand_list = []
        try:
            # Clean json string
            json_str = extraction_output
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            json_str = json_str.strip()
            if json_str.startswith("<think>"):
                 json_str = json_str.split("</think>")[-1].strip()
            
            ext_json = json.loads(json_str)
            if "memory_list" in ext_json:
                for item in ext_json["memory_list"]:
                    cand_list.append({
                        "id": id_counter,
                        "key": item.get("key", "Unknown"),
                        "value": item.get("value", "")
                    })
                    id_counter += 1
        except:
            pass # Invalid JSON, prompt will be empty/broken for candidates
            
        return ctx_list, cand_list

    def generate(self, model, context_memories: List[List[Any]], extraction_outputs: List[str]) -> Tuple[List[str], Samples]:
        """
        Generate update samples.
        Note: extraction_outputs is flattened [BS * NumGen]
        context_memories is [BS] -> needs to be repeated
        """
        num_generations = len(extraction_outputs) // len(context_memories)
        prompts = []
        
        for i, ctx_mem in enumerate(context_memories):
            for j in range(num_generations):
                global_idx = i * num_generations + j
                ext_out = extraction_outputs[global_idx]
                
                ctx_fmt, cand_fmt = self.prepare_memory_lists(ctx_mem, ext_out)
                
                prompt = UPDATE_MEMORY_PROMPT.format(
                    context_memory=json.dumps(ctx_fmt, ensure_ascii=False, indent=2),
                    candidate_memory=json.dumps(cand_fmt, ensure_ascii=False, indent=2)
                )
                prompts.append(prompt)
        
        msgs_list = [[{"role": "user", "content": p}] for p in prompts]
        formatted_prompts = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs_list]
        
        tokenized = self.tokenizer(formatted_prompts, padding=True, return_tensors='pt').to(self.device)
        outputs = self._generate_with_pytorch(model, tokenized['input_ids'], tokenized['attention_mask'])
        
        input_len = tokenized['input_ids'].size(1)
        generated_ids = outputs[:, input_len:]
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        attention_mask = (outputs.ne(self.tokenizer.pad_token_id)).long()
        action_mask = (generated_ids.ne(self.tokenizer.eos_token_id) & 
                       generated_ids.ne(self.tokenizer.pad_token_id)).long()
        
        samples = Samples(
            prompt_response_ids=outputs,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=action_mask.size(1),
            rewards=None,
            step_type='update',
            response_length=action_mask.float().sum(dim=-1)
        )
        
        return generated_texts, samples

    def inference(self, batch_data, **kwargs):
        return []

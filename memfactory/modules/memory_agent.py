import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from ..common.registry import MODULE_REGISTRY
from .base import BaseModule, Samples
from ..common.utils import TEMPLATE, TEMPLATE_FINAL_BOXED

@MODULE_REGISTRY.register("recurrent_memory_module")
class RecurrentMemoryModule(BaseModule):
    """
    A recurrent memory module that implements the iterative reading/updating loop logic.
    This corresponds to the core logic of the original MemoryAgent.
    """
    def __init__(self, tokenizer, device="cuda", **kwargs):
        super().__init__(tokenizer, device)
        self.chunk_size = kwargs.get("chunk_size", 2048)
        self.max_chunk_number = kwargs.get("max_chunk_number", 5)
        self.num_generations = kwargs.get("num_generations", 16)
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

    def rollout(self, model: Any, batch_data: Dict[str, Any], **kwargs) -> Optional[List[Tuple[str, str, float]]]:
        """
        Executes the recurrent memory loop.
        Returns raw trajectories: List of (prompt, response, advantage) tuples.
        Note: Advantage calculation requires reward_fn.
        """
        reward_fn = kwargs.get('reward_fn')
        if not reward_fn:
            # If no reward_fn, we can't compute advantages here.
            # But the original code structure mixed generation and eval.
            # We will follow that for now.
            raise ValueError("reward_fn is required for rollout")

        results = []
        bs = len(batch_data['context_ids'])
        
        for i in range(bs):
            context_ids = batch_data['context_ids'][i]
            question = batch_data['question'][i]
            ground_truth = batch_data['ground_truth'][i]
            
            total_length = len(context_ids)
            num_chunks = (total_length + self.chunk_size - 1) // self.chunk_size
            assert num_chunks <= self.max_chunk_number, f"num_chunks({num_chunks}) > max_chunk_number({self.max_chunk_number})"

            memories = ["No previous memory"] * self.num_generations
            trajectories = [[] for _ in range(self.num_generations)]
            
            # 1. Iterative Updates
            for step in range(num_chunks):
                start_idx = step * self.chunk_size
                end_idx = min((step + 1) * self.chunk_size, total_length)
                chunk_text = self.tokenizer.decode(context_ids[start_idx:end_idx], skip_special_tokens=True)
                
                step_prompts = [TEMPLATE.format(prompt=question, memory=memories[j], chunk=chunk_text) for j in range(self.num_generations)]
                
                msgs_list = [[{"role": "user", "content": p}] for p in step_prompts]
                formatted_prompts = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs_list]
                
                tokenized = self.tokenizer(formatted_prompts, padding=True, return_tensors='pt').to(self.device)
                outputs = self._generate_with_pytorch(model, tokenized['input_ids'], tokenized['attention_mask'])
                
                input_len = tokenized['input_ids'].size(1)
                generated_texts = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
                
                for j in range(self.num_generations):
                    response_text = generated_texts[j]
                    memories[j] = response_text
                    if "<think>" in response_text:
                         memories[j] = response_text.split("</think>")[-1].strip() if "</think>" in response_text else response_text[-100:].strip()
                    trajectories[j].append((formatted_prompts[j], response_text))

            # 2. Final Answer
            final_prompts = [TEMPLATE_FINAL_BOXED.format(prompt=question, memory=memories[j]) for j in range(self.num_generations)]
            msgs_list = [[{"role": "user", "content": p}] for p in final_prompts]
            formatted_final_prompts = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs_list]
            
            tokenized = self.tokenizer(formatted_final_prompts, padding=True, return_tensors='pt').to(self.device)
            outputs = self._generate_with_pytorch(model, tokenized['input_ids'], tokenized['attention_mask'])
            
            generated_texts = self.tokenizer.batch_decode(outputs[:, tokenized['input_ids'].size(1):], skip_special_tokens=True)
            
            # 3. Evaluation
            final_responses = []
            for j in range(self.num_generations):
                trajectories[j].append((formatted_final_prompts[j], generated_texts[j]))
                final_responses.append(generated_texts[j])
            
            scores = reward_fn(final_responses, [ground_truth] * self.num_generations, [question] * self.num_generations)
            
            # 4. GRPO Advantage Calculation
            scores_tensor = torch.tensor(scores, dtype=torch.float32, device=self.device)
            mean_score = scores_tensor.mean()
            std_score = scores_tensor.std()
            
            if std_score.item() < 1e-6:
                continue # Skip if no variance
                
            advantages = (scores_tensor - mean_score) / (std_score + 1e-8)
            
            for j in range(self.num_generations):
                for p, r in trajectories[j]:
                    results.append((p, r, advantages[j].item()))
                    
        return results

    def inference(self, batch_data: Dict[str, Any], **kwargs) -> List[str]:
        return []

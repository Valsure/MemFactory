import json
import os
import pandas as pd
from typing import List, Any
from ..common.registry import ENV_REGISTRY
from .base import BaseEnv
from ..common.utils import evaluate_memory_agent_batch

@ENV_REGISTRY.register("longcontext")
class LongContextMemoryEnv(BaseEnv):
    def __init__(self, data_path: str, tokenizer: Any, **kwargs):
        super().__init__(data_path, tokenizer)
        self.data = []
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.data_path):
            if self.data_path.endswith('.json'):
                try:
                    with open(self.data_path, 'r', encoding='utf-8') as f:
                        data_list = json.load(f)
                    for item in data_list:
                        context = item.get('context', '')
                        question = item.get('input', '')
                        answers = item.get('answers', [])
                        # Ensure answers is a list
                        if not isinstance(answers, list):
                            answers = [answers]
                            
                        # Pre-encode context as requested
                        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
                        
                        self.data.append({
                            'context_ids': context_ids,
                            'context': context,
                            'question': question,
                            'extra_info': {'question': question},
                            'prompt': question,
                            'ground_truth': answers[0] if answers else "",
                            'reward_model': {'ground_truth': answers}
                        })
                except Exception as e:
                    print(f"Error loading json file: {e}")
            else:
                try:
                    df = pd.read_parquet(self.data_path)
                    for _, row in df.iterrows():
                        context = row['context']
                        extra_info = row['extra_info']
                        prompt = row['prompt']
                        reward_model = row['reward_model']
                        
                        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
                        
                        self.data.append({
                            'context_ids': context_ids,
                            'context': context,
                            'question': extra_info.get('question', ''),
                            'extra_info': extra_info,
                            'prompt': prompt,
                            'ground_truth': reward_model.get('ground_truth', ''),
                            'reward_model': reward_model
                        })
                except Exception as e:
                    print(f"Error loading parquet file: {e}")
        else:
            print(f"Warning: {self.data_path} not found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def compute_reward(self, predictions: List[str], ground_truths: List[str], questions: List[str], **kwargs) -> List[float]:
        """
        Compute rewards using the LLM-based judge.
        """
        # kwargs might contain 'llm_client' or 'max_workers'
        return evaluate_memory_agent_batch(
            responses=predictions,
            ground_truths=ground_truths,
            questions=questions,
            max_workers=kwargs.get('max_workers', 16),
            llm_client=kwargs.get('llm_client', None)
        )

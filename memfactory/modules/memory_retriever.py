from typing import List, Any
from ..common.registry import MODULE_REGISTRY
from .base import BaseModule

@MODULE_REGISTRY.register("naive_retriever")
class NaiveRetriever(BaseModule):
    """
    Naive Retriever that wraps a store/search function.
    Not trainable.
    """
    def rollout(self, model, batch_data, **kwargs):
        return None

    def inference(self, batch_data, **kwargs):
        # This module doesn't generate text in the RL sense, 
        # it retrieves documents.
        pass
    
    def retrieve(self, query: str, store: Any, top_k: int = 3):
        # This mirrors the logic in mem_utils.MemoryEvaluator.retrieve
        # But here we assume 'store' is passed or managed by Env.
        # Actually, in this framework, retrieval is part of the Environment logic for Evaluation,
        # OR it's a tool used by the Agent.
        # Since the user asked for a Retriever Module, we provide it.
        if hasattr(store, 'search_similar'):
             results = store.search_similar(query, top_k=top_k)
             return [m for m, s in results]
        return []

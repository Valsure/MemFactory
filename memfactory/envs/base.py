from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List

class BaseEnv(Dataset, ABC):
    """
    Abstract Base Class for Memory Environments.
    Should handle data loading and reward computation.
    """
    def __init__(self, data_path: str, tokenizer: Any):
        self.data_path = data_path
        self.tokenizer = tokenizer

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def compute_reward(self, predictions: List[str], ground_truths: List[str], **kwargs) -> List[float]:
        """
        Compute rewards for a batch of predictions.
        """
        pass

    def collate_fn(self, batch):
        """
        Default collate function. Can be overridden.
        """
        if not batch:
            return {}
        keys = batch[0].keys()
        return {key: [d[key] for d in batch] for key in keys}

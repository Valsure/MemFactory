from dataclasses import dataclass
from typing import List, Optional, Union, Any, Dict, Tuple
import torch
from abc import ABC, abstractmethod

@dataclass
class Samples:
    """
    Standard output structure for RL training.
    """
    prompt_response_ids: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    rewards: Optional[torch.Tensor] = None
    
    # Metadata for debugging/logging
    prompt_length: Optional[torch.Tensor] = None
    response_length: Optional[torch.Tensor] = None
    step_type: str = 'extraction'

class BaseModule(ABC):
    """
    Abstract Base Class for Memory Modules.
    """
    def __init__(self, tokenizer: Any, device: str = "cuda", **kwargs):
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def rollout(self, model: Any, batch_data: Dict[str, Any], **kwargs) -> Union[Samples, Dict[str, Samples], None]:
        """
        Execute the policy with the given model to generate trajectories for training.
        Returns a Samples object (or Dict of Samples) containing tensors ready for loss computation.
        """
        pass

    @abstractmethod
    def inference(self, batch_data: Dict[str, Any], **kwargs) -> List[str]:
        """
        Execute the policy (possibly using an API) to get final answers.
        Returns a list of answer strings.
        """
        pass

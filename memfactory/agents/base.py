from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional
from ..modules.base import Samples

class BaseAgent(ABC):
    """
    Abstract Base Class for Agents.
    Agents orchestrate Modules to perform tasks.
    """
    def __init__(self, tokenizer: Any, device: str = "cuda", **kwargs):
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def rollout(self, model: Any, batch_data: Dict[str, Any], **kwargs) -> Union[Samples, Dict[str, Samples], None]:
        """
        Execute the agent's policy with the given model to generate trajectories for training.
        """
        pass

    @abstractmethod
    def inference(self, batch_data: Dict[str, Any], **kwargs) -> List[str]:
        """
        Execute the agent's policy to get final answers.
        """
        pass

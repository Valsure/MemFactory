from typing import Dict, Type, Any

class Registry:
    """
    A simple registry to map string names to classes.
    Usage:
        REGISTRY = Registry("MyRegistry")
        
        @REGISTRY.register("my_class")
        class MyClass: ...
        
        cls = REGISTRY.get("my_class")
    """
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type[Any]] = {}

    def register(self, name: str = None):
        def _register(cls):
            register_name = name if name else cls.__name__
            if register_name in self._registry:
                raise ValueError(f"Class {register_name} already registered in {self.name}")
            self._registry[register_name] = cls
            return cls
        return _register

    def get(self, name: str) -> Type[Any]:
        if name not in self._registry:
            raise ValueError(f"Class {name} not found in {self.name}. Available: {list(self._registry.keys())}")
        return self._registry[name]

# Global registries
ENV_REGISTRY = Registry("Environments")
MODULE_REGISTRY = Registry("Modules")
AGENT_REGISTRY = Registry("Agents")
TRAINER_REGISTRY = Registry("Trainers")

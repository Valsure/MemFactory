from .base import BaseModule

class MemoryExtractor(BaseModule):
    """Placeholder for Memory Extractor Module"""
    def rollout(self, model, batch_data, **kwargs): pass
    def inference(self, batch_data, **kwargs): pass

class MemoryUpdater(BaseModule):
    """Placeholder for Memory Updater Module"""
    def rollout(self, model, batch_data, **kwargs): pass
    def inference(self, batch_data, **kwargs): pass

class MemoryRetriever(BaseModule):
    """Placeholder for Memory Retriever Module"""
    def rollout(self, model, batch_data, **kwargs): pass
    def inference(self, batch_data, **kwargs): pass

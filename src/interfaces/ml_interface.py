"""ML Model Interface for dependency injection."""
from abc import ABC, abstractmethod
from typing import List, Any

class MLModelInterface(ABC):
    """Abstract interface for ML models."""
    
    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate text response."""
        pass
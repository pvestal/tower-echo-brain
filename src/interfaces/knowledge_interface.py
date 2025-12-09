"""Knowledge Manager Interface for dependency injection."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class KnowledgeManagerInterface(ABC):
    """Abstract interface for knowledge management."""
    
    @abstractmethod
    def store_knowledge(self, title: str, content: str, category: str) -> int:
        """Store knowledge article."""
        pass
    
    @abstractmethod
    def retrieve_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge."""
        pass
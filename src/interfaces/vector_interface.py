"""Vector Database Interface for dependency injection."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorDatabaseInterface(ABC):
    """Abstract interface for vector databases."""
    
    @abstractmethod
    def add_point(self, collection: str, text: str, metadata: Dict[str, Any]) -> None:
        """Add a point to the vector database."""
        pass
    
    @abstractmethod
    def search(self, collection: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass
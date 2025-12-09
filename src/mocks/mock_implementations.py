"""Mock implementations for testing without ML dependencies."""
from typing import List, Dict, Any
from src.interfaces.ml_interface import MLModelInterface
from src.interfaces.vector_interface import VectorDatabaseInterface
from src.interfaces.knowledge_interface import KnowledgeManagerInterface

class MockMLModel(MLModelInterface):
    """Mock ML model for testing."""
    
    def generate_embedding(self, text: str) -> List[float]:
        """Return fake embedding."""
        return [0.1] * 768  # Fake 768D embedding
    
    def generate_response(self, prompt: str) -> str:
        """Return mock response."""
        if "fix" in prompt.lower():
            return "Implementing fix for the issue"
        return "Mock response generated"

class MockVectorDatabase(VectorDatabaseInterface):
    """Mock vector database for testing."""
    
    def __init__(self):
        self.data = {}
    
    def add_point(self, collection: str, text: str, metadata: Dict[str, Any]) -> None:
        """Store point in memory."""
        if collection not in self.data:
            self.data[collection] = []
        self.data[collection].append({
            'text': text,
            'metadata': metadata
        })
    
    def search(self, collection: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search in memory storage."""
        if collection not in self.data:
            return []
        
        # Simple keyword matching for mock
        results = []
        query_lower = query.lower()
        for item in self.data[collection]:
            if any(word in item['text'].lower() for word in query_lower.split()):
                results.append(item)
                if len(results) >= limit:
                    break
        return results

class MockKnowledgeManager(KnowledgeManagerInterface):
    """Mock knowledge manager for testing."""
    
    def __init__(self):
        self.articles = []
    
    def store_knowledge(self, title: str, content: str, category: str) -> int:
        """Store article in memory."""
        article_id = len(self.articles) + 1
        self.articles.append({
            'id': article_id,
            'title': title,
            'content': content,
            'category': category
        })
        return article_id
    
    def retrieve_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve articles matching query."""
        results = []
        query_lower = query.lower()
        for article in self.articles:
            if query_lower in article['title'].lower() or query_lower in article['content'].lower():
                results.append(article)
        return results
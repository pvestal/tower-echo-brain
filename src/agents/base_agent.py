"""Base Agent class for all Echo Brain agents"""
import logging
import httpx
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Abstract base class for agents"""
    
    def __init__(self, name: str, model: str = "deepseek-r1:8b"):
        self.name = name
        self.model = model
        self.ollama_url = "http://localhost:11434"
        self.history: List[Dict] = []
    
    async def call_model(self, prompt: str, system: str = None) -> str:
        """Call Ollama model with prompt"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        if system:
            payload["system"] = system
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload
                )
                result = response.json()
                return result.get("response", "")
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            return f"Error: {e}"
    
    @abstractmethod
    async def process(self, task: str, context: Dict = None) -> Dict:
        """Process a task - implemented by subclasses"""
        pass
    
    def add_to_history(self, task: str, result: Dict):
        """Track agent history"""
        self.history.append({"task": task[:100], "result": result})
        # Keep last 50 entries
        if len(self.history) > 50:
            self.history = self.history[-50:]

"""
Board of Directors Integration for Echo Brain
Adds transparent decision-making to Echo's intelligence
"""

import aiohttp
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BoardIntegration:
    """
    Connects Echo to Board of Directors for transparent decision-making
    """
    
    def __init__(self, board_url: str = "http://127.0.0.1:8410"):
        self.board_url = board_url
        self.session = None
        logger.info(f"Board Integration initialized with {board_url}")
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    async def evaluate_with_board(self, code: str, context: Dict = None) -> Dict:
        """
        Send code to Board for evaluation before execution
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        task = {
            "task": {
                "id": f"echo-{datetime.now().timestamp()}",
                "task_type": "code_review",
                "code": code,
                "requirements": context.get("requirements", "") if context else ""
            },
            "context": context or {}
        }
        
        try:
            async with self.session.post(
                f"{self.board_url}/api/board/evaluate",
                json=task,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    decision = await resp.json()
                    logger.info(f"Board decision: {decision['final_recommendation']}")
                    return decision
                else:
                    logger.error(f"Board API error: {resp.status}")
                    return {"error": f"Board API returned {resp.status}"}
        except Exception as e:
            logger.error(f"Board connection error: {e}")
            return {"error": str(e)}
    
    def should_execute(self, board_decision: Dict) -> bool:
        """
        Determine if code should be executed based on Board decision
        """
        if "error" in board_decision:
            # Board unavailable, default to cautious
            return False
        
        recommendation = board_decision.get("final_recommendation", "reject")
        
        # Only execute if approved or modify (with caution)
        if recommendation == "reject":
            logger.warning("Board rejected code execution")
            return False
        
        # Check if Security Director objected
        for director in board_decision.get("directors", []):
            if director["name"] == "Security Director" and director["recommendation"] == "reject":
                if director["confidence"] >= 0.8:
                    logger.error(f"Security Director veto: {director['reasoning']}")
                    return False
        
        return True
    
    def get_board_reasoning(self, board_decision: Dict) -> str:
        """
        Extract human-readable reasoning from Board decision
        """
        if "error" in board_decision:
            return f"Board unavailable: {board_decision['error']}"
        
        reasoning = []
        for director in board_decision.get("directors", []):
            if director["recommendation"] != "approve":
                reasoning.append(f"{director['name']}: {director['reasoning'][:100]}")
        
        return " | ".join(reasoning) if reasoning else "All directors approve"

# Import datetime for timestamp
from datetime import datetime

# Global board instance
board = BoardIntegration()

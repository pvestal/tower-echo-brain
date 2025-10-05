#!/usr/bin/env python3
"""
Dynamic Escalation Manager
Replaces hard-coded model selection with persona-driven adaptive system
"""

import asyncio
import requests
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime
import sys
sys.path.append('/opt/tower-echo-brain/src')
from engines.persona_threshold_engine import PersonaThresholdEngine

logger = logging.getLogger(__name__)

class DynamicEscalationManager:
    """
    Adaptive model selection using persona + thresholds
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.engine = None
        self.response_history = []
        
    async def initialize(self):
        """Initialize persona engine"""
        self.engine = PersonaThresholdEngine()
        await self.engine.initialize()
        logger.info("Dynamic escalation manager initialized")
        
    async def process_message(self, message: str, context: Dict = None) -> Dict:
        """
        Process message with dynamic model selection
        Returns: {
            'tier': str,
            'model': str,
            'response': str,
            'processing_time': float,
            'success': bool
        }
        """
        start_time = datetime.now()
        
        # Get dynamic tier selection
        tier, config = await self.engine.select_tier(message, context)
        model = config.get('model_name', 'llama3.2:3b')
        timeout = config.get('timeout_seconds', 30)
        
        logger.info(f"Processing with tier='{tier}', model='{model}', timeout={timeout}s")
        
        # Query Ollama
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": message,
                    "stream": False
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                result_text = response.json().get("response", "")
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Calculate quality score (basic heuristic)
                quality = self._calculate_quality(result_text, message)
                success = len(result_text) > 50 and quality > 0.5
                
                # Provide feedback to persona engine
                await self.engine.update_from_feedback(tier, success, quality)
                
                result = {
                    'tier': tier,
                    'model': model,
                    'response': result_text,
                    'processing_time': processing_time,
                    'success': success,
                    'quality': quality,
                    'config': config
                }
                
                self.response_history.append(result)
                return result
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return self._fallback_response(tier, model)
                
        except requests.Timeout:
            logger.warning(f"Timeout with {model} after {timeout}s")
            return self._fallback_response(tier, model, timeout=True)
        except Exception as e:
            logger.error(f"Error with {model}: {e}")
            return self._fallback_response(tier, model)
            
    def _calculate_quality(self, response: str, question: str) -> float:
        """Calculate response quality (0-1)"""
        if not response:
            return 0.0
            
        # Basic quality metrics
        word_count = len(response.split())
        has_code = 'def ' in response or 'class ' in response
        addresses_question = any(word in response.lower() for word in question.lower().split()[:5])
        
        quality = 0.0
        if word_count > 100:
            quality += 0.3
        if word_count > 200:
            quality += 0.2
        if has_code and 'code' in question.lower():
            quality += 0.3
        if addresses_question:
            quality += 0.2
            
        return min(1.0, quality)
        
    def _fallback_response(self, tier: str, model: str, timeout: bool = False) -> Dict:
        """Fallback response when model fails"""
        return {
            'tier': tier,
            'model': model,
            'response': f"Model {model} {'timed out' if timeout else 'failed'}. Please try again.",
            'processing_time': 0.0,
            'success': False,
            'quality': 0.0,
            'error': 'timeout' if timeout else 'error'
        }
        
    async def close(self):
        """Cleanup"""
        if self.engine:
            await self.engine.close()

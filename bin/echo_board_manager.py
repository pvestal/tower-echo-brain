#!/usr/bin/env python3
"""
Echo Board Manager - Phase 1: Board Member Implementation
Adds board of directors functionality to Echo service
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional
import aiohttp
from datetime import datetime

logger = logging.getLogger(__name__)

class EchoBoardManager:
    def __init__(self):
        self.ollama_base_url = "http://localhost:11434/api"
        self.board_members = {
            'mistral': {'model': 'mistral:latest', 'weight': 1.0, 'specialty': 'creative'},
            'deepseek': {'model': 'deepseek-coder-v2:16b', 'weight': 1.0, 'specialty': 'technical'}, 
            'codellama': {'model': 'codellama:7b', 'weight': 1.0, 'specialty': 'patterns'},
            'echo': {'model': 'echo_enhanced_v1', 'weight': 1.0, 'specialty': 'context'}
        }
        self.learning_metrics = {
            'voting_accuracy': 0.0,
            'unique_contributions': 0.0,
            'user_satisfaction': 0.0,
            'decisions_participated': 0
        }
        
    async def analyze_complexity(self, request: str) -> int:
        """Analyze request complexity for routing decisions"""
        score = 0
        
        # Basic complexity indicators
        score += len(request.split()) * 0.5
        score += request.count('and') * 2
        score += len(request.split('.')) * 3
        
        # Domain complexity
        complex_terms = ['database', 'microservice', 'architecture', 'system', 'integration']
        for term in complex_terms:
            if term.lower() in request.lower():
                score += 15
                
        return min(int(score), 100)
    
    async def get_board_response(self, member: str, prompt: str) -> Dict:
        """Get response from a specific board member"""
        if member == 'echo':
            return {
                'member': 'echo',
                'response': 'As Echo, I bring context awareness and user preference learning to this decision.',
                'confidence': 0.85,
                'reasoning': 'Based on previous interactions and project context'
            }
            
        member_config = self.board_members.get(member)
        if not member_config:
            return {'error': f'Unknown board member: {member}'}
            
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    'model': member_config['model'],
                    'prompt': f"As a {member_config['specialty']} specialist: {prompt}",
                    'stream': False
                }
                
                async with session.post(f"{self.ollama_base_url}/generate", 
                                      json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            'member': member,
                            'response': result.get('response', ''),
                            'confidence': 0.8,  # Default confidence
                            'reasoning': f"{member_config['specialty']} analysis"
                        }
                    else:
                        return {'error': f'{member} unavailable'}
                        
        except Exception as e:
            logger.error(f"Error getting {member} response: {e}")
            return {'error': str(e)}
    
    async def convene_board(self, request: str) -> Dict:
        """Convene board of directors for complex decisions"""
        logger.info(f"Convening board for request: {request[:100]}...")
        
        # Get responses from all board members in parallel
        tasks = [self.get_board_response(member, request) for member in self.board_members.keys()]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process responses and determine consensus
        valid_responses = [r for r in responses if not isinstance(r, Exception) and 'error' not in r]
        
        if len(valid_responses) < 2:
            return {
                'decision': 'Insufficient board participation',
                'confidence': 0.1,
                'board_responses': responses
            }
            
        # Simple voting mechanism (can be enhanced later)
        consensus_threshold = 0.67
        avg_confidence = sum(r.get('confidence', 0) for r in valid_responses) / len(valid_responses)
        
        # Echo's learning: track participation
        self.learning_metrics['decisions_participated'] += 1
        
        return {
            'decision': 'Board consensus achieved' if avg_confidence > consensus_threshold else 'Board split decision',
            'confidence': avg_confidence,
            'board_responses': valid_responses,
            'complexity_routing': 'board_consensus',
            'timestamp': datetime.now().isoformat()
        }
    
    async def route_request(self, request: str) -> Dict:
        """Route request based on complexity analysis"""
        complexity = await self.analyze_complexity(request)
        
        if complexity < 30:
            return {
                'routing': 'simple',
                'handler': 'echo_direct',
                'complexity_score': complexity
            }
        elif complexity < 70:
            return {
                'routing': 'medium', 
                'handler': 'single_specialist',
                'complexity_score': complexity,
                'recommended_specialist': 'mistral'  # Default for medium complexity
            }
        else:
            # High complexity - convene full board
            board_result = await self.convene_board(request)
            board_result['routing'] = 'complex'
            board_result['handler'] = 'board_consensus'
            board_result['complexity_score'] = complexity
            return board_result
    
    def get_learning_metrics(self) -> Dict:
        """Get Echo's current learning progress"""
        return self.learning_metrics.copy()
        
    def update_learning_metric(self, metric: str, value: float):
        """Update Echo's learning progress"""
        if metric in self.learning_metrics:
            self.learning_metrics[metric] = value
            logger.info(f"Updated {metric} to {value}")

# Global board manager instance
board_manager = EchoBoardManager()

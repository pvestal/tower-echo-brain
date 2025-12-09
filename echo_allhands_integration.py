#!/usr/bin/env python3
"""
Echo AllHands Integration - Connects autonomous engine with Board of Directors
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import aiohttp
from pathlib import Path

logger = logging.getLogger(__name__)

class AllHandsIntegration:
    """
    Integrates Echo's autonomous engine with the Board of Directors system
    for transparent decision making and task delegation
    """
    
    def __init__(self):
        self.board_api_url = "http://localhost:8309/api/board"
        self.directors = {
            'Patrick': 'User/Owner - Final authority',
            'AI Assist': 'Primary AI Assistant',
            'Mistral': 'Creative reasoning (70B)',
            'DeepSeek': 'Code generation specialist',
            'Claude': 'Analysis and planning',
            'GPT': 'General knowledge',
            'Gemini': 'Multi-modal understanding'
        }
        self.active_tasks = {}
        self.decision_log = []
        
    async def submit_to_board(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Submit autonomous task to Board of Directors for approval"""
        try:
            # Prepare task for board review
            board_task = {
                'task_description': task.get('description', ''),
                'user_id': 'echo_autonomous',
                'priority': self._map_priority(task.get('priority', 3)),
                'context': {
                    'source': 'autonomous_engine',
                    'task_type': task.get('name', 'unknown'),
                    'autonomous_goal': task.get('goal', 'self_improvement'),
                    'estimated_impact': task.get('impact', 'low')
                }
            }
            
            # Submit to board API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.board_api_url}/submit-task",
                    json=board_task
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        logger.info(f"✅ Task submitted to Board: {result.get('task_id')}")
                        return result
                    else:
                        logger.error(f"❌ Board submission failed: {resp.status}")
                        return {'status': 'failed', 'error': f'HTTP {resp.status}'}
                        
        except Exception as e:
            logger.error(f"AllHands integration error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def get_board_decision(self, task_id: str) -> Dict[str, Any]:
        """Get board decision on a submitted task"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.board_api_url}/decision/{task_id}"
                ) as resp:
                    if resp.status == 200:
                        decision = await resp.json()
                        self.decision_log.append({
                            'task_id': task_id,
                            'decision': decision,
                            'timestamp': datetime.now().isoformat()
                        })
                        return decision
                    return {'status': 'pending'}
        except Exception as e:
            logger.error(f"Error getting board decision: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def delegate_to_director(self, director: str, task: Dict) -> Dict:
        """Delegate specific task to a director (AI model)"""
        delegation = {
            'director': director,
            'task': task,
            'delegated_at': datetime.now().isoformat()
        }
        
        if director == 'Mistral':
            # Use Mistral for creative tasks
            endpoint = 'http://localhost:11434/api/generate'
            model = 'mistral:70b'
        elif director == 'DeepSeek':
            # Use DeepSeek for code tasks
            endpoint = 'http://localhost:11434/api/generate'
            model = 'deepseek-coder:latest'
        elif director == 'Claude':
            # Use Claude API for analysis
            return await self._delegate_to_claude(task)
        else:
            # Default to local Ollama
            endpoint = 'http://localhost:11434/api/generate'
            model = 'llama3.2:latest'
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json={
                        'model': model,
                        'prompt': self._format_prompt_for_director(director, task),
                        'stream': False
                    }
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return {
                            'director': director,
                            'response': result.get('response', ''),
                            'success': True
                        }
        except Exception as e:
            logger.error(f"Delegation to {director} failed: {e}")
            return {'director': director, 'success': False, 'error': str(e)}
    
    async def coordinate_allhands_meeting(self, topic: str, participants: List[str]) -> Dict:
        """
        Coordinate an 'all hands' meeting where multiple directors collaborate
        """
        meeting = {
            'topic': topic,
            'participants': participants,
            'started_at': datetime.now().isoformat(),
            'contributions': {}
        }
        
        # Get input from each participant
        tasks = []
        for participant in participants:
            if participant in self.directors:
                task = {
                    'type': 'meeting_contribution',
                    'topic': topic,
                    'role': self.directors[participant]
                }
                tasks.append(self.delegate_to_director(participant, task))
        
        # Gather all responses
        responses = await asyncio.gather(*tasks)
        
        for i, participant in enumerate(participants):
            if i < len(responses):
                meeting['contributions'][participant] = responses[i]
        
        # Synthesize consensus
        meeting['consensus'] = await self._synthesize_consensus(meeting['contributions'])
        meeting['completed_at'] = datetime.now().isoformat()
        
        return meeting
    
    async def monitor_autonomous_decisions(self):
        """
        Monitor and log all autonomous decisions for transparency
        """
        monitoring_data = {
            'timestamp': datetime.now().isoformat(),
            'active_tasks': len(self.active_tasks),
            'recent_decisions': self.decision_log[-10:],
            'board_status': await self._check_board_status(),
            'director_availability': await self._check_director_availability()
        }
        
        # Save to file for transparency
        log_path = Path('/opt/tower-echo-brain/autonomous_decisions.json')
        with open(log_path, 'w') as f:
            json.dump(monitoring_data, f, indent=2)
        
        logger.info(f"📊 Monitoring: {len(self.active_tasks)} active tasks, {len(self.decision_log)} decisions logged")
        
        return monitoring_data
    
    def _map_priority(self, priority_value: int) -> str:
        """Map numeric priority to string"""
        mapping = {1: 'critical', 2: 'high', 3: 'normal', 4: 'low', 5: 'background'}
        return mapping.get(priority_value, 'normal')
    
    def _format_prompt_for_director(self, director: str, task: Dict) -> str:
        """Format task prompt for specific director"""
        base_prompt = f"As {director} on the Board of Directors, please help with: {task.get('topic', task)}"
        
        if director == 'Mistral':
            return f"{base_prompt}\n\nProvide creative and innovative solutions."
        elif director == 'DeepSeek':
            return f"{base_prompt}\n\nFocus on technical implementation and code quality."
        elif director == 'Claude':
            return f"{base_prompt}\n\nProvide thorough analysis and strategic planning."
        else:
            return base_prompt
    
    async def _delegate_to_claude(self, task: Dict) -> Dict:
        """Special handling for Claude delegation"""
        # Since we're running as Claude, we self-reflect
        return {
            'director': 'Claude',
            'response': f"Analyzing task: {task}. Recommendation: Proceed with structured approach.",
            'success': True,
            'note': 'Self-reflection mode'
        }
    
    async def _synthesize_consensus(self, contributions: Dict) -> str:
        """Synthesize consensus from multiple director inputs"""
        consensus_points = []
        
        for director, contribution in contributions.items():
            if contribution.get('success'):
                response = contribution.get('response', '')
                if response:
                    consensus_points.append(f"{director}: {response[:200]}...")
        
        if consensus_points:
            return "Board Consensus:\n" + "\n".join(consensus_points)
        return "No consensus reached"
    
    async def _check_board_status(self) -> str:
        """Check if board API is responsive"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.board_api_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    return 'active' if resp.status == 200 else 'degraded'
        except:
            return 'offline'
    
    async def _check_director_availability(self) -> Dict[str, bool]:
        """Check which directors (AI models) are available"""
        availability = {}
        
        # Check Ollama models
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:11434/api/tags') as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = [m['name'] for m in data.get('models', [])]
                        availability['Mistral'] = any('mistral' in m for m in models)
                        availability['DeepSeek'] = any('deepseek' in m for m in models)
        except:
            pass
        
        availability['AI Assist'] = True  # Echo is always available
        availability['Claude'] = True  # We are Claude
        availability['Patrick'] = False  # Human, async response
        
        return availability

async def test_allhands():
    """Test AllHands integration"""
    integration = AllHandsIntegration()
    
    # Test board submission
    test_task = {
        'name': 'System Optimization',
        'description': 'Analyze and optimize Tower service performance',
        'priority': 2,
        'goal': 'performance_improvement'
    }
    
    result = await integration.submit_to_board(test_task)
    print(f"Board submission: {result}")
    
    # Test all hands meeting
    meeting = await integration.coordinate_allhands_meeting(
        topic="How to improve Echo's autonomous capabilities?",
        participants=['AI Assist', 'Mistral', 'Claude']
    )
    print(f"AllHands meeting: {json.dumps(meeting, indent=2)}")
    
    # Monitor decisions
    monitoring = await integration.monitor_autonomous_decisions()
    print(f"Monitoring: {json.dumps(monitoring, indent=2)}")

if __name__ == "__main__":
    asyncio.run(test_allhands())

#!/usr/bin/env python3
"""
Echo Brain Board of Directors Integration
Connects the new Board of Directors system with Echo Brain's unified service
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from directors import (
    DirectorRegistry,
    SecurityDirector,
    QualityDirector,
    PerformanceDirector,
    EthicsDirector,
    UXDirector
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EchoBoardOfDirectors:
    """Integration layer between Echo Brain and Board of Directors"""

    def __init__(self):
        self.registry = DirectorRegistry()
        self._initialize_directors()
        logger.info("Echo Board of Directors initialized with 5 specialized directors")

    def _initialize_directors(self):
        """Initialize and register all specialized directors"""
        self.registry.register_director(SecurityDirector())
        self.registry.register_director(QualityDirector())
        self.registry.register_director(PerformanceDirector())
        self.registry.register_director(EthicsDirector())
        self.registry.register_director(UXDirector())

    async def evaluate_task(self, task: Dict) -> Dict:
        """
        Evaluate a task using the Board of Directors
        Returns consensus decision with detailed reasoning from each director
        """
        try:
            # Run evaluation (synchronous but can be wrapped in executor if needed)
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.registry.evaluate_task, task
            )

            # Process the result into Echo Brain format
            processed_result = self._process_board_decision(result, task)
            return processed_result

        except Exception as e:
            logger.error(f"Board evaluation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback': 'Using Echo Brain default evaluation'
            }

    def _process_board_decision(self, board_result: Dict, original_task: Dict) -> Dict:
        """Process board decision into Echo Brain compatible format"""

        # Extract evaluations
        evaluations = board_result.get('evaluations', [])

        # Calculate consensus
        if evaluations:
            avg_confidence = sum(e.get('confidence', 0) for e in evaluations) / len(evaluations)
            consensus_achieved = avg_confidence >= 60
        else:
            avg_confidence = 0
            consensus_achieved = False

        # Compile recommendations from all directors
        all_recommendations = []
        director_summaries = []

        for eval in evaluations:
            director_name = eval.get('director', 'Unknown')
            confidence = eval.get('confidence', 0)

            # Extract recommendations
            recommendations = eval.get('recommendations', [])
            for rec in recommendations:
                if isinstance(rec, dict):
                    all_recommendations.append({
                        'director': director_name,
                        'category': rec.get('category', 'general'),
                        'description': rec.get('description', ''),
                        'priority': rec.get('priority', 'medium'),
                        'effort': rec.get('effort', 'unknown')
                    })
                else:
                    all_recommendations.append({
                        'director': director_name,
                        'description': str(rec),
                        'priority': 'medium'
                    })

            # Create director summary
            director_summaries.append({
                'director': director_name,
                'confidence': confidence,
                'num_recommendations': len(recommendations),
                'reasoning': eval.get('reasoning', 'No detailed reasoning provided')
            })

        # Sort recommendations by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        all_recommendations.sort(
            key=lambda x: priority_order.get(x.get('priority', 'medium'), 2)
        )

        # Build final decision
        decision = "Board consensus achieved" if consensus_achieved else "Board requires further review"

        return {
            'success': True,
            'decision': decision,
            'consensus_achieved': consensus_achieved,
            'average_confidence': avg_confidence,
            'director_count': len(evaluations),
            'director_summaries': director_summaries,
            'recommendations': all_recommendations[:10],  # Top 10 recommendations
            'total_recommendations': len(all_recommendations),
            'task_type': original_task.get('task_type', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'board_composition': [
                'SecurityDirector',
                'QualityDirector',
                'PerformanceDirector',
                'EthicsDirector',
                'UXDirector'
            ]
        }

    def get_board_status(self) -> Dict:
        """Get current status of the Board of Directors"""
        analytics = self.registry.get_board_analytics()

        return {
            'active_directors': len(self.registry.directors),
            'total_evaluations': analytics.get('total_evaluations', 0),
            'consensus_rate': analytics.get('consensus_rate', 0),
            'average_response_time': analytics.get('average_response_time', 0),
            'directors': [
                {
                    'name': director.__class__.__name__,
                    'expertise': director.expertise,
                    'performance': director.get_performance_metrics()
                }
                for director in self.registry.directors.values()
            ]
        }

    def get_director_details(self, director_name: str) -> Dict:
        """Get detailed information about a specific director"""
        for director in self.registry.directors.values():
            if director.__class__.__name__ == director_name:
                return {
                    'name': director_name,
                    'expertise': director.expertise,
                    'knowledge_base': {
                        'best_practices': len(director.knowledge_base.get('best_practices', [])),
                        'anti_patterns': len(director.knowledge_base.get('anti_patterns', [])),
                        'risk_factors': len(director.knowledge_base.get('risk_factors', [])),
                        'optimization_strategies': len(director.knowledge_base.get('optimization_strategies', []))
                    },
                    'performance': director.get_performance_metrics()
                }

        return {'error': f'Director {director_name} not found'}


# Example integration test
async def test_board_integration():
    """Test the Board of Directors integration"""

    board = EchoBoardOfDirectors()

    # Test task with security vulnerability
    test_task = {
        'task_type': 'code_review',
        'description': 'Review authentication system for production deployment',
        'code': '''
def authenticate(username, password):
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    result = db.execute(query)

    # Hardcoded credentials
    if username == "admin" and password == "password123":
        return {"success": True, "admin": True}

    if result:
        # Missing proper session management
        return {"success": True, "user": result}

    return {"success": False}
''',
        'context': {
            'environment': 'production',
            'users': 50000,
            'sensitivity': 'high'
        }
    }

    # Evaluate with board
    print("Testing Board of Directors Integration...")
    print("=" * 60)

    result = await board.evaluate_task(test_task)

    if result['success']:
        print(f"✅ Decision: {result['decision']}")
        print(f"📊 Consensus: {result['consensus_achieved']} ({result['average_confidence']:.1f}% confidence)")
        print(f"👥 Directors: {result['director_count']} participated")

        print("\n📋 Director Summaries:")
        for summary in result['director_summaries']:
            print(f"  • {summary['director']}: {summary['confidence']:.1f}% confidence, {summary['num_recommendations']} recommendations")

        print(f"\n🎯 Top Recommendations (showing {min(5, len(result['recommendations']))} of {result['total_recommendations']}):")
        for i, rec in enumerate(result['recommendations'][:5], 1):
            print(f"  {i}. [{rec.get('priority', 'medium').upper()}] {rec['description'][:100]}...")
            print(f"     - From: {rec['director']}")

        print("\n📈 Board Status:")
        status = board.get_board_status()
        print(f"  Active Directors: {status['active_directors']}")
        print(f"  Total Evaluations: {status['total_evaluations']}")
    else:
        print(f"❌ Evaluation failed: {result.get('error', 'Unknown error')}")

    print("=" * 60)
    print("Integration test complete!")


if __name__ == "__main__":
    # Run integration test
    asyncio.run(test_board_integration())
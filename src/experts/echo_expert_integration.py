#!/usr/bin/env python3
"""
Integration module to connect Expert System with Echo Brain
"""
import sys
import os
sys.path.append('/opt/tower-echo-brain/src')

from src.experts.echo_expert_system import EchoExpertSystem, ExpertType
from typing import Dict, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

class EchoExpertIntegration:
    """Integration layer between Echo Brain and Expert System"""

    def __init__(self):
        self.expert_system = EchoExpertSystem()
        logger.info("🧠 Expert System initialized for Echo Brain")

    async def process_with_experts(self, context: Dict) -> Dict:
        """Process request through expert system"""
        try:
            # Add Echo-specific context
            context['source'] = 'echo_brain'
            context['timestamp'] = context.get('timestamp', '')

            # Get expert analysis
            analysis = await self.expert_system.analyze(context)

            # Add Echo formatting
            return self._format_for_echo(analysis)

        except Exception as e:
            logger.error(f"Expert system error: {e}")
            return {
                "status": "error",
                "message": str(e),
                "fallback": "continuing without expert analysis"
            }

    def _format_for_echo(self, analysis: Dict) -> Dict:
        """Format expert analysis for Echo's response format"""
        consensus = analysis.get('consensus', {})

        return {
            "expert_analysis": {
                "recommendation": consensus.get('recommendation', 'neutral'),
                "confidence": consensus.get('confidence', 0.5),
                "experts_consulted": analysis.get('experts_consulted', []),
                "reasoning": self._compile_reasoning(analysis),
                "warnings": self._extract_warnings(analysis)
            },
            "raw_analysis": analysis
        }

    def _compile_reasoning(self, analysis: Dict) -> str:
        """Compile expert reasoning into concise format"""
        reasons = []
        for expert_analysis in analysis.get('individual_analyses', []):
            if expert_analysis.get('confidence', 0) > 0.7:
                expert = expert_analysis.get('expert', 'Unknown')
                reasons.append(f"{expert}: High confidence")
        return " | ".join(reasons) if reasons else "Standard analysis"

    def _extract_warnings(self, analysis: Dict) -> list:
        """Extract any warnings from expert analyses"""
        warnings = []
        for expert_analysis in analysis.get('individual_analyses', []):
            if 'warning' in expert_analysis:
                warnings.append(expert_analysis['warning'])
            if expert_analysis.get('risk_level') in ['high', 'critical']:
                warnings.append(f"High risk detected by {expert_analysis.get('expert')}")
        return warnings

# Global instance
expert_integration = None

def get_expert_integration():
    """Get or create expert integration instance"""
    global expert_integration
    if expert_integration is None:
        expert_integration = EchoExpertIntegration()
    return expert_integration

# Integration with Echo's main loop
async def enhance_echo_decision(echo_context: Dict) -> Dict:
    """Enhance Echo's decision with expert analysis"""
    integration = get_expert_integration()

    # Get Echo's original decision
    echo_decision = echo_context.get('decision', {})

    # Get expert analysis
    expert_result = await integration.process_with_experts(echo_context)

    # Merge decisions
    enhanced_decision = {
        **echo_decision,
        'expert_enhanced': True,
        'expert_analysis': expert_result.get('expert_analysis'),
        'final_confidence': (
            echo_decision.get('confidence', 0.5) * 0.6 +
            expert_result.get('expert_analysis', {}).get('confidence', 0.5) * 0.4
        )
    }

    # Override if experts strongly disagree
    if expert_result.get('expert_analysis', {}).get('recommendation') == 'reject':
        if expert_result.get('expert_analysis', {}).get('confidence', 0) > 0.8:
            enhanced_decision['action'] = 'alert'
            enhanced_decision['requires_human_approval'] = True
            enhanced_decision['reason'] = 'Expert system strongly recommends against this action'

    return enhanced_decision

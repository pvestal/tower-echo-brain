"""
Board of Directors Module
Autonomous decision-making system for critical operations
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import random

logger = logging.getLogger(__name__)

@dataclass
class Decision:
    """Represents a board decision"""
    topic: str
    proposal: str
    votes_for: int
    votes_against: int
    abstentions: int
    approved: bool
    reasoning: List[str]
    timestamp: datetime

class BoardMember:
    """Represents a board member with specific expertise"""

    def __init__(self, name: str, role: str, expertise: List[str], risk_tolerance: float):
        self.name = name
        self.role = role
        self.expertise = expertise
        self.risk_tolerance = risk_tolerance  # 0.0 (conservative) to 1.0 (aggressive)

    async def vote(self, proposal: Dict[str, Any]) -> tuple[str, str]:
        """Vote on a proposal"""

        # Analyze proposal based on expertise
        relevance_score = self._calculate_relevance(proposal)
        risk_score = proposal.get('risk_level', 0.5)

        # Make decision based on risk tolerance and relevance
        if relevance_score < 0.3:
            # Not in area of expertise - abstain
            return 'abstain', f"{self.name}: Outside my expertise area"

        if risk_score > self.risk_tolerance:
            # Too risky
            return 'against', f"{self.name}: Risk level ({risk_score:.2f}) exceeds my tolerance ({self.risk_tolerance:.2f})"

        if relevance_score > 0.7 and risk_score < self.risk_tolerance * 0.7:
            # Strong match and acceptable risk
            return 'for', f"{self.name}: Strong alignment with {self.role} objectives"

        # Default to cautious approval
        if risk_score < 0.3:
            return 'for', f"{self.name}: Low risk, acceptable proposal"
        else:
            return 'against', f"{self.name}: Uncertain risk-benefit ratio"

    def _calculate_relevance(self, proposal: Dict[str, Any]) -> float:
        """Calculate how relevant a proposal is to this member's expertise"""

        proposal_text = f"{proposal.get('topic', '')} {proposal.get('description', '')}".lower()
        matches = sum(1 for exp in self.expertise if exp.lower() in proposal_text)
        return min(1.0, matches / max(1, len(self.expertise)))

class BoardOfDirectors:
    """Autonomous decision-making board"""

    def __init__(self):
        # Initialize board members with different perspectives
        self.members = [
            BoardMember(
                "Chief Autonomy Officer",
                "Autonomy",
                ["self-improvement", "automation", "efficiency"],
                0.8  # High risk tolerance
            ),
            BoardMember(
                "Chief Security Officer",
                "Security",
                ["security", "safety", "protection", "sandbox"],
                0.2  # Low risk tolerance
            ),
            BoardMember(
                "Chief Performance Officer",
                "Performance",
                ["optimization", "speed", "resource", "benchmark"],
                0.6  # Moderate risk tolerance
            ),
            BoardMember(
                "Chief Learning Officer",
                "Learning",
                ["training", "model", "knowledge", "memory"],
                0.7  # Moderate-high risk tolerance
            ),
            BoardMember(
                "Chief Integration Officer",
                "Integration",
                ["api", "integration", "compatibility", "interface"],
                0.5  # Balanced risk tolerance
            )
        ]

        self.decision_history = []

    async def deliberate(self, proposal: Dict[str, Any]) -> Decision:
        """Have the board deliberate on a proposal"""

        logger.info(f"Board deliberating on: {proposal.get('topic', 'Unknown')}")

        # Collect votes from all members
        votes = {'for': 0, 'against': 0, 'abstain': 0}
        reasoning = []

        for member in self.members:
            vote, reason = await member.vote(proposal)
            votes[vote] += 1
            reasoning.append(reason)
            logger.info(f"  {reason}")

        # Determine if approved (simple majority of non-abstaining votes)
        total_votes = votes['for'] + votes['against']
        approved = votes['for'] > votes['against'] if total_votes > 0 else False

        # Special veto for critical operations
        if proposal.get('critical', False) and votes['against'] > 0:
            # Any dissent on critical operations triggers veto
            approved = False
            reasoning.append("VETO: Critical operation requires unanimous approval")

        decision = Decision(
            topic=proposal.get('topic', 'Unknown'),
            proposal=proposal.get('description', ''),
            votes_for=votes['for'],
            votes_against=votes['against'],
            abstentions=votes['abstain'],
            approved=approved,
            reasoning=reasoning,
            timestamp=datetime.now()
        )

        self.decision_history.append(decision)

        logger.info(f"Board decision: {'APPROVED' if approved else 'REJECTED'} ({votes['for']}/{votes['against']}/{votes['abstain']})")

        return decision

    async def request_approval(
        self,
        topic: str,
        description: str,
        risk_level: float = 0.5,
        critical: bool = False,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Request board approval for an action"""

        proposal = {
            'topic': topic,
            'description': description,
            'risk_level': risk_level,
            'critical': critical,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }

        decision = await self.deliberate(proposal)

        return decision.approved

    def get_recent_decisions(self, limit: int = 10) -> List[Decision]:
        """Get recent board decisions"""
        return self.decision_history[-limit:]

    async def emergency_override(self, reason: str) -> bool:
        """Emergency override requiring special conditions"""

        logger.warning(f"EMERGENCY OVERRIDE REQUESTED: {reason}")

        # Require at least 4/5 members to approve emergency action
        emergency_proposal = {
            'topic': 'Emergency Override',
            'description': reason,
            'risk_level': 0.9,
            'critical': True
        }

        decision = await self.deliberate(emergency_proposal)

        if decision.votes_for >= 4:
            logger.warning("EMERGENCY OVERRIDE APPROVED")
            return True
        else:
            logger.warning("EMERGENCY OVERRIDE DENIED")
            return False

# Global board instance
board = BoardOfDirectors()
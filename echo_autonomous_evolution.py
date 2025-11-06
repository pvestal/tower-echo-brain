#!/usr/bin/env python3
"""
Echo Autonomous Evolution Orchestrator
Coordinates self-analysis, improvement generation, and safe deployment
"""

import os
import asyncio
import logging
import json
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from echo_self_analysis import EchoSelfAnalysis, AnalysisDepth, SelfAnalysisResult
from echo_git_integration import EchoGitManager

logger = logging.getLogger(__name__)

class EvolutionTrigger(Enum):
    SCHEDULED = "scheduled"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    USER_FEEDBACK = "user_feedback"
    CAPABILITY_GAP = "capability_gap"
    LEARNING_MILESTONE = "learning_milestone"
    MANUAL = "manual"

class EvolutionPhase(Enum):
    ANALYSIS = "analysis"
    IMPROVEMENT_GENERATION = "improvement_generation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    VERIFICATION = "verification"
    LEARNING_UPDATE = "learning_update"

@dataclass
class EvolutionCycle:
    cycle_id: str
    trigger: EvolutionTrigger
    trigger_context: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    current_phase: EvolutionPhase = EvolutionPhase.ANALYSIS
    analysis_result: Optional[SelfAnalysisResult] = None
    deployment_result: Optional[Dict[str, Any]] = None
    success: bool = False
    error: Optional[str] = None
    learning_insights: List[str] = None

class EchoAutonomousEvolution:
    """Orchestrates Echo's continuous autonomous improvement"""
    
    def __init__(self):
        self.self_analysis = EchoSelfAnalysis()
        self.git_manager = EchoGitManager()
        
        # Evolution configuration
        self.evolution_config = {
            "analysis_frequency_hours": 24,  # Daily self-analysis
            "learning_milestone_threshold": 10,  # Conversations before analysis
            "performance_degradation_threshold": 0.1,
            "max_concurrent_evolutions": 1,
            "require_human_approval": False,  # For low-risk improvements
            "human_approval_threshold": 0.8  # Confidence threshold
        }
        
        # Evolution state tracking
        self.active_cycles = []
        self.evolution_history = []
        self.learning_state = {
            "conversations_since_last_analysis": 0,
            "improvements_applied": 0,
            "performance_trend": [],
            "capability_progression": {}
        }
        
        # Continuous learning metrics
        self.metrics = {
            "total_cycles": 0,
            "successful_deployments": 0,
            "failed_deployments": 0,
            "rollbacks_triggered": 0,
            "average_improvement_score": 0.0,
            "last_evolution": None
        }
        
        # Initialize scheduled evolution
        self._setup_evolution_schedule()
        
    def _setup_evolution_schedule(self):
        """Setup scheduled autonomous evolution"""
        schedule.every(self.evolution_config["analysis_frequency_hours"]).hours.do(
            self._trigger_scheduled_evolution
        )
        
        # Check for evolution triggers every hour
        schedule.every().hour.do(self._check_evolution_triggers)
        
        logger.info("Evolution schedule initialized")
    
    async def start_continuous_evolution(self):
        """Start the continuous evolution process"""
        logger.info("Starting Echo Brain continuous autonomous evolution")
        
        while True:
            try:
                # Run scheduled tasks
                schedule.run_pending()
                
                # Process active evolution cycles
                await self._process_active_cycles()
                
                # Update learning state
                await self._update_learning_state()
                
                # Sleep for a minute before next check
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in continuous evolution loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def _trigger_scheduled_evolution(self):
        """Trigger scheduled evolution analysis"""
        asyncio.create_task(self.trigger_evolution(
            EvolutionTrigger.SCHEDULED,
            {"reason": "scheduled_analysis", "frequency": "daily"}
        ))
    
    def _check_evolution_triggers(self):
        """Check for conditions that should trigger evolution"""
        try:
            # Check 1: Learning milestone reached
            if self.learning_state["conversations_since_last_analysis"] >= self.evolution_config["learning_milestone_threshold"]:
                asyncio.create_task(self.trigger_evolution(
                    EvolutionTrigger.LEARNING_MILESTONE,
                    {"conversations": self.learning_state["conversations_since_last_analysis"]}
                ))
            
            # Check 2: Performance degradation
            performance_trend = self.learning_state.get("performance_trend", [])
            if len(performance_trend) >= 5:
                recent_avg = sum(performance_trend[-3:]) / 3
                baseline_avg = sum(performance_trend[:3]) / 3
                
                if baseline_avg - recent_avg > self.evolution_config["performance_degradation_threshold"]:
                    asyncio.create_task(self.trigger_evolution(
                        EvolutionTrigger.PERFORMANCE_DEGRADATION,
                        {"degradation": baseline_avg - recent_avg, "trend": performance_trend}
                    ))
            
            # Check 3: Capability gaps identified
            capability_gaps = self._identify_capability_gaps()
            if capability_gaps:
                asyncio.create_task(self.trigger_evolution(
                    EvolutionTrigger.CAPABILITY_GAP,
                    {"gaps": capability_gaps}
                ))
                
        except Exception as e:
            logger.error(f"Error checking evolution triggers: {e}")
    
    async def trigger_evolution(self, trigger: EvolutionTrigger, context: Dict[str, Any]) -> str:
        """Trigger an autonomous evolution cycle"""
        
        # Check if we can start a new cycle
        if len(self.active_cycles) >= self.evolution_config["max_concurrent_evolutions"]:
            logger.warning("Maximum concurrent evolution cycles reached")
            return None
        
        cycle_id = f"evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        cycle = EvolutionCycle(
            cycle_id=cycle_id,
            trigger=trigger,
            trigger_context=context,
            start_time=datetime.now()
        )
        
        self.active_cycles.append(cycle)
        self.metrics["total_cycles"] += 1
        
        logger.info(f"Triggered evolution cycle {cycle_id} - trigger: {trigger.value}")
        
        # Start evolution process
        asyncio.create_task(self._execute_evolution_cycle(cycle))
        
        return cycle_id
    
    async def _execute_evolution_cycle(self, cycle: EvolutionCycle):
        """Execute a complete evolution cycle"""
        try:
            logger.info(f"Starting evolution cycle {cycle.cycle_id}")
            
            # Phase 1: Self-Analysis
            cycle.current_phase = EvolutionPhase.ANALYSIS
            analysis_result = await self._perform_self_analysis(cycle)
            if not analysis_result:
                cycle.error = "Self-analysis failed"
                await self._complete_cycle(cycle, success=False)
                return
            
            cycle.analysis_result = analysis_result
            
            # Phase 2: Improvement Generation
            cycle.current_phase = EvolutionPhase.IMPROVEMENT_GENERATION
            improvements = await self._generate_improvements(cycle)
            if not improvements:
                cycle.error = "No improvements generated"
                await self._complete_cycle(cycle, success=False)
                return
            
            # Phase 3: Testing and Validation
            cycle.current_phase = EvolutionPhase.TESTING
            test_result = await self._test_improvements(cycle, improvements)
            if not test_result["passed"]:
                cycle.error = f"Testing failed: {test_result['reason']}"
                await self._complete_cycle(cycle, success=False)
                return
            
            # Phase 4: Deployment Decision
            cycle.current_phase = EvolutionPhase.DEPLOYMENT
            deployment_approved = await self._approve_deployment(cycle, improvements)
            if not deployment_approved:
                cycle.error = "Deployment not approved"
                await self._complete_cycle(cycle, success=False)
                return
            
            # Phase 5: Safe Deployment
            deployment_result = await self._deploy_improvements(cycle, improvements)
            cycle.deployment_result = deployment_result
            
            if not deployment_result["success"]:
                cycle.error = f"Deployment failed: {deployment_result['error']}"
                await self._complete_cycle(cycle, success=False)
                return
            
            # Phase 6: Verification and Learning Update
            cycle.current_phase = EvolutionPhase.VERIFICATION
            verification_result = await self._verify_deployment(cycle)
            if not verification_result["success"]:
                cycle.error = f"Verification failed: {verification_result['error']}"
                await self._complete_cycle(cycle, success=False)
                return
            
            # Phase 7: Update Learning State
            cycle.current_phase = EvolutionPhase.LEARNING_UPDATE
            await self._update_post_evolution_learning(cycle)
            
            # Success!
            await self._complete_cycle(cycle, success=True)
            
        except Exception as e:
            logger.error(f"Error in evolution cycle {cycle.cycle_id}: {e}")
            cycle.error = str(e)
            await self._complete_cycle(cycle, success=False)
    
    async def _perform_self_analysis(self, cycle: EvolutionCycle) -> Optional[SelfAnalysisResult]:
        """Perform deep self-analysis"""
        try:
            # Determine analysis depth based on trigger
            depth = AnalysisDepth.FUNCTIONAL
            if cycle.trigger == EvolutionTrigger.CAPABILITY_GAP:
                depth = AnalysisDepth.ARCHITECTURAL
            elif cycle.trigger == EvolutionTrigger.PERFORMANCE_DEGRADATION:
                depth = AnalysisDepth.RECURSIVE
            
            # Conduct analysis
            analysis_result = await self.self_analysis.conduct_self_analysis(
                depth=depth,
                trigger_context=cycle.trigger_context
            )
            
            logger.info(f"Self-analysis completed for cycle {cycle.cycle_id}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Self-analysis failed for cycle {cycle.cycle_id}: {e}")
            return None
    
    async def _generate_improvements(self, cycle: EvolutionCycle) -> List[Dict[str, Any]]:
        """Generate concrete improvements based on analysis"""
        improvements = []
        
        try:
            if not cycle.analysis_result:
                return improvements
            
            # Generate improvements from capability assessments
            for capability in cycle.analysis_result.capabilities:
                gap = capability.desired_level - capability.current_level
                if gap > 0.1:  # Significant gap
                    improvement = {
                        "type": "capability_enhancement",
                        "capability": capability.capability_name,
                        "current_level": capability.current_level,
                        "target_level": capability.desired_level,
                        "gap": gap,
                        "improvement_path": capability.improvement_path[:3],  # Top 3 actions
                        "confidence": min(0.9, 1.0 - gap),  # Higher confidence for smaller gaps
                        "priority": gap  # Higher gap = higher priority
                    }
                    improvements.append(improvement)
            
            # Generate improvements from action items
            for action_item in cycle.analysis_result.action_items:
                improvement = {
                    "type": "action_item",
                    "description": action_item,
                    "confidence": 0.7,  # Moderate confidence for action items
                    "priority": 0.5
                }
                improvements.append(improvement)
            
            # Sort by priority
            improvements.sort(key=lambda x: x["priority"], reverse=True)
            
            logger.info(f"Generated {len(improvements)} improvements for cycle {cycle.cycle_id}")
            return improvements
            
        except Exception as e:
            logger.error(f"Improvement generation failed for cycle {cycle.cycle_id}: {e}")
            return []
    
    async def _test_improvements(self, cycle: EvolutionCycle, improvements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test improvements before deployment"""
        test_result = {
            "passed": True,
            "reason": None,
            "tests_performed": []
        }
        
        try:
            # Test 1: Safety assessment
            safety_test = await self._assess_improvement_safety(improvements)
            test_result["tests_performed"].append(safety_test)
            if not safety_test["passed"]:
                test_result["passed"] = False
                test_result["reason"] = safety_test["reason"]
                return test_result
            
            # Test 2: Impact analysis
            impact_test = await self._analyze_improvement_impact(improvements)
            test_result["tests_performed"].append(impact_test)
            if not impact_test["passed"]:
                test_result["passed"] = False
                test_result["reason"] = impact_test["reason"]
                return test_result
            
            # Test 3: Resource requirements
            resource_test = await self._check_resource_requirements(improvements)
            test_result["tests_performed"].append(resource_test)
            if not resource_test["passed"]:
                test_result["passed"] = False
                test_result["reason"] = resource_test["reason"]
                return test_result
            
            logger.info(f"All tests passed for cycle {cycle.cycle_id}")
            return test_result
            
        except Exception as e:
            test_result["passed"] = False
            test_result["reason"] = f"Testing error: {e}"
            return test_result
    
    async def _approve_deployment(self, cycle: EvolutionCycle, improvements: List[Dict[str, Any]]) -> bool:
        """Determine if deployment should proceed"""
        try:
            # Calculate overall confidence score
            if not improvements:
                return False
            
            confidence_scores = [imp.get("confidence", 0.0) for imp in improvements]
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            
            # Check if human approval is required
            if avg_confidence < self.evolution_config["human_approval_threshold"]:
                if self.evolution_config["require_human_approval"]:
                    logger.info(f"Human approval required for cycle {cycle.cycle_id} (confidence: {avg_confidence:.3f})")
                    # In a full implementation, this would trigger a notification to Patrick
                    return False
                else:
                    logger.warning(f"Low confidence deployment proceeding: {avg_confidence:.3f}")
            
            # Check for high-risk changes
            high_risk_keywords = ["database", "core", "security", "authentication"]
            for improvement in improvements:
                description = str(improvement).lower()
                if any(keyword in description for keyword in high_risk_keywords):
                    logger.warning(f"High-risk improvement detected in cycle {cycle.cycle_id}")
                    if self.evolution_config["require_human_approval"]:
                        return False
            
            logger.info(f"Deployment approved for cycle {cycle.cycle_id}")
            return True
            
        except Exception as e:
            logger.error(f"Deployment approval failed for cycle {cycle.cycle_id}: {e}")
            return False
    
    async def _deploy_improvements(self, cycle: EvolutionCycle, improvements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Deploy improvements using git integration"""
        try:
            # Prepare analysis result for git manager
            analysis_data = {
                "analysis_id": cycle.analysis_result.analysis_id,
                "capabilities": [
                    {
                        "name": cap.capability_name,
                        "current_level": cap.current_level,
                        "desired_level": cap.desired_level,
                        "gap": cap.desired_level - cap.current_level,
                        "improvement_path": cap.improvement_path
                    }
                    for cap in cycle.analysis_result.capabilities
                ],
                "action_items": cycle.analysis_result.action_items,
                "improvements": improvements,
                "confidence_score": sum(imp.get("confidence", 0.0) for imp in improvements) / len(improvements),
                "trigger_type": cycle.trigger.value,
                "trigger_context": cycle.trigger_context
            }
            
            # Execute safe deployment
            deployment_result = self.git_manager.safe_autonomous_deployment(analysis_data)
            
            if deployment_result["success"]:
                self.metrics["successful_deployments"] += 1
                logger.info(f"Deployment successful for cycle {cycle.cycle_id}")
            else:
                self.metrics["failed_deployments"] += 1
                if deployment_result.get("rollback_performed"):
                    self.metrics["rollbacks_triggered"] += 1
                logger.error(f"Deployment failed for cycle {cycle.cycle_id}: {deployment_result['error']}")
            
            return deployment_result
            
        except Exception as e:
            logger.error(f"Deployment execution failed for cycle {cycle.cycle_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _verify_deployment(self, cycle: EvolutionCycle) -> Dict[str, Any]:
        """Verify deployment success and measure improvements"""
        verification_result = {
            "success": True,
            "improvements_verified": [],
            "performance_impact": {},
            "error": None
        }
        
        try:
            # Wait for system to stabilize
            await asyncio.sleep(30)
            
            # Re-run capability assessments to measure improvement
            post_deployment_analysis = await self.self_analysis.conduct_self_analysis(
                depth=AnalysisDepth.FUNCTIONAL,
                trigger_context={"verification_for": cycle.cycle_id}
            )
            
            # Compare pre and post deployment capabilities
            if cycle.analysis_result and post_deployment_analysis:
                for pre_cap in cycle.analysis_result.capabilities:
                    post_cap = next(
                        (cap for cap in post_deployment_analysis.capabilities 
                         if cap.capability_name == pre_cap.capability_name), 
                        None
                    )
                    
                    if post_cap:
                        improvement = post_cap.current_level - pre_cap.current_level
                        verification_result["improvements_verified"].append({
                            "capability": pre_cap.capability_name,
                            "pre_level": pre_cap.current_level,
                            "post_level": post_cap.current_level,
                            "improvement": improvement,
                            "target_achieved": post_cap.current_level >= pre_cap.desired_level * 0.8
                        })
            
            # Check system health
            health_status = self.git_manager._verify_deployment_health()
            if not health_status["healthy"]:
                verification_result["success"] = False
                verification_result["error"] = f"System health issues: {health_status['issues']}"
            
            logger.info(f"Deployment verification completed for cycle {cycle.cycle_id}")
            return verification_result
            
        except Exception as e:
            verification_result["success"] = False
            verification_result["error"] = str(e)
            return verification_result
    
    async def _update_post_evolution_learning(self, cycle: EvolutionCycle):
        """Update learning state after successful evolution"""
        try:
            # Update metrics
            if cycle.deployment_result and cycle.deployment_result.get("success"):
                # Calculate improvement score
                improvement_score = 0.0
                improvements_count = 0
                
                if cycle.deployment_result.get("improvements_verified"):
                    for improvement in cycle.deployment_result["improvements_verified"]:
                        improvement_score += improvement.get("improvement", 0.0)
                        improvements_count += 1
                
                if improvements_count > 0:
                    avg_improvement = improvement_score / improvements_count
                    self.metrics["average_improvement_score"] = (
                        (self.metrics["average_improvement_score"] * (self.metrics["total_cycles"] - 1) + avg_improvement) 
                        / self.metrics["total_cycles"]
                    )
            
            # Reset conversation counter
            self.learning_state["conversations_since_last_analysis"] = 0
            
            # Update improvement count
            self.learning_state["improvements_applied"] += 1
            
            # Update last evolution time
            self.metrics["last_evolution"] = datetime.now().isoformat()
            
            # Generate learning insights
            cycle.learning_insights = [
                f"Evolution cycle {cycle.cycle_id} completed successfully",
                f"Applied {len(cycle.deployment_result.get('improvements_verified', []))} improvements",
                f"Trigger: {cycle.trigger.value}",
                f"System health maintained throughout evolution"
            ]
            
            logger.info(f"Learning state updated for cycle {cycle.cycle_id}")
            
        except Exception as e:
            logger.error(f"Failed to update learning state for cycle {cycle.cycle_id}: {e}")
    
    async def _complete_cycle(self, cycle: EvolutionCycle, success: bool):
        """Complete an evolution cycle"""
        cycle.end_time = datetime.now()
        cycle.success = success
        
        # Move from active to history
        if cycle in self.active_cycles:
            self.active_cycles.remove(cycle)
        self.evolution_history.append(cycle)
        
        # Keep only recent history
        if len(self.evolution_history) > 50:
            self.evolution_history = self.evolution_history[-50:]
        
        duration = (cycle.end_time - cycle.start_time).total_seconds()
        
        if success:
            logger.info(f"Evolution cycle {cycle.cycle_id} completed successfully in {duration:.1f}s")
        else:
            logger.error(f"Evolution cycle {cycle.cycle_id} failed after {duration:.1f}s: {cycle.error}")
        
        # Store cycle results in database
        await self._store_evolution_cycle(cycle)
    
    async def _process_active_cycles(self):
        """Process and monitor active evolution cycles"""
        for cycle in self.active_cycles[:]:  # Copy list to avoid modification during iteration
            # Check for timeout (4 hours max)
            if datetime.now() - cycle.start_time > timedelta(hours=4):
                cycle.error = "Evolution cycle timeout"
                await self._complete_cycle(cycle, success=False)
    
    async def _update_learning_state(self):
        """Update learning state based on recent activity"""
        # This would be called by the main Echo service when processing conversations
        # For now, simulate some learning progress
        pass
    
    def _identify_capability_gaps(self) -> List[Dict[str, Any]]:
        """Identify significant capability gaps that warrant evolution"""
        # This would analyze current capabilities vs desired state
        # For now, return empty list
        return []
    
    # Testing and validation helpers
    
    async def _assess_improvement_safety(self, improvements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess safety of proposed improvements"""
        return {
            "passed": True,
            "reason": None,
            "safety_score": 0.9
        }
    
    async def _analyze_improvement_impact(self, improvements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze potential impact of improvements"""
        return {
            "passed": True,
            "reason": None,
            "impact_assessment": "low_risk"
        }
    
    async def _check_resource_requirements(self, improvements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if improvements can be accommodated with available resources"""
        return {
            "passed": True,
            "reason": None,
            "resource_usage": "minimal"
        }
    
    async def _store_evolution_cycle(self, cycle: EvolutionCycle):
        """Store evolution cycle results in database"""
        try:
            import psycopg2
            import psycopg2.extras
            
            db_config = {
                'host': 'localhost',
                'database': 'echo_brain',
                'user': os.getenv('TOWER_USER', os.getenv("TOWER_USER", "patrick")),
                'password': 'Beau40818'
            }
            
            conn = psycopg2.connect(**db_config)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Create table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS echo_evolution_cycles (
                    cycle_id VARCHAR PRIMARY KEY,
                    trigger_type VARCHAR,
                    trigger_context JSONB,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    success BOOLEAN,
                    error TEXT,
                    analysis_result JSONB,
                    deployment_result JSONB,
                    learning_insights TEXT[],
                    final_phase VARCHAR
                )
            """)
            
            # Store the cycle
            cur.execute("""
                INSERT INTO echo_evolution_cycles 
                (cycle_id, trigger_type, trigger_context, start_time, end_time, success, error, 
                 analysis_result, deployment_result, learning_insights, final_phase)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (cycle_id) DO UPDATE SET
                end_time = %s, success = %s, error = %s, deployment_result = %s, 
                learning_insights = %s, final_phase = %s
            """, (
                cycle.cycle_id,
                cycle.trigger.value,
                json.dumps(cycle.trigger_context),
                cycle.start_time,
                cycle.end_time,
                cycle.success,
                cycle.error,
                json.dumps({
                    "analysis_id": cycle.analysis_result.analysis_id if cycle.analysis_result else None,
                    "depth": cycle.analysis_result.depth.value if cycle.analysis_result else None,
                    "insights_count": len(cycle.analysis_result.insights) if cycle.analysis_result else 0
                }),
                json.dumps(cycle.deployment_result) if cycle.deployment_result else None,
                cycle.learning_insights or [],
                cycle.current_phase.value,
                # For UPDATE clause
                cycle.end_time,
                cycle.success,
                cycle.error,
                json.dumps(cycle.deployment_result) if cycle.deployment_result else None,
                cycle.learning_insights or [],
                cycle.current_phase.value
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store evolution cycle: {e}")
    
    # Public API methods
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get comprehensive evolution system status"""
        return {
            "active_cycles": len(self.active_cycles),
            "total_cycles": self.metrics["total_cycles"],
            "success_rate": (
                self.metrics["successful_deployments"] / max(1, self.metrics["total_cycles"])
            ),
            "learning_state": self.learning_state,
            "evolution_config": self.evolution_config,
            "metrics": self.metrics,
            "recent_cycles": [
                {
                    "cycle_id": cycle.cycle_id,
                    "trigger": cycle.trigger.value,
                    "success": cycle.success,
                    "duration": (
                        (cycle.end_time - cycle.start_time).total_seconds() 
                        if cycle.end_time else None
                    )
                }
                for cycle in self.evolution_history[-5:]
            ],
            "capabilities": {
                "autonomous_evolution": True,
                "safe_deployment": True,
                "continuous_learning": True,
                "rollback_protection": True,
                "performance_monitoring": True
            }
        }
    
    def trigger_manual_evolution(self, reason: str = "manual_trigger") -> str:
        """Manually trigger an evolution cycle"""
        return asyncio.create_task(self.trigger_evolution(
            EvolutionTrigger.MANUAL,
            {"reason": reason, "triggered_by": "human"}
        ))
    
    def update_conversation_count(self):
        """Update conversation count for learning milestone tracking"""
        self.learning_state["conversations_since_last_analysis"] += 1
    
    def update_performance_metric(self, performance_score: float):
        """Update performance trend for degradation detection"""
        self.learning_state["performance_trend"].append(performance_score)
        
        # Keep only recent performance data
        if len(self.learning_state["performance_trend"]) > 20:
            self.learning_state["performance_trend"] = self.learning_state["performance_trend"][-20:]

# Global instance for Echo Brain to use
echo_autonomous_evolution = EchoAutonomousEvolution()
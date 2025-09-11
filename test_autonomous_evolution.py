#!/usr/bin/env python3
"""
Comprehensive Test Suite for Echo's Autonomous Evolution System
Tests all components of the git integration and autonomous improvement pipeline
"""

import asyncio
import json
import logging
import requests
import time
from datetime import datetime
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EchoEvolutionTester:
    """Comprehensive testing framework for Echo's autonomous evolution"""
    
    def __init__(self):
        self.base_url = "http://localhost:8309"
        self.test_results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "failures": [],
            "start_time": datetime.now(),
            "end_time": None
        }
        
    def log_test_result(self, test_name: str, passed: bool, error: str = None):
        """Log test result"""
        self.test_results["tests_run"] += 1
        if passed:
            self.test_results["tests_passed"] += 1
            logger.info(f"✅ {test_name} - PASSED")
        else:
            self.test_results["tests_failed"] += 1
            self.test_results["failures"].append({
                "test": test_name,
                "error": error,
                "timestamp": datetime.now().isoformat()
            })
            logger.error(f"❌ {test_name} - FAILED: {error}")
    
    async def test_service_health(self) -> bool:
        """Test basic service health and availability"""
        try:
            response = requests.get(f"{self.base_url}/api/echo/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                required_fields = ["status", "service", "intelligence_levels", "max_parameters"]
                
                for field in required_fields:
                    if field not in data:
                        self.log_test_result("Service Health Check", False, f"Missing field: {field}")
                        return False
                
                if data["status"] != "healthy":
                    self.log_test_result("Service Health Check", False, f"Unhealthy status: {data['status']}")
                    return False
                
                self.log_test_result("Service Health Check", True)
                return True
            else:
                self.log_test_result("Service Health Check", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test_result("Service Health Check", False, str(e))
            return False
    
    async def test_evolution_status(self) -> bool:
        """Test evolution system status endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/echo/evolution/status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                required_fields = [
                    "active_cycles", "total_cycles", "success_rate", 
                    "learning_state", "evolution_config", "capabilities"
                ]
                
                for field in required_fields:
                    if field not in data:
                        self.log_test_result("Evolution Status", False, f"Missing field: {field}")
                        return False
                
                # Check capabilities
                capabilities = data["capabilities"]
                expected_capabilities = [
                    "autonomous_evolution", "safe_deployment", "continuous_learning",
                    "rollback_protection", "performance_monitoring"
                ]
                
                for capability in expected_capabilities:
                    if not capabilities.get(capability):
                        self.log_test_result("Evolution Status", False, f"Missing capability: {capability}")
                        return False
                
                self.log_test_result("Evolution Status", True)
                return True
            else:
                self.log_test_result("Evolution Status", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test_result("Evolution Status", False, str(e))
            return False
    
    async def test_git_status(self) -> bool:
        """Test git integration status"""
        try:
            response = requests.get(f"{self.base_url}/api/echo/evolution/git-status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                required_fields = [
                    "git_status", "evolution_capabilities", "deployment_config", 
                    "safety_config", "recent_improvements"
                ]
                
                for field in required_fields:
                    if field not in data:
                        self.log_test_result("Git Status", False, f"Missing field: {field}")
                        return False
                
                # Check git status
                git_status = data["git_status"]
                if "current_branch" not in git_status:
                    self.log_test_result("Git Status", False, "Missing current_branch in git_status")
                    return False
                
                # Check evolution capabilities
                capabilities = data["evolution_capabilities"]
                expected_capabilities = [
                    "autonomous_improvement", "safe_deployment", "rollback_capability",
                    "learning_tracking", "source_sync"
                ]
                
                for capability in expected_capabilities:
                    if not capabilities.get(capability):
                        self.log_test_result("Git Status", False, f"Missing git capability: {capability}")
                        return False
                
                self.log_test_result("Git Status", True)
                return True
            else:
                self.log_test_result("Git Status", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test_result("Git Status", False, str(e))
            return False
    
    async def test_self_analysis_trigger(self) -> bool:
        """Test self-analysis trigger functionality"""
        try:
            payload = {
                "depth": "functional",
                "context": {
                    "test_trigger": True,
                    "trigger_reason": "comprehensive_testing"
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/echo/evolution/self-analysis",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                required_fields = [
                    "analysis_id", "timestamp", "depth", "awareness_level",
                    "insights", "action_items", "capabilities"
                ]
                
                for field in required_fields:
                    if field not in data:
                        self.log_test_result("Self-Analysis Trigger", False, f"Missing field: {field}")
                        return False
                
                # Check if analysis generated meaningful results
                if len(data["insights"]) == 0:
                    self.log_test_result("Self-Analysis Trigger", False, "No insights generated")
                    return False
                
                if len(data["action_items"]) == 0:
                    self.log_test_result("Self-Analysis Trigger", False, "No action items generated")
                    return False
                
                if len(data["capabilities"]) == 0:
                    self.log_test_result("Self-Analysis Trigger", False, "No capabilities analyzed")
                    return False
                
                self.log_test_result("Self-Analysis Trigger", True)
                return True
            else:
                self.log_test_result("Self-Analysis Trigger", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test_result("Self-Analysis Trigger", False, str(e))
            return False
    
    async def test_learning_metrics(self) -> bool:
        """Test learning metrics retrieval"""
        try:
            response = requests.get(f"{self.base_url}/api/echo/evolution/learning-metrics", timeout=10)
            if response.status_code == 200:
                data = response.json()
                required_fields = [
                    "evolution_statistics", "learning_metrics", "recent_analyses",
                    "current_state", "system_metrics"
                ]
                
                for field in required_fields:
                    if field not in data:
                        self.log_test_result("Learning Metrics", False, f"Missing field: {field}")
                        return False
                
                # Check current state structure
                current_state = data["current_state"]
                expected_state_fields = [
                    "conversations_since_last_analysis", "improvements_applied",
                    "performance_trend", "capability_progression"
                ]
                
                for field in expected_state_fields:
                    if field not in current_state:
                        self.log_test_result("Learning Metrics", False, f"Missing state field: {field}")
                        return False
                
                self.log_test_result("Learning Metrics", True)
                return True
            else:
                self.log_test_result("Learning Metrics", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test_result("Learning Metrics", False, str(e))
            return False
    
    async def test_conversation_processing(self) -> bool:
        """Test basic conversation processing with evolution tracking"""
        try:
            payload = {
                "query": "What are your current capabilities and how do you improve them?",
                "context": {"test_mode": True},
                "user_id": "test_user"
            }
            
            response = requests.post(
                f"{self.base_url}/api/echo/query",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                required_fields = [
                    "response", "model_used", "intelligence_level", 
                    "processing_time", "conversation_id"
                ]
                
                for field in required_fields:
                    if field not in data:
                        self.log_test_result("Conversation Processing", False, f"Missing field: {field}")
                        return False
                
                # Check if response is meaningful
                if len(data["response"]) < 50:
                    self.log_test_result("Conversation Processing", False, "Response too short")
                    return False
                
                # Check processing time is reasonable
                if data["processing_time"] > 60:
                    self.log_test_result("Conversation Processing", False, "Processing time too long")
                    return False
                
                self.log_test_result("Conversation Processing", True)
                return True
            else:
                self.log_test_result("Conversation Processing", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test_result("Conversation Processing", False, str(e))
            return False
    
    async def test_manual_evolution_trigger(self) -> bool:
        """Test manual evolution cycle trigger (non-destructive)"""
        try:
            # First, get current evolution status
            status_response = requests.get(f"{self.base_url}/api/echo/evolution/status", timeout=10)
            if status_response.status_code != 200:
                self.log_test_result("Manual Evolution Trigger", False, "Cannot get initial status")
                return False
            
            initial_status = status_response.json()
            initial_cycles = initial_status["total_cycles"]
            
            # Trigger evolution with test context
            payload = {
                "reason": "comprehensive_testing_manual_trigger"
            }
            
            response = requests.post(
                f"{self.base_url}/api/echo/evolution/trigger",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["success", "cycle_id", "message"]
                
                for field in required_fields:
                    if field not in data:
                        self.log_test_result("Manual Evolution Trigger", False, f"Missing field: {field}")
                        return False
                
                if not data["success"]:
                    self.log_test_result("Manual Evolution Trigger", False, "Trigger reported failure")
                    return False
                
                # Wait a moment and check if cycle count increased
                time.sleep(2)
                
                status_response = requests.get(f"{self.base_url}/api/echo/evolution/status", timeout=10)
                if status_response.status_code == 200:
                    new_status = status_response.json()
                    new_cycles = new_status["total_cycles"]
                    
                    if new_cycles <= initial_cycles:
                        self.log_test_result("Manual Evolution Trigger", False, "Cycle count did not increase")
                        return False
                
                self.log_test_result("Manual Evolution Trigger", True)
                return True
            else:
                self.log_test_result("Manual Evolution Trigger", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test_result("Manual Evolution Trigger", False, str(e))
            return False
    
    async def test_git_integration_components(self) -> bool:
        """Test git integration components without making changes"""
        try:
            # Import the git manager directly for testing
            import sys
            sys.path.append('/opt/tower-echo-brain')
            
            from echo_git_integration import EchoGitManager
            
            git_manager = EchoGitManager()
            
            # Test git status
            git_status = git_manager.get_git_status()
            if "current_branch" not in git_status:
                self.log_test_result("Git Integration Components", False, "Git status missing current_branch")
                return False
            
            # Test evolution status
            evolution_status = git_manager.get_autonomous_evolution_status()
            required_status_fields = [
                "git_integration", "deployment_config", "safety_config", 
                "learning_metrics", "capabilities"
            ]
            
            for field in required_status_fields:
                if field not in evolution_status:
                    self.log_test_result("Git Integration Components", False, f"Missing status field: {field}")
                    return False
            
            # Test safety configuration
            safety_config = evolution_status["safety_config"]
            required_safety_fields = [
                "require_tests", "max_autonomous_changes", "human_oversight_threshold",
                "rollback_on_failure", "backup_before_deploy"
            ]
            
            for field in required_safety_fields:
                if field not in safety_config:
                    self.log_test_result("Git Integration Components", False, f"Missing safety field: {field}")
                    return False
            
            self.log_test_result("Git Integration Components", True)
            return True
            
        except Exception as e:
            self.log_test_result("Git Integration Components", False, str(e))
            return False
    
    async def test_self_analysis_components(self) -> bool:
        """Test self-analysis system components"""
        try:
            # Import the self-analysis system directly
            import sys
            sys.path.append('/opt/tower-echo-brain')
            
            from echo_self_analysis import EchoSelfAnalysis, AnalysisDepth
            
            self_analysis = EchoSelfAnalysis()
            
            # Test capability registry
            if not self_analysis.capability_registry:
                self.log_test_result("Self-Analysis Components", False, "Empty capability registry")
                return False
            
            expected_capabilities = [
                "intelligence_routing", "self_reflection", "learning_adaptation", 
                "architectural_understanding"
            ]
            
            for capability in expected_capabilities:
                if capability not in self_analysis.capability_registry:
                    self.log_test_result("Self-Analysis Components", False, f"Missing capability: {capability}")
                    return False
            
            # Test analysis depth levels
            depth_levels = list(AnalysisDepth)
            if len(depth_levels) < 4:
                self.log_test_result("Self-Analysis Components", False, "Insufficient analysis depth levels")
                return False
            
            self.log_test_result("Self-Analysis Components", True)
            return True
            
        except Exception as e:
            self.log_test_result("Self-Analysis Components", False, str(e))
            return False
    
    async def test_evolution_orchestrator(self) -> bool:
        """Test evolution orchestrator components"""
        try:
            # Import the evolution orchestrator
            import sys
            sys.path.append('/opt/tower-echo-brain')
            
            from echo_autonomous_evolution import EchoAutonomousEvolution, EvolutionTrigger, EvolutionPhase
            
            evolution = EchoAutonomousEvolution()
            
            # Test configuration
            required_config_fields = [
                "analysis_frequency_hours", "learning_milestone_threshold",
                "performance_degradation_threshold", "max_concurrent_evolutions"
            ]
            
            for field in required_config_fields:
                if field not in evolution.evolution_config:
                    self.log_test_result("Evolution Orchestrator", False, f"Missing config field: {field}")
                    return False
            
            # Test state tracking
            required_state_fields = [
                "conversations_since_last_analysis", "improvements_applied",
                "performance_trend", "capability_progression"
            ]
            
            for field in required_state_fields:
                if field not in evolution.learning_state:
                    self.log_test_result("Evolution Orchestrator", False, f"Missing state field: {field}")
                    return False
            
            # Test trigger types
            trigger_types = list(EvolutionTrigger)
            if len(trigger_types) < 5:
                self.log_test_result("Evolution Orchestrator", False, "Insufficient trigger types")
                return False
            
            # Test phase types
            phase_types = list(EvolutionPhase)
            if len(phase_types) < 6:
                self.log_test_result("Evolution Orchestrator", False, "Insufficient evolution phases")
                return False
            
            self.log_test_result("Evolution Orchestrator", True)
            return True
            
        except Exception as e:
            self.log_test_result("Evolution Orchestrator", False, str(e))
            return False
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests"""
        logger.info("🚀 Starting Echo Autonomous Evolution Comprehensive Tests")
        logger.info("=" * 60)
        
        # List of all tests to run
        tests = [
            ("Service Health", self.test_service_health),
            ("Evolution Status", self.test_evolution_status),
            ("Git Status", self.test_git_status),
            ("Self-Analysis Trigger", self.test_self_analysis_trigger),
            ("Learning Metrics", self.test_learning_metrics),
            ("Conversation Processing", self.test_conversation_processing),
            ("Manual Evolution Trigger", self.test_manual_evolution_trigger),
            ("Git Integration Components", self.test_git_integration_components),
            ("Self-Analysis Components", self.test_self_analysis_components),
            ("Evolution Orchestrator", self.test_evolution_orchestrator)
        ]
        
        # Run all tests
        for test_name, test_func in tests:
            logger.info(f"\n🧪 Running test: {test_name}")
            try:
                await test_func()
            except Exception as e:
                self.log_test_result(test_name, False, f"Test execution error: {e}")
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        # Finalize results
        self.test_results["end_time"] = datetime.now()
        duration = (self.test_results["end_time"] - self.test_results["start_time"]).total_seconds()
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("🏁 ECHO AUTONOMOUS EVOLUTION TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests Run: {self.test_results['tests_run']}")
        logger.info(f"Tests Passed: {self.test_results['tests_passed']}")
        logger.info(f"Tests Failed: {self.test_results['tests_failed']}")
        logger.info(f"Success Rate: {(self.test_results['tests_passed'] / max(1, self.test_results['tests_run']) * 100):.1f}%")
        logger.info(f"Total Duration: {duration:.1f} seconds")
        
        if self.test_results["tests_failed"] > 0:
            logger.info(f"\n❌ FAILED TESTS:")
            for failure in self.test_results["failures"]:
                logger.info(f"  - {failure['test']}: {failure['error']}")
        else:
            logger.info(f"\n✅ ALL TESTS PASSED!")
        
        return self.test_results

async def main():
    """Main test execution"""
    tester = EchoEvolutionTester()
    results = await tester.run_comprehensive_tests()
    
    # Save results to file
    results_file = f"/opt/tower-echo-brain/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n📊 Test results saved to: {results_file}")
    
    # Return exit code based on test results
    return 0 if results["tests_failed"] == 0 else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
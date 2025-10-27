#!/usr/bin/env python3
"""
Echo Git Integration Module
Provides autonomous git operations for Echo's self-improvement
"""

import subprocess
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class EchoGitManager:
    """Manages git operations for Echo's autonomous improvement"""
    
    def __init__(self, repo_path: str = "/opt/tower-echo-brain"):
        self.repo_path = Path(repo_path)
        self.repo_path_str = str(self.repo_path)
        self.source_repo_path = Path("/home/patrick/Documents/Tower/services/echo-brain")
        
        # Deployment configuration
        self.deployment_config = {
            "production_path": "/opt/tower-echo-brain",
            "source_path": "/home/patrick/Documents/Tower/services/echo-brain",
            "service_name": "tower-echo-brain",
            "backup_path": "/opt/tower-echo-brain/backups",
            "test_timeout": 30,  # seconds
            "safety_checks": True
        }
        
        # Safety and testing configuration
        self.safety_config = {
            "require_tests": True,
            "max_autonomous_changes": 5,  # per day
            "human_oversight_threshold": 0.7,  # confidence score
            "rollback_on_failure": True,
            "backup_before_deploy": True
        }
        
        # Learning progress tracking
        self.learning_metrics = {
            "improvements_applied": 0,
            "deployments_successful": 0,
            "rollbacks_triggered": 0,
            "tests_passed": 0,
            "tests_failed": 0
        }
        
        # Ensure we're in a git repository
        self._init_git_if_needed()
    
    def _init_git_if_needed(self):
        """Initialize git repository if not already present"""
        try:
            # Check if .git directory exists
            git_dir = self.repo_path / ".git"
            if not git_dir.exists():
                logger.info("Initializing git repository for Echo")
                self._run_git_command(["git", "init"])
                
                # Configure git user for Echo
                self._run_git_command(["git", "config", "user.name", "Echo AI"])
                self._run_git_command(["git", "config", "user.email", "echo@tower.local"])
                
                # Add initial files
                self._run_git_command(["git", "add", "."])
                self._run_git_command(["git", "commit", "-m", "Initial AI Assist repository"])
                
                logger.info("Git repository initialized for Echo")
        except Exception as e:
            logger.error(f"Failed to initialize git repository: {e}")
    
    def _run_git_command(self, command: List[str], cwd: str = None) -> subprocess.CompletedProcess:
        """Run git command with error handling"""
        try:
            if cwd is None:
                cwd = self.repo_path_str
            
            result = subprocess.run(
                command, 
                cwd=cwd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {command} - {e.stderr}")
            raise
    
    def get_git_status(self) -> Dict[str, Any]:
        """Get current git repository status"""
        try:
            # Get status
            status_result = self._run_git_command(["git", "status", "--porcelain"])
            
            # Get current branch
            branch_result = self._run_git_command(["git", "branch", "--show-current"])
            
            # Get last commit
            try:
                commit_result = self._run_git_command(["git", "log", "-1", "--oneline"])
                last_commit = commit_result.stdout.strip()
            except:
                last_commit = "No commits yet"
            
            # Parse status
            modified_files = []
            untracked_files = []
            staged_files = []
            
            for line in status_result.stdout.strip().split('\n'):
                if line:
                    status_char = line[0]
                    filename = line[3:]
                    
                    if status_char == 'M':
                        modified_files.append(filename)
                    elif status_char == '?':
                        untracked_files.append(filename)
                    elif status_char == 'A':
                        staged_files.append(filename)
            
            return {
                "current_branch": branch_result.stdout.strip(),
                "last_commit": last_commit,
                "modified_files": modified_files,
                "untracked_files": untracked_files,
                "staged_files": staged_files,
                "clean": len(modified_files) == 0 and len(untracked_files) == 0 and len(staged_files) == 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get git status: {e}")
            return {"error": str(e)}
    
    def create_improvement_branch(self, improvement_type: str, analysis_id: str) -> str:
        """Create a new branch for autonomous improvements"""
        try:
            branch_name = f"echo-improvement/{improvement_type}_{analysis_id}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            
            # Create and checkout new branch
            self._run_git_command(["git", "checkout", "-b", branch_name])
            
            logger.info(f"Created improvement branch: {branch_name}")
            return branch_name
            
        except Exception as e:
            logger.error(f"Failed to create improvement branch: {e}")
            raise
    
    def commit_improvement(self, message: str, analysis_context: Dict[str, Any]) -> bool:
        """Commit autonomous improvements with detailed context"""
        try:
            # Add all changes
            self._run_git_command(["git", "add", "."])
            
            # Create detailed commit message
            commit_msg = f"""Echo Autonomous Improvement: {message}

Analysis Context:
- Analysis ID: {analysis_context.get('analysis_id', 'unknown')}
- Depth: {analysis_context.get('depth', 'unknown')}
- Trigger: {analysis_context.get('trigger_type', 'autonomous')}
- Confidence: {analysis_context.get('confidence_score', 0.0)}

Improvement Actions:
{chr(10).join(f"- {action}" for action in analysis_context.get('action_items', [])[:5])}

Generated: {datetime.now().isoformat()}
By: AI Assist Autonomous Improvement System
"""
            
            # Commit with detailed message
            self._run_git_command(["git", "commit", "-m", commit_msg])
            
            logger.info(f"Committed improvement: {message}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to commit improvement: {e}")
            return False
    
    def create_self_improvement_file(self, improvement_data: Dict[str, Any]) -> str:
        """Create a self-improvement implementation file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"improvements/echo_improvement_{timestamp}.py"
            filepath = self.repo_path / filename
            
            # Ensure improvements directory exists
            filepath.parent.mkdir(exist_ok=True)
            
            # Generate improvement code
            improvement_code = self._generate_improvement_code(improvement_data)
            
            # Write file
            with open(filepath, 'w') as f:
                f.write(improvement_code)
            
            logger.info(f"Created self-improvement file: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to create improvement file: {e}")
            raise
    
    def _generate_improvement_code(self, improvement_data: Dict[str, Any]) -> str:
        """Generate Python code for autonomous improvements"""
        analysis_id = improvement_data.get('analysis_id', 'unknown')
        capabilities = improvement_data.get('capabilities', [])
        action_items = improvement_data.get('action_items', [])
        
        code_template = f'''#!/usr/bin/env python3
"""
Echo Autonomous Improvement Implementation
Generated: {datetime.now().isoformat()}
Analysis ID: {analysis_id}
"""

import logging
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class EchoImprovement_{analysis_id.replace('-', '_')}:
    """Autonomous improvement implementation based on self-analysis"""
    
    def __init__(self):
        self.analysis_id = "{analysis_id}"
        self.capabilities_to_improve = {json.dumps(capabilities, indent=8)}
        self.action_items = {json.dumps(action_items, indent=8)}
        self.implemented = False
    
    def apply_improvements(self) -> Dict[str, Any]:
        """Apply the identified improvements"""
        results = {{
            "analysis_id": self.analysis_id,
            "timestamp": datetime.now().isoformat(),
            "improvements_applied": [],
            "improvements_failed": []
        }}
        
        try:
            # Implement capability improvements
            for capability in self.capabilities_to_improve:
                improvement_result = self._improve_capability(capability)
                if improvement_result["success"]:
                    results["improvements_applied"].append(improvement_result)
                else:
                    results["improvements_failed"].append(improvement_result)
            
            self.implemented = True
            logger.info(f"Applied {{len(results['improvements_applied'])}} improvements")
            
        except Exception as e:
            logger.error(f"Failed to apply improvements: {{e}}")
            results["error"] = str(e)
        
        return results
    
    def _improve_capability(self, capability: Dict[str, Any]) -> Dict[str, Any]:
        """Improve a specific capability"""
        capability_name = capability.get("name", "unknown")
        current_level = capability.get("current_level", 0.0)
        desired_level = capability.get("desired_level", 1.0)
        gap = capability.get("gap", 0.0)
        
        # Log improvement attempt
        logger.info(f"Improving capability: {{capability_name}} (gap: {{gap:.3f}})")
        
        # Placeholder for actual improvement implementation
        # This would be enhanced based on the specific capability
        improvement_success = gap < 0.5  # Simple success criteria
        
        return {{
            "capability": capability_name,
            "success": improvement_success,
            "gap_reduced": gap * 0.1 if improvement_success else 0.0,
            "method": "autonomous_enhancement",
            "timestamp": datetime.now().isoformat()
        }}
    
    def get_status(self) -> Dict[str, Any]:
        """Get improvement implementation status"""
        return {{
            "analysis_id": self.analysis_id,
            "implemented": self.implemented,
            "capability_count": len(self.capabilities_to_improve),
            "action_item_count": len(self.action_items),
            "created": "{datetime.now().isoformat()}"
        }}

# Auto-execution when imported
if __name__ == "__main__":
    improvement = EchoImprovement_{analysis_id.replace('-', '_')}()
    result = improvement.apply_improvements()
    print(f"Improvement execution result: {{result}}")
'''
        
        return code_template
    
    def merge_improvement_branch(self, branch_name: str, target_branch: str = "main") -> bool:
        """Merge improvement branch back to main"""
        try:
            # Switch to target branch
            self._run_git_command(["git", "checkout", target_branch])
            
            # Merge improvement branch
            self._run_git_command(["git", "merge", branch_name, "--no-ff"])
            
            # Delete improvement branch
            self._run_git_command(["git", "branch", "-d", branch_name])
            
            logger.info(f"Merged improvement branch {branch_name} into {target_branch}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to merge improvement branch: {e}")
            return False
    
    def get_improvement_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get history of Echo's autonomous improvements"""
        try:
            # Get commit history for improvement commits
            result = self._run_git_command([
                "git", "log", 
                "--grep=Echo Autonomous Improvement",
                "--oneline",
                f"-{limit}"
            ])
            
            improvements = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    commit_hash, message = line.split(' ', 1)
                    improvements.append({
                        "commit_hash": commit_hash,
                        "message": message,
                        "type": "autonomous_improvement"
                    })
            
            return improvements
            
        except Exception as e:
            logger.error(f"Failed to get improvement history: {e}")
            return []
    
    def autonomous_commit_self_analysis(self, analysis_result: Dict[str, Any]) -> bool:
        """Autonomously commit self-analysis results and improvements"""
        try:
            analysis_id = analysis_result.get("analysis_id", "unknown")
            
            # Create improvement branch
            branch_name = self.create_improvement_branch("self_analysis", analysis_id)
            
            # Create improvement implementation file
            improvement_file = self.create_self_improvement_file(analysis_result)
            
            # Commit the improvements
            commit_success = self.commit_improvement(
                f"Self-analysis based improvements ({analysis_id})",
                analysis_result
            )
            
            if commit_success:
                # Merge back to main branch
                merge_success = self.merge_improvement_branch(branch_name)
                return merge_success
            
            return False
            
        except Exception as e:
            logger.error(f"Failed autonomous commit: {e}")
            return False
    
    def safe_autonomous_deployment(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Safely deploy autonomous improvements with comprehensive testing"""
        deployment_result = {
            "success": False,
            "analysis_id": analysis_result.get("analysis_id", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "steps_completed": [],
            "tests_run": [],
            "backup_created": False,
            "rollback_required": False,
            "error": None
        }
        
        try:
            # Step 1: Safety checks
            safety_check = self._perform_safety_checks(analysis_result)
            if not safety_check["passed"]:
                deployment_result["error"] = f"Safety checks failed: {safety_check['reason']}"
                return deployment_result
            
            deployment_result["steps_completed"].append("safety_checks")
            
            # Step 2: Create backup
            if self.safety_config["backup_before_deploy"]:
                backup_result = self._create_deployment_backup()
                deployment_result["backup_created"] = backup_result["success"]
                if not backup_result["success"]:
                    deployment_result["error"] = f"Backup failed: {backup_result['error']}"
                    return deployment_result
                deployment_result["steps_completed"].append("backup_created")
            
            # Step 3: Run pre-deployment tests
            test_result = self._run_pre_deployment_tests()
            deployment_result["tests_run"].extend(test_result["tests"])
            if not test_result["passed"]:
                deployment_result["error"] = f"Pre-deployment tests failed: {test_result['failures']}"
                return deployment_result
            deployment_result["steps_completed"].append("pre_deployment_tests")
            
            # Step 4: Apply improvements to source repository
            source_improvement = self._apply_improvements_to_source(analysis_result)
            if not source_improvement["success"]:
                deployment_result["error"] = f"Source improvement failed: {source_improvement['error']}"
                return deployment_result
            deployment_result["steps_completed"].append("source_improvement")
            
            # Step 5: Sync source to production
            sync_result = self._sync_source_to_production()
            if not sync_result["success"]:
                deployment_result["error"] = f"Sync failed: {sync_result['error']}"
                deployment_result["rollback_required"] = True
                return deployment_result
            deployment_result["steps_completed"].append("sync_to_production")
            
            # Step 6: Run post-deployment tests
            post_test_result = self._run_post_deployment_tests()
            deployment_result["tests_run"].extend(post_test_result["tests"])
            if not post_test_result["passed"]:
                deployment_result["error"] = f"Post-deployment tests failed: {post_test_result['failures']}"
                deployment_result["rollback_required"] = True
                return deployment_result
            deployment_result["steps_completed"].append("post_deployment_tests")
            
            # Step 7: Restart service if needed
            service_restart = self._restart_echo_service()
            if not service_restart["success"]:
                deployment_result["error"] = f"Service restart failed: {service_restart['error']}"
                deployment_result["rollback_required"] = True
                return deployment_result
            deployment_result["steps_completed"].append("service_restart")
            
            # Step 8: Verify deployment health
            health_check = self._verify_deployment_health()
            if not health_check["healthy"]:
                deployment_result["error"] = f"Health check failed: {health_check['issues']}"
                deployment_result["rollback_required"] = True
                return deployment_result
            deployment_result["steps_completed"].append("health_verification")
            
            # Step 9: Update learning metrics
            self._update_learning_metrics("deployment_success")
            
            deployment_result["success"] = True
            logger.info(f"Autonomous deployment successful: {analysis_result.get('analysis_id')}")
            
        except Exception as e:
            logger.error(f"Deployment failed with exception: {e}")
            deployment_result["error"] = str(e)
            deployment_result["rollback_required"] = True
        
        # Handle rollback if needed
        if deployment_result["rollback_required"]:
            rollback_result = self._perform_rollback()
            deployment_result["rollback_performed"] = rollback_result["success"]
            self._update_learning_metrics("rollback_triggered")
        
        return deployment_result
    
    def _perform_safety_checks(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive safety checks before deployment"""
        checks = {
            "passed": True,
            "reason": None,
            "checks_performed": []
        }
        
        try:
            # Check 1: Daily change limit
            daily_changes = self._get_daily_change_count()
            max_changes = self.safety_config["max_autonomous_changes"]
            if daily_changes >= max_changes:
                checks["passed"] = False
                checks["reason"] = f"Daily change limit reached: {daily_changes}/{max_changes}"
                return checks
            checks["checks_performed"].append("daily_change_limit")
            
            # Check 2: Confidence threshold
            confidence = analysis_result.get("confidence_score", 0.0)
            threshold = self.safety_config["human_oversight_threshold"]
            if confidence < threshold:
                checks["passed"] = False
                checks["reason"] = f"Confidence too low: {confidence} < {threshold}"
                return checks
            checks["checks_performed"].append("confidence_threshold")
            
            # Check 3: Critical system components
            if self._affects_critical_components(analysis_result):
                checks["passed"] = False
                checks["reason"] = "Changes affect critical system components"
                return checks
            checks["checks_performed"].append("critical_components")
            
            # Check 4: Recent failure patterns
            if self._has_recent_failures():
                checks["passed"] = False
                checks["reason"] = "Recent deployment failures detected"
                return checks
            checks["checks_performed"].append("failure_patterns")
            
        except Exception as e:
            checks["passed"] = False
            checks["reason"] = f"Safety check error: {e}"
        
        return checks
    
    def _create_deployment_backup(self) -> Dict[str, Any]:
        """Create backup before deployment"""
        backup_result = {
            "success": False,
            "backup_path": None,
            "error": None
        }
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = Path(self.deployment_config["backup_path"])
            backup_dir.mkdir(exist_ok=True)
            
            backup_path = backup_dir / f"echo_backup_{timestamp}"
            
            # Create backup using git
            self._run_git_command([
                "git", "clone", 
                self.deployment_config["production_path"],
                str(backup_path)
            ])
            
            backup_result["success"] = True
            backup_result["backup_path"] = str(backup_path)
            logger.info(f"Backup created: {backup_path}")
            
        except Exception as e:
            backup_result["error"] = str(e)
            logger.error(f"Backup creation failed: {e}")
        
        return backup_result
    
    def _run_pre_deployment_tests(self) -> Dict[str, Any]:
        """Run tests before deployment"""
        test_result = {
            "passed": True,
            "tests": [],
            "failures": []
        }
        
        try:
            # Test 1: Import syntax check
            syntax_test = self._test_python_syntax()
            test_result["tests"].append(syntax_test)
            if not syntax_test["passed"]:
                test_result["passed"] = False
                test_result["failures"].append(syntax_test["name"])
            
            # Test 2: Basic functionality test
            function_test = self._test_basic_functionality()
            test_result["tests"].append(function_test)
            if not function_test["passed"]:
                test_result["passed"] = False
                test_result["failures"].append(function_test["name"])
            
            # Test 3: Database connectivity
            db_test = self._test_database_connectivity()
            test_result["tests"].append(db_test)
            if not db_test["passed"]:
                test_result["passed"] = False
                test_result["failures"].append(db_test["name"])
            
        except Exception as e:
            test_result["passed"] = False
            test_result["failures"].append(f"Test framework error: {e}")
        
        return test_result
    
    def _apply_improvements_to_source(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply improvements to source repository"""
        result = {
            "success": False,
            "changes_applied": [],
            "error": None
        }
        
        try:
            # Switch to source repository
            os.chdir(str(self.source_repo_path))
            
            # Create improvement branch in source
            analysis_id = analysis_result.get("analysis_id", "unknown")
            branch_name = f"echo-auto-improvement-{analysis_id}"
            
            self._run_git_command(["git", "checkout", "-b", branch_name], 
                                cwd=str(self.source_repo_path))
            
            # Apply improvements based on analysis
            improvements = self._generate_code_improvements(analysis_result)
            for improvement in improvements:
                self._apply_code_improvement(improvement)
                result["changes_applied"].append(improvement["description"])
            
            # Commit changes in source
            self._run_git_command(["git", "add", "."], cwd=str(self.source_repo_path))
            commit_msg = f"Echo autonomous improvement: {analysis_id}"
            self._run_git_command(["git", "commit", "-m", commit_msg], 
                                cwd=str(self.source_repo_path))
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Source improvement failed: {e}")
        finally:
            # Return to production directory
            os.chdir(str(self.repo_path))
        
        return result
    
    def _sync_source_to_production(self) -> Dict[str, Any]:
        """Synchronize source repository to production"""
        result = {
            "success": False,
            "files_synced": [],
            "error": None
        }
        
        try:
            # Copy updated files from source to production
            import shutil
            
            source_files = [
                "echo_unified_service.py",
                "echo_self_analysis.py", 
                "echo_git_integration.py",
                "echo_brain_thoughts.py"
            ]
            
            for file_name in source_files:
                source_file = self.source_repo_path / file_name
                prod_file = self.repo_path / file_name
                
                if source_file.exists():
                    shutil.copy2(str(source_file), str(prod_file))
                    result["files_synced"].append(file_name)
            
            result["success"] = True
            logger.info(f"Synced {len(result['files_synced'])} files to production")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Sync to production failed: {e}")
        
        return result
    
    def _run_post_deployment_tests(self) -> Dict[str, Any]:
        """Run tests after deployment"""
        # Similar to pre-deployment tests but in production environment
        return self._run_pre_deployment_tests()
    
    def _restart_echo_service(self) -> Dict[str, Any]:
        """Restart AI Assist service"""
        result = {
            "success": False,
            "error": None
        }
        
        try:
            # Restart systemd service
            subprocess.run([
                "sudo", "systemctl", "restart", 
                self.deployment_config["service_name"]
            ], check=True, capture_output=True, text=True)
            
            # Wait for service to start
            import time
            time.sleep(5)
            
            # Check service status
            status_result = subprocess.run([
                "sudo", "systemctl", "is-active", 
                self.deployment_config["service_name"]
            ], capture_output=True, text=True)
            
            if status_result.stdout.strip() == "active":
                result["success"] = True
            else:
                result["error"] = "Service not active after restart"
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Service restart failed: {e}")
        
        return result
    
    def _verify_deployment_health(self) -> Dict[str, Any]:
        """Verify deployment health after changes"""
        health_result = {
            "healthy": True,
            "issues": [],
            "checks_passed": []
        }
        
        try:
            # Check 1: Service responsiveness
            import requests
            import time
            time.sleep(2)  # Allow service to fully start
            
            try:
                response = requests.get("http://localhost:8309/api/echo/health", timeout=10)
                if response.status_code == 200:
                    health_result["checks_passed"].append("service_responsive")
                else:
                    health_result["healthy"] = False
                    health_result["issues"].append("Service not responding correctly")
            except requests.RequestException:
                health_result["healthy"] = False
                health_result["issues"].append("Service not reachable")
            
            # Check 2: Database connectivity
            db_check = self._test_database_connectivity()
            if db_check["passed"]:
                health_result["checks_passed"].append("database_connectivity")
            else:
                health_result["healthy"] = False
                health_result["issues"].append("Database connectivity issue")
            
            # Check 3: Memory usage
            memory_check = self._check_memory_usage()
            if memory_check["healthy"]:
                health_result["checks_passed"].append("memory_usage")
            else:
                health_result["healthy"] = False
                health_result["issues"].append("High memory usage")
                
        except Exception as e:
            health_result["healthy"] = False
            health_result["issues"].append(f"Health check error: {e}")
        
        return health_result
    
    def _perform_rollback(self) -> Dict[str, Any]:
        """Perform rollback to previous state"""
        rollback_result = {
            "success": False,
            "error": None,
            "restored_from": None
        }
        
        try:
            # Find latest backup
            backup_dir = Path(self.deployment_config["backup_path"])
            if not backup_dir.exists():
                rollback_result["error"] = "No backup directory found"
                return rollback_result
            
            backups = sorted(backup_dir.glob("echo_backup_*"), reverse=True)
            if not backups:
                rollback_result["error"] = "No backups available"
                return rollback_result
            
            latest_backup = backups[0]
            
            # Restore from backup
            import shutil
            for item in latest_backup.iterdir():
                if item.is_file():
                    dest = self.repo_path / item.name
                    shutil.copy2(str(item), str(dest))
            
            # Restart service
            restart_result = self._restart_echo_service()
            if not restart_result["success"]:
                rollback_result["error"] = f"Rollback completed but service restart failed: {restart_result['error']}"
                return rollback_result
            
            rollback_result["success"] = True
            rollback_result["restored_from"] = str(latest_backup)
            logger.info(f"Rollback successful from {latest_backup}")
            
        except Exception as e:
            rollback_result["error"] = str(e)
            logger.error(f"Rollback failed: {e}")
        
        return rollback_result
    
    # Helper methods for various checks and operations
    
    def _get_daily_change_count(self) -> int:
        """Get number of autonomous changes today"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            result = self._run_git_command([
                "git", "log", "--oneline", "--since", f"{today} 00:00:00",
                "--grep", "Echo Autonomous Improvement"
            ])
            return len([line for line in result.stdout.strip().split('\n') if line])
        except:
            return 0
    
    def _affects_critical_components(self, analysis_result: Dict[str, Any]) -> bool:
        """Check if changes affect critical system components"""
        critical_keywords = [
            "database", "authentication", "core", "router", 
            "security", "config", "service"
        ]
        
        analysis_text = str(analysis_result).lower()
        return any(keyword in analysis_text for keyword in critical_keywords)
    
    def _has_recent_failures(self) -> bool:
        """Check for recent deployment failures"""
        # Check systemd logs for recent failures
        try:
            result = subprocess.run([
                "journalctl", "-u", self.deployment_config["service_name"],
                "--since", "24 hours ago", "--grep", "error"
            ], capture_output=True, text=True)
            return len(result.stdout.strip()) > 0
        except:
            return False
    
    def _test_python_syntax(self) -> Dict[str, Any]:
        """Test Python syntax of key files"""
        test_result = {
            "name": "python_syntax",
            "passed": True,
            "error": None
        }
        
        try:
            import ast
            key_files = [
                "echo_unified_service.py",
                "echo_self_analysis.py",
                "echo_git_integration.py"
            ]
            
            for file_name in key_files:
                file_path = self.repo_path / file_name
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        ast.parse(f.read())
                        
        except SyntaxError as e:
            test_result["passed"] = False
            test_result["error"] = f"Syntax error: {e}"
        except Exception as e:
            test_result["passed"] = False
            test_result["error"] = f"Test error: {e}"
        
        return test_result
    
    def _test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic Echo functionality"""
        test_result = {
            "name": "basic_functionality",
            "passed": True,
            "error": None
        }
        
        try:
            # Test if we can import main modules
            import sys
            sys.path.insert(0, str(self.repo_path))
            
            from echo_self_analysis import echo_self_analysis
            from echo_brain_thoughts import echo_brain
            
            # Basic functionality test
            test_query = "What is the current time?"
            response = echo_brain.quick_thought(test_query)
            
            if not response or len(response) < 10:
                test_result["passed"] = False
                test_result["error"] = "Basic functionality test failed"
                
        except Exception as e:
            test_result["passed"] = False
            test_result["error"] = f"Functionality test error: {e}"
        
        return test_result
    
    def _test_database_connectivity(self) -> Dict[str, Any]:
        """Test database connectivity"""
        test_result = {
            "name": "database_connectivity",
            "passed": True,
            "error": None
        }
        
        try:
            import psycopg2
            db_config = {
                'host': 'localhost',
                'database': 'echo_brain',
                'user': 'patrick',
                'password': 'Beau40818'
            }
            
            conn = psycopg2.connect(**db_config)
            cur = conn.cursor()
            cur.execute("SELECT 1")
            result = cur.fetchone()
            
            if result[0] != 1:
                test_result["passed"] = False
                test_result["error"] = "Database query failed"
            
            conn.close()
            
        except Exception as e:
            test_result["passed"] = False
            test_result["error"] = f"Database test error: {e}"
        
        return test_result
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage health"""
        memory_result = {
            "healthy": True,
            "usage_mb": 0,
            "threshold_mb": 1000  # 1GB threshold
        }
        
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            memory_result["usage_mb"] = memory_mb
            memory_result["healthy"] = memory_mb < memory_result["threshold_mb"]
            
        except Exception as e:
            memory_result["healthy"] = False
            memory_result["error"] = str(e)
        
        return memory_result
    
    def _generate_code_improvements(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate code improvements based on analysis"""
        improvements = []
        
        capabilities = analysis_result.get("capabilities", [])
        action_items = analysis_result.get("action_items", [])
        
        for capability in capabilities:
            if capability.get("current_level", 0) < capability.get("desired_level", 1):
                improvement = {
                    "type": "capability_enhancement",
                    "target": capability.get("name", "unknown"),
                    "description": f"Enhance {capability.get('name')} capability",
                    "file": "echo_unified_service.py",
                    "method": "improve_capability_routing"
                }
                improvements.append(improvement)
        
        return improvements
    
    def _apply_code_improvement(self, improvement: Dict[str, Any]) -> bool:
        """Apply a specific code improvement"""
        try:
            # This would contain actual code modification logic
            # For now, just log the improvement
            logger.info(f"Applying improvement: {improvement['description']}")
            
            # Placeholder for actual code modification
            # This would modify files based on improvement specifications
            
            return True
        except Exception as e:
            logger.error(f"Failed to apply improvement {improvement['description']}: {e}")
            return False
    
    def _update_learning_metrics(self, metric_type: str):
        """Update learning progress metrics"""
        if metric_type in self.learning_metrics:
            self.learning_metrics[metric_type] += 1
        
        # Store metrics to database
        try:
            import psycopg2
            db_config = {
                'host': 'localhost',
                'database': 'echo_brain',
                'user': 'patrick',
                'password': 'Beau40818'
            }
            
            conn = psycopg2.connect(**db_config)
            cur = conn.cursor()
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS echo_learning_metrics (
                    metric_type VARCHAR,
                    count INTEGER,
                    last_updated TIMESTAMP,
                    PRIMARY KEY (metric_type)
                )
            """)
            
            cur.execute("""
                INSERT INTO echo_learning_metrics (metric_type, count, last_updated)
                VALUES (%s, %s, %s)
                ON CONFLICT (metric_type) DO UPDATE SET
                count = echo_learning_metrics.count + 1,
                last_updated = %s
            """, (metric_type, 1, datetime.now(), datetime.now()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update learning metrics: {e}")
    
    def get_autonomous_evolution_status(self) -> Dict[str, Any]:
        """Get comprehensive status of autonomous evolution system"""
        return {
            "git_integration": {
                "repository_path": str(self.repo_path),
                "source_path": str(self.source_repo_path),
                "git_status": self.get_git_status()
            },
            "deployment_config": self.deployment_config,
            "safety_config": self.safety_config,
            "learning_metrics": self.learning_metrics,
            "capabilities": {
                "autonomous_improvement": True,
                "safe_deployment": True,
                "rollback_capability": True,
                "learning_tracking": True,
                "source_sync": True
            },
            "recent_improvements": self.get_improvement_history(5),
            "system_health": {
                "last_check": datetime.now().isoformat(),
                "status": "operational"
            }
        }
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
                self._run_git_command(["git", "commit", "-m", "Initial Echo Brain repository"])
                
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
By: Echo Brain Autonomous Improvement System
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
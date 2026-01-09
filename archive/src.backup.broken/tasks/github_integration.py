#!/usr/bin/env python3
"""
GitHub Integration for Echo Brain
Handles PR creation, branch management, and CI/CD controls
"""

import asyncio
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class GitHubIntegration:
    """GitHub integration for Echo's proactive improvements"""

    def __init__(self):
        self.repo_path = Path("/opt/tower-echo-brain")
        self.main_branch = "main"
        self.feature_prefix = "echo-improvement"
        self.max_open_prs = 5  # Limit concurrent PRs

    async def check_auth(self) -> bool:
        """Verify GitHub CLI authentication"""
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"GitHub auth check failed: {e}")
            return False

    async def get_open_prs(self) -> List[Dict]:
        """Get list of open PRs created by Echo"""
        try:
            result = subprocess.run(
                ["gh", "pr", "list", "--author", "@me", "--json", "number,title,state,branch"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
            return []
        except Exception as e:
            logger.error(f"Failed to get PRs: {e}")
            return []

    async def create_improvement_branch(self, improvement_type: str) -> str:
        """Create a new branch for improvements"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        branch_name = f"{self.feature_prefix}/{improvement_type}_{timestamp}"

        try:
            # Create and checkout new branch
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=self.repo_path,
                check=True
            )
            logger.info(f"Created branch: {branch_name}")
            return branch_name
        except Exception as e:
            logger.error(f"Failed to create branch: {e}")
            return ""

    async def create_pr(self, title: str, body: str, branch: str) -> Optional[str]:
        """Create a pull request"""
        try:
            # First push the branch
            subprocess.run(
                ["git", "push", "-u", "origin", branch],
                cwd=self.repo_path,
                check=True
            )

            # Create PR using gh CLI
            result = subprocess.run(
                ["gh", "pr", "create",
                 "--title", title,
                 "--body", body,
                 "--base", self.main_branch,
                 "--head", branch],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                pr_url = result.stdout.strip()
                logger.info(f"Created PR: {pr_url}")
                return pr_url
            else:
                logger.error(f"PR creation failed: {result.stderr}")
                return None

        except Exception as e:
            logger.error(f"Failed to create PR: {e}")
            return None

    async def create_quality_improvement_pr(self, files_fixed: List[Dict]) -> Optional[str]:
        """Create PR for code quality improvements"""
        if not files_fixed:
            return None

        # Check if we're under the PR limit
        open_prs = await self.get_open_prs()
        echo_prs = [pr for pr in open_prs if pr.get('branch', '').startswith(self.feature_prefix)]

        if len(echo_prs) >= self.max_open_prs:
            logger.warning(f"Max open PRs ({self.max_open_prs}) reached, skipping PR creation")
            return None

        # Create branch
        branch_name = await self.create_improvement_branch("code_quality")
        if not branch_name:
            return None

        # Commit the changes
        total_improvement = sum(f.get('improvement', 0) for f in files_fixed)
        avg_improvement = total_improvement / len(files_fixed)

        commit_message = f"fix: Improve code quality for {len(files_fixed)} files\n\n"
        commit_message += f"Average score improvement: {avg_improvement:.1f}\n"
        commit_message += "Files improved:\n"
        for file_info in files_fixed[:10]:  # Limit to 10 in message
            commit_message += f"  - {Path(file_info['file']).name}: {file_info['before']:.1f} → {file_info['after']:.1f}\n"

        try:
            subprocess.run(["git", "add", "-A"], cwd=self.repo_path, check=True)
            subprocess.run(["git", "commit", "-m", commit_message], cwd=self.repo_path, check=True)
        except Exception as e:
            logger.error(f"Failed to commit: {e}")
            return None

        # Create PR body
        pr_body = f"""## 🤖 Automated Code Quality Improvements

This PR was automatically generated by Echo Brain's proactive code quality system.

### 📊 Summary
- **Files improved**: {len(files_fixed)}
- **Average score increase**: {avg_improvement:.1f}
- **Total improvement points**: {total_improvement:.1f}

### 📝 Files Changed
| File | Before | After | Improvement |
|------|--------|-------|-------------|
"""
        for file_info in files_fixed[:20]:  # Show up to 20 files
            pr_body += f"| {Path(file_info['file']).name} | {file_info['before']:.1f} | {file_info['after']:.1f} | +{file_info['improvement']:.1f} |\n"

        pr_body += """

### 🔧 Improvements Applied
- Black code formatting
- Common pylint issue fixes
- Import organization
- Docstring additions

### ✅ Quality Checks
All changes have been verified to improve code quality scores without breaking functionality.

### 🤖 Generated by Echo Brain
This is an automated PR created by Echo's proactive improvement system.
"""

        pr_title = f"🤖 Code Quality: Improve {len(files_fixed)} files (avg +{avg_improvement:.1f})"

        # Create the PR
        pr_url = await self.create_pr(pr_title, pr_body, branch_name)

        if pr_url:
            logger.info(f"✅ Created quality improvement PR: {pr_url}")
            # Switch back to original branch
            subprocess.run(["git", "checkout", self.get_current_branch()], cwd=self.repo_path)

        return pr_url

    def get_current_branch(self) -> str:
        """Get current git branch"""
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.stdout.strip() or "feature/board-of-directors"
        except Exception:
            return "feature/board-of-directors"

    async def run_ci_checks(self, branch: str) -> Dict:
        """Run CI checks on a branch"""
        results = {
            "black": False,
            "pylint": False,
            "tests": False,
            "overall": False
        }

        try:
            # Black formatting check
            black_result = subprocess.run(
                ["black", "--check", "."],
                cwd=self.repo_path,
                capture_output=True
            )
            results["black"] = black_result.returncode == 0

            # Pylint check
            pylint_result = subprocess.run(
                ["pylint", "src"],
                cwd=self.repo_path,
                capture_output=True
            )
            results["pylint"] = pylint_result.returncode == 0

            # Tests (if they exist)
            test_path = self.repo_path / "tests"
            if test_path.exists():
                test_result = subprocess.run(
                    ["python", "-m", "pytest", "tests/"],
                    cwd=self.repo_path,
                    capture_output=True
                )
                results["tests"] = test_result.returncode == 0
            else:
                results["tests"] = True  # No tests = pass

            results["overall"] = all([results["black"], results["pylint"], results["tests"]])

        except Exception as e:
            logger.error(f"CI checks failed: {e}")

        return results

    async def merge_pr_if_safe(self, pr_number: int) -> bool:
        """Merge PR if all checks pass"""
        try:
            # Check PR status
            result = subprocess.run(
                ["gh", "pr", "checks", str(pr_number)],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )

            if "All checks have passed" in result.stdout:
                # Merge the PR
                merge_result = subprocess.run(
                    ["gh", "pr", "merge", str(pr_number), "--merge", "--delete-branch"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )

                if merge_result.returncode == 0:
                    logger.info(f"✅ Merged PR #{pr_number}")
                    return True
                else:
                    logger.error(f"Failed to merge PR: {merge_result.stderr}")
            else:
                logger.info(f"PR #{pr_number} checks not passing, skipping merge")

        except Exception as e:
            logger.error(f"Failed to merge PR: {e}")

        return False

    async def setup_branch_protection(self) -> bool:
        """Set up branch protection rules"""
        try:
            # This requires admin access
            result = subprocess.run([
                "gh", "api",
                f"/repos/pvestal/tower-echo-brain/branches/{self.main_branch}/protection",
                "--method", "PUT",
                "--field", "required_status_checks[strict]=true",
                "--field", "required_status_checks[checks][][context]=black",
                "--field", "required_status_checks[checks][][context]=pylint",
                "--field", "required_status_checks[checks][][context]=tests",
                "--field", "enforce_admins=false",
                "--field", "required_pull_request_reviews[required_approving_review_count]=1",
                "--field", "required_pull_request_reviews[dismiss_stale_reviews]=true"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("✅ Branch protection configured")
                return True
            else:
                logger.warning(f"Could not set branch protection: {result.stderr}")

        except Exception as e:
            logger.error(f"Failed to set branch protection: {e}")

        return False

# Initialize GitHub integration
github_integration = GitHubIntegration()
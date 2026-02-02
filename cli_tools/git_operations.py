#!/usr/bin/env python3
"""
Git and GitHub operations for Echo Brain.
Provides version control, PR creation, and documentation management.
"""

import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import asyncio
import re

logger = logging.getLogger(__name__)

class GitOperation(Enum):
    """Types of git operations Echo can perform."""
    COMMIT = "commit"
    BRANCH = "branch"
    PUSH = "push"
    PULL = "pull"
    STASH = "stash"
    MERGE = "merge"
    REBASE = "rebase"
    STATUS = "status"
    DIFF = "diff"
    LOG = "log"

@dataclass
class GitStatus:
    """Current git repository status."""
    branch: str
    modified_files: List[str]
    untracked_files: List[str]
    staged_files: List[str]
    ahead: int
    behind: int
    has_conflicts: bool
    is_clean: bool

@dataclass
class PRInfo:
    """Pull request information."""
    number: int
    title: str
    branch: str
    state: str
    url: str
    author: str
    created_at: datetime
    checks_passing: bool

class GitOperationsManager:
    """
    Manages git operations for Echo Brain with safety checks.

    Key features:
    - Automatic commit message generation
    - PR creation with descriptions
    - Safety checks before destructive operations
    - Integration with GitHub CLI
    """

    def __init__(self, repo_path: Path = Path("/opt/tower-echo-brain")):
        self.repo_path = repo_path
        self.gh_available = self._check_gh_cli()

    def _check_gh_cli(self) -> bool:
        """Check if GitHub CLI is available and authenticated."""
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.warning("GitHub CLI (gh) not installed")
            return False

    def _run_git(self, *args, check: bool = True) -> subprocess.CompletedProcess:
        """Execute git command in repo."""
        return subprocess.run(
            ["git", "-C", str(self.repo_path)] + list(args),
            capture_output=True,
            text=True,
            check=check
        )

    def _run_gh(self, *args) -> subprocess.CompletedProcess:
        """Execute GitHub CLI command."""
        return subprocess.run(
            ["gh"] + list(args),
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )

    async def get_status(self) -> GitStatus:
        """Get comprehensive git status."""
        # Get branch name
        branch_result = self._run_git("branch", "--show-current")
        branch = branch_result.stdout.strip()

        # Get file status
        status_result = self._run_git("status", "--porcelain")
        modified = []
        untracked = []
        staged = []

        for line in status_result.stdout.splitlines():
            if line.startswith("??"):
                untracked.append(line[3:])
            elif line.startswith("M ") or line.startswith(" M"):
                modified.append(line[3:])
            elif line.startswith("A ") or line.startswith("AM"):
                staged.append(line[3:])

        # Check if clean
        is_clean = len(modified) == 0 and len(untracked) == 0 and len(staged) == 0

        # Get ahead/behind info
        ahead = behind = 0
        upstream_result = self._run_git(
            "rev-list", "--count", "--left-right", f"{branch}...origin/{branch}",
            check=False
        )
        if upstream_result.returncode == 0 and upstream_result.stdout:
            parts = upstream_result.stdout.strip().split()
            if len(parts) == 2:
                ahead, behind = int(parts[0]), int(parts[1])

        # Check for conflicts
        merge_result = self._run_git("ls-files", "-u", check=False)
        has_conflicts = bool(merge_result.stdout)

        return GitStatus(
            branch=branch,
            modified_files=modified,
            untracked_files=untracked,
            staged_files=staged,
            ahead=ahead,
            behind=behind,
            has_conflicts=has_conflicts,
            is_clean=is_clean
        )

    async def smart_commit(
        self,
        files: Optional[List[str]] = None,
        message: Optional[str] = None,
        category: str = "update"
    ) -> Tuple[bool, str]:
        """
        Create a smart commit with auto-generated message if needed.

        Args:
            files: Specific files to commit (None = all changes)
            message: Commit message (None = auto-generate)
            category: Type of change (feature, fix, refactor, docs, test)

        Returns:
            Tuple of (success, commit_hash or error_message)
        """
        # Stage files
        if files:
            for file in files:
                self._run_git("add", file)
        else:
            # Stage all modified files (not untracked)
            self._run_git("add", "-u")

        # Check what's staged
        diff_result = self._run_git("diff", "--cached", "--name-status")
        if not diff_result.stdout:
            return False, "Nothing staged to commit"

        # Generate commit message if not provided
        if not message:
            message = await self._generate_commit_message(diff_result.stdout, category)

        # Add Echo Brain signature
        full_message = f"{message}\n\n🤖 Committed by Echo Brain\nAutomated commit at {datetime.now().isoformat()}"

        # Commit
        commit_result = self._run_git(
            "commit", "-m", full_message,
            check=False
        )

        if commit_result.returncode == 0:
            # Get commit hash
            hash_result = self._run_git("rev-parse", "HEAD")
            return True, hash_result.stdout.strip()[:8]
        else:
            return False, commit_result.stderr

    async def _generate_commit_message(self, diff_output: str, category: str) -> str:
        """Generate intelligent commit message from changes."""
        lines = diff_output.splitlines()

        # Count changes by type
        added = sum(1 for l in lines if l.startswith("A"))
        modified = sum(1 for l in lines if l.startswith("M"))
        deleted = sum(1 for l in lines if l.startswith("D"))

        # Get main directories affected
        dirs = set()
        for line in lines:
            parts = line.split("\t")
            if len(parts) > 1:
                path = Path(parts[1])
                if path.parts:
                    dirs.add(path.parts[0])

        # Build message
        prefixes = {
            "feature": "feat",
            "fix": "fix",
            "refactor": "refactor",
            "docs": "docs",
            "test": "test",
            "update": "update"
        }

        prefix = prefixes.get(category, "update")

        # Describe what changed
        changes = []
        if added:
            changes.append(f"{added} new file{'s' if added > 1 else ''}")
        if modified:
            changes.append(f"{modified} modification{'s' if modified > 1 else ''}")
        if deleted:
            changes.append(f"{deleted} deletion{'s' if deleted > 1 else ''}")

        scope = ""
        if len(dirs) == 1:
            scope = f"({list(dirs)[0]})"
        elif len(dirs) <= 3:
            scope = f"({'/'.join(sorted(dirs))})"

        return f"{prefix}{scope}: {', '.join(changes)}"

    async def create_feature_branch(self, feature_name: str) -> Tuple[bool, str]:
        """
        Create a new feature branch.

        Returns:
            Tuple of (success, branch_name or error)
        """
        # Sanitize branch name
        branch_name = f"echo/{feature_name.lower().replace(' ', '-')}-{datetime.now().strftime('%Y%m%d')}"
        branch_name = re.sub(r'[^a-z0-9\-/]', '', branch_name)

        # Check for uncommitted changes
        status = await self.get_status()
        if not status.is_clean:
            # Stash changes
            self._run_git("stash", "push", "-m", f"Auto-stash for {branch_name}")

        # Create and checkout branch
        result = self._run_git("checkout", "-b", branch_name, check=False)

        if result.returncode == 0:
            return True, branch_name
        else:
            return False, result.stderr

    async def create_pull_request(
        self,
        title: Optional[str] = None,
        body: Optional[str] = None,
        base: str = "main",
        draft: bool = False
    ) -> Tuple[bool, Optional[PRInfo]]:
        """
        Create a GitHub pull request using gh CLI.

        Returns:
            Tuple of (success, PRInfo or None)
        """
        if not self.gh_available:
            return False, None

        # Get current branch
        status = await self.get_status()
        if status.branch == base:
            return False, None  # Can't PR to same branch

        # Push current branch
        push_result = self._run_git("push", "-u", "origin", status.branch, check=False)
        if push_result.returncode != 0:
            logger.error(f"Failed to push: {push_result.stderr}")
            return False, None

        # Generate title if not provided
        if not title:
            # Get last commit message
            log_result = self._run_git("log", "-1", "--pretty=%s")
            title = log_result.stdout.strip()

        # Generate body if not provided
        if not body:
            body = await self._generate_pr_body(base)

        # Create PR
        args = ["pr", "create", "--title", title, "--body", body, "--base", base]
        if draft:
            args.append("--draft")

        pr_result = self._run_gh(*args)

        if pr_result.returncode == 0:
            # Parse PR URL to get number
            url = pr_result.stdout.strip()
            pr_number = url.split("/")[-1]

            # Get PR info
            info_result = self._run_gh("pr", "view", pr_number, "--json",
                "number,title,headRefName,state,url,author,createdAt")

            if info_result.returncode == 0:
                data = json.loads(info_result.stdout)
                return True, PRInfo(
                    number=data["number"],
                    title=data["title"],
                    branch=data["headRefName"],
                    state=data["state"],
                    url=data["url"],
                    author=data["author"]["login"],
                    created_at=datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00")),
                    checks_passing=True  # Would need separate query
                )

        return False, None

    async def _generate_pr_body(self, base_branch: str) -> str:
        """Generate PR description from commits and changes."""
        # Get commits since base
        log_result = self._run_git(
            "log", f"{base_branch}..HEAD",
            "--pretty=format:- %s", "--reverse"
        )

        # Get file changes summary
        diff_result = self._run_git(
            "diff", f"{base_branch}...HEAD",
            "--stat", "--stat-width=100"
        )

        # Count changes
        changes = diff_result.stdout.strip().split("\n")[-1] if diff_result.stdout else ""

        body = f"""## Summary
This PR was created automatically by Echo Brain.

## Changes
{log_result.stdout}

## Files Modified
{changes}

## Testing
- [ ] Code has been tested
- [ ] No breaking changes
- [ ] Documentation updated if needed

---
🤖 Generated by Echo Brain at {datetime.now().isoformat()}
"""
        return body

    async def sync_with_remote(self) -> Tuple[bool, str]:
        """
        Sync with remote repository (pull/push as needed).

        Returns:
            Tuple of (success, message)
        """
        status = await self.get_status()

        # If behind, pull
        if status.behind > 0:
            pull_result = self._run_git("pull", "--rebase", check=False)
            if pull_result.returncode != 0:
                return False, f"Pull failed: {pull_result.stderr}"

        # If ahead, push
        if status.ahead > 0:
            push_result = self._run_git("push", check=False)
            if push_result.returncode != 0:
                return False, f"Push failed: {push_result.stderr}"

        return True, f"Synced (pulled {status.behind}, pushed {status.ahead} commits)"

    async def auto_document_changes(self) -> Tuple[bool, str]:
        """
        Automatically generate documentation for recent changes.

        Returns:
            Tuple of (success, doc_path or error)
        """
        # Get recent commits
        log_result = self._run_git(
            "log", "--since='1 week ago'",
            "--pretty=format:%h %s", "--reverse"
        )

        if not log_result.stdout:
            return False, "No recent changes to document"

        # Generate changelog
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_path = self.repo_path / f"docs/CHANGELOG_{timestamp}.md"
        doc_path.parent.mkdir(exist_ok=True)

        content = f"""# Echo Brain Changelog
Generated: {datetime.now().isoformat()}

## Recent Changes

"""
        for line in log_result.stdout.splitlines():
            content += f"- {line}\n"

        # Add file change summary
        diff_result = self._run_git(
            "diff", "--stat", "HEAD~10...HEAD",
            check=False
        )

        if diff_result.stdout:
            content += f"\n## File Statistics\n```\n{diff_result.stdout}```\n"

        # Write documentation
        doc_path.write_text(content)

        return True, str(doc_path)


class GitHubOperations:
    """
    GitHub-specific operations using gh CLI.
    """

    def __init__(self, repo_path: Path = Path("/opt/tower-echo-brain")):
        self.repo_path = repo_path

    def _run_gh(self, *args) -> subprocess.CompletedProcess:
        """Execute gh command."""
        return subprocess.run(
            ["gh"] + list(args),
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )

    async def list_issues(self, state: str = "open") -> List[Dict[str, Any]]:
        """List GitHub issues."""
        result = self._run_gh(
            "issue", "list",
            "--state", state,
            "--json", "number,title,author,createdAt,labels"
        )

        if result.returncode == 0:
            return json.loads(result.stdout)
        return []

    async def create_issue(
        self,
        title: str,
        body: str,
        labels: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[int]]:
        """Create a GitHub issue."""
        args = ["issue", "create", "--title", title, "--body", body]

        if labels:
            args.extend(["--label", ",".join(labels)])

        result = self._run_gh(*args)

        if result.returncode == 0:
            # Extract issue number from URL
            url = result.stdout.strip()
            issue_number = int(url.split("/")[-1])
            return True, issue_number

        return False, None

    async def list_workflows(self) -> List[Dict[str, Any]]:
        """List GitHub Actions workflows."""
        result = self._run_gh(
            "workflow", "list",
            "--json", "name,state,id"
        )

        if result.returncode == 0:
            return json.loads(result.stdout)
        return []

    async def trigger_workflow(self, workflow_name: str) -> bool:
        """Trigger a GitHub Actions workflow."""
        result = self._run_gh(
            "workflow", "run", workflow_name,
            check=False
        )
        return result.returncode == 0

    async def check_pr_status(self, pr_number: int) -> Dict[str, Any]:
        """Check PR status including checks."""
        result = self._run_gh(
            "pr", "checks", str(pr_number),
            "--json", "name,status,conclusion"
        )

        if result.returncode == 0:
            checks = json.loads(result.stdout)

            # Summarize
            passing = all(c.get("conclusion") == "success" for c in checks)
            pending = any(c.get("status") == "pending" for c in checks)

            return {
                "checks": checks,
                "all_passing": passing,
                "has_pending": pending
            }

        return {"checks": [], "all_passing": False, "has_pending": False}


async def test_git_operations():
    """Test git operations."""
    git_ops = GitOperationsManager()
    gh_ops = GitHubOperations()

    # Test status
    status = await git_ops.get_status()
    print(f"Git Status:")
    print(f"  Branch: {status.branch}")
    print(f"  Modified: {len(status.modified_files)} files")
    print(f"  Untracked: {len(status.untracked_files)} files")
    print(f"  Clean: {status.is_clean}")

    # Test GitHub operations if available
    if git_ops.gh_available:
        print("\nGitHub Integration: ✅ Available")

        # List recent issues
        issues = await gh_ops.list_issues()
        print(f"  Open Issues: {len(issues)}")

        # List workflows
        workflows = await gh_ops.list_workflows()
        print(f"  Workflows: {len(workflows)}")
    else:
        print("\nGitHub Integration: ❌ Not available")

    print("\n✅ Git operations test complete")


if __name__ == "__main__":
    asyncio.run(test_git_operations())
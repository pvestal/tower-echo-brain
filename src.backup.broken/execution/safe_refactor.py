#!/usr/bin/env python3
"""
Git-integrated refactoring with automatic rollback on failure.
Ensures no code changes are permanent until verified.
"""

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Optional
import uuid

@dataclass
class RefactorResult:
    """Result of a refactoring operation."""
    success: bool
    files_modified: list[Path]
    commit_hash: Optional[str]
    rollback_available: bool
    error_message: Optional[str] = None

class SafeRefactor:
    """
    Performs code refactoring with git safety net.

    Workflow:
    1. Create branch for changes
    2. Apply refactoring
    3. Run tests
    4. If tests pass: merge or leave for review
    5. If tests fail: automatic rollback
    """

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self._verify_git_repo()

    def _verify_git_repo(self) -> None:
        """Ensure we're in a git repository."""
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=self.repo_root,
            capture_output=True
        )
        if result.returncode != 0:
            raise ValueError(f"{self.repo_root} is not a git repository")

    def _run_git(self, *args) -> subprocess.CompletedProcess:
        """Execute git command in repo."""
        return subprocess.run(
            ["git"] + list(args),
            cwd=self.repo_root,
            capture_output=True,
            text=True
        )

    def create_refactor_branch(self, description: str) -> str:
        """Create isolated branch for refactoring."""
        branch_name = f"echo-refactor/{uuid.uuid4().hex[:8]}-{description.replace(' ', '-')[:20]}"

        # Stash any uncommitted changes
        self._run_git("stash", "push", "-m", "echo-auto-stash")

        # Create and checkout branch
        result = self._run_git("checkout", "-b", branch_name)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create branch: {result.stderr}")

        return branch_name

    def apply_changes(self, modified_files: list[Path], commit_message: str) -> str:
        """Stage and commit changes."""
        for file in modified_files:
            self._run_git("add", str(file))

        result = self._run_git("commit", "-m", commit_message)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to commit: {result.stderr}")

        # Get commit hash
        result = self._run_git("rev-parse", "HEAD")
        return result.stdout.strip()

    def run_tests(self, test_command: list[str]) -> bool:
        """Run test suite to verify changes don't break anything."""
        try:
            result = subprocess.run(
                test_command,
                cwd=self.repo_root,
                capture_output=True,
                timeout=300  # 5 minute timeout
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

    def rollback(self, original_branch: str = "main") -> None:
        """Abandon refactor branch and return to original state."""
        current = self._run_git("branch", "--show-current").stdout.strip()

        # Checkout original branch
        self._run_git("checkout", original_branch)

        # Delete refactor branch
        if current.startswith("echo-refactor/"):
            self._run_git("branch", "-D", current)

        # Restore stashed changes if any
        self._run_git("stash", "pop")

    def refactor_with_safety(
        self,
        description: str,
        refactor_func,  # Callable that modifies files and returns list[Path]
        test_command: list[str],
        original_branch: str = "main"
    ) -> RefactorResult:
        """
        Execute refactoring with full safety net.

        Args:
            description: What this refactoring does
            refactor_func: Function that performs changes, returns modified file paths
            test_command: Command to run tests (e.g., ["pytest", "tests/"])
            original_branch: Branch to return to on failure

        Returns:
            RefactorResult with success status and commit info
        """
        branch_name = None

        try:
            # Step 1: Create isolated branch
            branch_name = self.create_refactor_branch(description)

            # Step 2: Apply refactoring
            modified_files = refactor_func()

            if not modified_files:
                self.rollback(original_branch)
                return RefactorResult(
                    success=False,
                    files_modified=[],
                    commit_hash=None,
                    rollback_available=False,
                    error_message="Refactoring produced no changes"
                )

            # Step 3: Commit changes
            commit_hash = self.apply_changes(
                modified_files,
                f"[Echo Brain] {description}"
            )

            # Step 4: Run tests
            tests_passed = self.run_tests(test_command)

            if tests_passed:
                return RefactorResult(
                    success=True,
                    files_modified=modified_files,
                    commit_hash=commit_hash,
                    rollback_available=True
                )
            else:
                # Tests failed - rollback
                self.rollback(original_branch)
                return RefactorResult(
                    success=False,
                    files_modified=modified_files,
                    commit_hash=None,
                    rollback_available=False,
                    error_message="Tests failed after refactoring - changes rolled back"
                )

        except Exception as e:
            # Something went wrong - try to rollback
            if branch_name:
                try:
                    self.rollback(original_branch)
                except:
                    pass

            return RefactorResult(
                success=False,
                files_modified=[],
                commit_hash=None,
                rollback_available=False,
                error_message=str(e)
            )


def test_safe_refactor():
    """Test safe refactoring with git integration."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path)

        # Create initial file
        test_file = repo_path / "test.py"
        test_file.write_text("def hello():\n    return 'world'")
        subprocess.run(["git", "add", "."], cwd=repo_path)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path)

        # Create refactor instance
        refactor = SafeRefactor(repo_path)

        # Test successful refactoring
        def mock_refactor():
            test_file.write_text("def hello():\n    return 'refactored'")
            return [test_file]

        # Mock test that passes
        result = refactor.refactor_with_safety(
            description="test-refactor",
            refactor_func=mock_refactor,
            test_command=["true"],  # Always passes
            original_branch="master"
        )

        assert result.success, f"Refactoring should succeed: {result.error_message}"
        assert len(result.files_modified) == 1, "Should have modified one file"
        assert result.commit_hash is not None, "Should have commit hash"

        print("✅ Safe refactoring with git integration working")


if __name__ == "__main__":
    test_safe_refactor()
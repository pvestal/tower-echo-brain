"""Git operations manager — wraps git/gh CLI for Echo Brain."""
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_REPO = Path("/opt/tower-echo-brain")


@dataclass
class GitStatus:
    branch: str
    modified_files: List[str]
    untracked_files: List[str]
    staged_files: List[str]
    ahead: int
    behind: int
    has_conflicts: bool
    is_clean: bool


@dataclass
class PullRequestInfo:
    number: int
    title: str
    url: str
    branch: str
    state: str


async def _run(cmd: List[str], cwd: Path) -> Tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode().strip(), stderr.decode().strip()


class GitOperationsManager:
    """Git operations for a single repository."""

    def __init__(self, repo_path: Path = None):
        self.repo_path = repo_path or DEFAULT_REPO

    async def get_status(self) -> GitStatus:
        rc, branch_out, _ = await _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], self.repo_path)
        branch = branch_out if rc == 0 else "unknown"

        rc, porcelain, _ = await _run(["git", "status", "--porcelain"], self.repo_path)
        modified, untracked, staged = [], [], []
        for line in porcelain.splitlines():
            if not line:
                continue
            idx, wt = line[0], line[1]
            fname = line[3:]
            if idx in ("M", "A", "D", "R"):
                staged.append(fname)
            if wt == "M":
                modified.append(fname)
            elif line[:2] == "??":
                untracked.append(fname)

        ahead, behind = 0, 0
        rc, ab_out, _ = await _run(
            ["git", "rev-list", "--left-right", "--count", f"HEAD...@{{u}}"], self.repo_path
        )
        if rc == 0 and "\t" in ab_out:
            parts = ab_out.split("\t")
            ahead, behind = int(parts[0]), int(parts[1])

        has_conflicts = any(line.startswith("UU") or line.startswith("AA") for line in porcelain.splitlines())
        is_clean = len(porcelain.strip()) == 0

        return GitStatus(
            branch=branch,
            modified_files=modified,
            untracked_files=untracked,
            staged_files=staged,
            ahead=ahead,
            behind=behind,
            has_conflicts=has_conflicts,
            is_clean=is_clean,
        )

    async def smart_commit(
        self,
        files: Optional[List[str]] = None,
        message: Optional[str] = None,
        category: str = "update",
    ) -> Tuple[bool, str]:
        if files:
            for f in files:
                await _run(["git", "add", f], self.repo_path)
        else:
            await _run(["git", "add", "-A"], self.repo_path)

        if not message:
            rc, diff_stat, _ = await _run(["git", "diff", "--cached", "--stat"], self.repo_path)
            n_files = len(diff_stat.strip().splitlines()) - 1 if diff_stat.strip() else 0
            message = f"[echo-brain:{category}] Update {n_files} file(s)"

        rc, out, err = await _run(["git", "commit", "-m", message], self.repo_path)
        if rc != 0:
            return False, err or out

        rc, hash_out, _ = await _run(["git", "rev-parse", "--short", "HEAD"], self.repo_path)
        return True, hash_out

    async def create_feature_branch(self, name: str) -> Tuple[bool, str]:
        branch_name = f"feature/{name}"
        rc, out, err = await _run(["git", "checkout", "-b", branch_name], self.repo_path)
        if rc != 0:
            return False, err or out
        return True, branch_name

    async def create_pull_request(
        self,
        title: Optional[str] = None,
        body: Optional[str] = None,
        base: str = "main",
        draft: bool = False,
    ) -> Tuple[bool, Optional[PullRequestInfo]]:
        status = await self.get_status()
        if not title:
            title = f"[Echo Brain] Updates on {status.branch}"

        cmd = ["gh", "pr", "create", "--title", title, "--base", base]
        if body:
            cmd += ["--body", body]
        if draft:
            cmd.append("--draft")

        rc, out, err = await _run(cmd, self.repo_path)
        if rc != 0:
            logger.error(f"gh pr create failed: {err}")
            return False, None

        # out is the PR URL; parse PR number from it
        pr_url = out.strip()
        pr_number = 0
        if "/" in pr_url:
            try:
                pr_number = int(pr_url.rstrip("/").rsplit("/", 1)[-1])
            except ValueError:
                pass

        return True, PullRequestInfo(
            number=pr_number,
            title=title,
            url=pr_url,
            branch=status.branch,
            state="open",
        )


class GitHubOperations:
    """GitHub-specific helpers (delegated to by GitOperationsManager)."""

    def __init__(self):
        self.repo_path = DEFAULT_REPO

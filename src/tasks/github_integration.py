"""GitHub integration via gh CLI."""
import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

DEFAULT_REPO = Path("/opt/tower-echo-brain")


class GitHubIntegration:
    """GitHub integration using the gh CLI."""

    def __init__(self):
        self._repo_path = DEFAULT_REPO

    async def check_auth(self) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                "gh", "auth", "status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return proc.returncode == 0
        except FileNotFoundError:
            return False

    async def get_open_prs(self) -> List[Dict[str, Any]]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "gh", "pr", "list", "--json", "number,title,url,state",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._repo_path),
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0 and stdout:
                return json.loads(stdout.decode())
        except Exception as e:
            logger.debug(f"get_open_prs failed: {e}")
        return []

    def get_current_branch(self) -> str:
        """Synchronous — called without await in the API."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, cwd=str(self._repo_path),
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"


github_integration = GitHubIntegration()

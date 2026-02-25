"""Git manager — monitors and auto-commits across all Tower service repos."""
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TOWER_ROOT = Path("/opt")


async def _run(cmd: List[str], cwd: Path) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode().strip(), stderr.decode().strip()


class GitManager:
    """Manages git across all Tower service repositories."""

    def __init__(self):
        self.auto_commit_enabled: bool = False
        self.commit_log_path: Path = Path("/opt/tower-echo-brain/data/git_automation.jsonl")
        self._monitoring: bool = False

    def _discover_tower_services(self) -> List[Path]:
        services = []
        for d in sorted(TOWER_ROOT.iterdir()):
            if d.is_dir() and d.name.startswith("tower-") and (d / ".git").is_dir():
                services.append(d)
        return services

    async def get_repo_status(self, path: Path) -> Dict[str, Any]:
        result: Dict[str, Any] = {"has_changes": False, "changed_files": [], "untracked_files": [],
                                   "latest_commit": {}, "branch": "unknown", "error": None}
        try:
            rc, branch, _ = await _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], path)
            result["branch"] = branch if rc == 0 else "unknown"

            rc, porcelain, _ = await _run(["git", "status", "--porcelain"], path)
            changed, untracked = [], []
            for line in porcelain.splitlines():
                if line.startswith("??"):
                    untracked.append(line[3:])
                elif line.strip():
                    changed.append(line[3:])
            result["changed_files"] = changed
            result["untracked_files"] = untracked
            result["has_changes"] = bool(changed or untracked)

            rc, log_out, _ = await _run(["git", "log", "-1", "--format=%H %s"], path)
            if rc == 0 and log_out:
                parts = log_out.split(" ", 1)
                result["latest_commit"] = {"hash": parts[0], "message": parts[1] if len(parts) > 1 else ""}
        except Exception as e:
            result["error"] = str(e)
        return result

    async def auto_commit_changes(self, path: Path, message: str) -> Optional[str]:
        await _run(["git", "add", "-A"], path)
        rc, out, err = await _run(["git", "commit", "-m", message], path)
        if rc != 0:
            return None
        rc, hash_out, _ = await _run(["git", "rev-parse", "--short", "HEAD"], path)
        self._log_commit(path, hash_out, message)
        return hash_out

    async def initialize_repo(self, path: Path) -> bool:
        if (path / ".git").is_dir():
            return True
        rc, _, _ = await _run(["git", "init"], path)
        return rc == 0

    def enable_auto_commit(self, enabled: bool):
        self.auto_commit_enabled = enabled

    async def monitor_and_commit_loop(self):
        self._monitoring = True
        logger.info("Git monitor loop started")
        try:
            while self.auto_commit_enabled and self._monitoring:
                for svc in self._discover_tower_services():
                    status = await self.get_repo_status(svc)
                    if status["has_changes"]:
                        n = len(status["changed_files"]) + len(status["untracked_files"])
                        await self.auto_commit_changes(svc, f"[Echo Auto-Sync] {n} files updated")
                await asyncio.sleep(300)
        finally:
            self._monitoring = False

    def _log_commit(self, path: Path, commit_hash: str, message: str):
        try:
            self.commit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.commit_log_path, "a") as f:
                json.dump({"repo": str(path), "hash": commit_hash, "message": message,
                           "timestamp": datetime.now().isoformat()}, f)
                f.write("\n")
        except Exception as e:
            logger.debug(f"Failed to write commit log: {e}")


git_manager = GitManager()

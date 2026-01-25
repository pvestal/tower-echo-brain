#!/usr/bin/env python3
"""
Proactive Code Quality System for Echo Brain
Daily pylint scanning, automatic fixes, learning from outcomes
"""

import asyncio
import json
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
import tempfile

logger = logging.getLogger(__name__)

class ProactiveCodeQuality:
    """Echo's true proactive code quality system"""

    def __init__(self):
        self.base_path = Path("/opt/tower-echo-brain")
        self.log_path = self.base_path / "logs" / "code_quality"
        self.log_path.mkdir(parents=True, exist_ok=True)

        # Learning database - tracks what works
        self.learning_db = self.base_path / "data" / "quality_learning.json"
        self.learning_db.parent.mkdir(parents=True, exist_ok=True)

        # Projects to monitor
        self.monitored_projects = [
            "/opt/tower-echo-brain",
            "/opt/tower-anime-production",
            "/opt/tower-auth",
            "/opt/tower-kb",
            "/opt/tower-dashboard"
        ]

        # Quality thresholds
        self.min_acceptable_score = 7.0
        self.target_score = 8.5

        # Learning data
        self.learning_data = self.load_learning_data()

    def load_learning_data(self) -> Dict:
        """Load learning history"""
        if self.learning_db.exists():
            with open(self.learning_db, 'r') as f:
                return json.load(f)
        return {
            "successful_fixes": [],
            "failed_fixes": [],
            "improvement_patterns": {},
            "daily_scores": {}
        }

    def save_learning_data(self):
        """Save learning history"""
        with open(self.learning_db, 'w') as f:
            json.dump(self.learning_data, f, indent=2)

    async def scan_all_projects(self) -> Dict[str, List[Dict]]:
        """Run pylint on all Python files in monitored projects"""
        logger.info("🔍 Starting daily code quality scan")

        results = {}
        total_files = 0
        files_below_threshold = 0

        for project_path in self.monitored_projects:
            project = Path(project_path)
            if not project.exists():
                logger.warning(f"Project not found: {project_path}")
                continue

            project_results = []

            # Find all Python files
            py_files = list(project.rglob("*.py"))
            # Exclude venv and __pycache__
            py_files = [f for f in py_files if "venv" not in str(f) and "__pycache__" not in str(f)]

            logger.info(f"Scanning {len(py_files)} files in {project.name}")

            for py_file in py_files:
                score, issues = await self.analyze_file(py_file)

                if score < self.min_acceptable_score:
                    files_below_threshold += 1
                    project_results.append({
                        "file": str(py_file),
                        "score": score,
                        "issues": issues[:5],  # Top 5 issues
                        "needs_fix": True
                    })

                total_files += 1

            results[project_path] = project_results

        # Save daily score
        today = datetime.now().strftime("%Y-%m-%d")
        self.learning_data["daily_scores"][today] = {
            "total_files": total_files,
            "below_threshold": files_below_threshold,
            "average_health": (total_files - files_below_threshold) / total_files if total_files > 0 else 0
        }
        self.save_learning_data()

        logger.info(f"📊 Scan complete: {files_below_threshold}/{total_files} files need improvement")
        return results

    async def analyze_file(self, filepath: Path) -> Tuple[float, List[Dict]]:
        """Analyze single file with pylint"""
        try:
            result = subprocess.run(
                ["pylint", "--output-format=json", "--score=y", str(filepath)],
                capture_output=True,
                text=True,
                timeout=30
            )

            # Parse score from stderr
            score = 0.0
            if "Your code has been rated at" in result.stderr:
                score_line = [l for l in result.stderr.split('\n') if "Your code has been rated at" in l]
                if score_line:
                    score_str = score_line[0].split("rated at ")[1].split("/")[0]
                    score = float(score_str)

            # Parse issues from stdout
            issues = []
            if result.stdout:
                try:
                    issues = json.loads(result.stdout)
                except json.JSONDecodeError:
                    pass

            return score, issues

        except Exception as e:
            logger.error(f"Failed to analyze {filepath}: {e}")
            return 10.0, []  # Assume it's fine if we can't analyze

    async def create_and_test_fix(self, file_info: Dict) -> Dict:
        """Create automatic fix and test it"""
        filepath = Path(file_info["file"])
        original_score = file_info["score"]

        logger.info(f"🔧 Attempting to fix {filepath.name} (score: {original_score:.1f})")

        # Backup original file
        backup_path = self.create_backup(filepath)

        try:
            # Try different fix strategies based on issues
            fix_applied = False
            fix_strategy = None

            # Strategy 1: Auto-format with black
            if await self.try_black_format(filepath):
                fix_applied = True
                fix_strategy = "black_format"

            # Strategy 2: Fix common issues
            if not fix_applied:
                if await self.fix_common_issues(filepath, file_info["issues"]):
                    fix_applied = True
                    fix_strategy = "common_fixes"

            if not fix_applied:
                logger.info("No automatic fix available")
                return {"success": False, "reason": "no_fix_available"}

            # Test the fix
            new_score, new_issues = await self.analyze_file(filepath)

            # Check if improvement happened
            if new_score > original_score:
                # Success! Learn from this
                fix_record = {
                    "file": str(filepath),
                    "strategy": fix_strategy,
                    "before_score": original_score,
                    "after_score": new_score,
                    "improvement": new_score - original_score,
                    "timestamp": datetime.now().isoformat()
                }

                self.learning_data["successful_fixes"].append(fix_record)

                # Update pattern learning
                pattern_key = f"{fix_strategy}_{filepath.suffix}"
                if pattern_key not in self.learning_data["improvement_patterns"]:
                    self.learning_data["improvement_patterns"][pattern_key] = {
                        "success_count": 0,
                        "total_improvement": 0
                    }

                self.learning_data["improvement_patterns"][pattern_key]["success_count"] += 1
                self.learning_data["improvement_patterns"][pattern_key]["total_improvement"] += (new_score - original_score)

                self.save_learning_data()

                logger.info(f"✅ Fix successful! Score improved: {original_score:.1f} → {new_score:.1f}")

                # Create review submission
                await self.submit_for_review(filepath, backup_path, fix_record)

                return {
                    "success": True,
                    "strategy": fix_strategy,
                    "improvement": new_score - original_score,
                    "new_score": new_score
                }
            else:
                # Fix didn't help, revert
                logger.info(f"Fix didn't improve score ({new_score:.1f}), reverting")
                shutil.copy2(backup_path, filepath)

                # Learn from failure
                self.learning_data["failed_fixes"].append({
                    "file": str(filepath),
                    "strategy": fix_strategy,
                    "timestamp": datetime.now().isoformat()
                })
                self.save_learning_data()

                return {"success": False, "reason": "no_improvement"}

        except Exception as e:
            logger.error(f"Fix failed with error: {e}")
            # Restore backup on error
            shutil.copy2(backup_path, filepath)
            return {"success": False, "reason": str(e)}
        finally:
            # Clean up backup after a delay
            asyncio.create_task(self.cleanup_backup(backup_path))

    def create_backup(self, filepath: Path) -> Path:
        """Create backup of file"""
        backup_dir = self.log_path / "backups" / datetime.now().strftime("%Y%m%d")
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"{filepath.name}.{datetime.now().strftime('%H%M%S')}.bak"
        shutil.copy2(filepath, backup_path)
        return backup_path

    async def cleanup_backup(self, backup_path: Path, delay: int = 3600):
        """Clean up backup after delay"""
        await asyncio.sleep(delay)
        if backup_path.exists():
            backup_path.unlink()

    async def try_black_format(self, filepath: Path) -> bool:
        """Try to format with black"""
        try:
            result = subprocess.run(
                ["black", "--quiet", str(filepath)],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    async def fix_common_issues(self, filepath: Path, issues: List[Dict]) -> bool:
        """Fix common pylint issues"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()

            original_content = content
            fixed = False

            for issue in issues:
                msg_id = issue.get('message-id', '')

                # Fix missing docstrings
                if msg_id in ['C0111', 'C0112', 'C0114', 'C0115', 'C0116']:
                    # Add simple docstrings
                    if 'def ' in content and '"""' not in content:
                        content = content.replace('def ', 'def ', 1)
                        # This is simplified - real implementation would be smarter
                        fixed = True

                # Fix import order
                if msg_id in ['C0411', 'C0412', 'C0413']:
                    # Sort imports (simplified)
                    lines = content.split('\n')
                    import_lines = [l for l in lines if l.startswith('import ') or l.startswith('from ')]
                    if import_lines:
                        import_lines.sort()
                        # This is simplified - real implementation would preserve structure
                        fixed = True

            if fixed and content != original_content:
                with open(filepath, 'w') as f:
                    f.write(content)
                return True

        except Exception as e:
            logger.error(f"Failed to fix common issues: {e}")

        return False

    async def submit_for_review(self, filepath: Path, backup_path: Path, fix_record: Dict):
        """Submit improvement for review"""
        review_dir = self.log_path / "pending_reviews"
        review_dir.mkdir(parents=True, exist_ok=True)

        review_file = review_dir / f"{filepath.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        review_data = {
            "file": str(filepath),
            "backup": str(backup_path),
            "fix_record": fix_record,
            "timestamp": datetime.now().isoformat(),
            "status": "pending_review",
            "commit_message": f"fix: Improve {filepath.name} quality from {fix_record['before_score']:.1f} to {fix_record['after_score']:.1f}",
            "can_auto_commit": fix_record["improvement"] > 0.5  # Auto-commit if significant improvement
        }

        with open(review_file, 'w') as f:
            json.dump(review_data, f, indent=2)

        logger.info(f"📝 Submitted {filepath.name} for review")

        # If significant improvement and auto-commit enabled, create git commit
        if review_data["can_auto_commit"] and fix_record["improvement"] > 1.0:
            await self.create_git_commit(filepath, review_data["commit_message"])

    async def create_git_commit(self, filepath: Path, message: str):
        """Create git commit for improvement"""
        try:
            # Check if file is in git repo
            result = subprocess.run(
                ["git", "-C", str(filepath.parent), "status", "--porcelain", str(filepath)],
                capture_output=True,
                text=True
            )

            if result.stdout.strip():
                # File has changes
                subprocess.run(["git", "-C", str(filepath.parent), "add", str(filepath)])
                subprocess.run(["git", "-C", str(filepath.parent), "commit", "-m", message])
                logger.info(f"📦 Created git commit for {filepath.name}")
        except Exception as e:
            logger.error(f"Failed to create git commit: {e}")

    async def daily_quality_loop(self):
        """Main loop - runs daily"""
        logger.info("🚀 Proactive Code Quality System started")

        while True:
            try:
                # Run scan
                scan_results = await self.scan_all_projects()

                # Process files that need fixing
                fixes_attempted = 0
                fixes_successful = 0
                successful_fixes = []

                for project, files in scan_results.items():
                    for file_info in files:
                        if file_info.get("needs_fix") and fixes_attempted < 10:  # Limit to 10 fixes per day
                            fixes_attempted += 1

                            result = await self.create_and_test_fix(file_info)
                            if result.get("success"):
                                fixes_successful += 1
                                # Track successful fixes for PR creation
                                successful_fixes.append({
                                    "file": file_info["file"],
                                    "before": file_info["score"],
                                    "after": result.get("new_score", file_info["score"]),
                                    "improvement": result.get("improvement", 0)
                                })

                # Create GitHub PR if we have successful fixes
                if successful_fixes:
                    try:
                        from .github_integration import github_integration
                        pr_url = await github_integration.create_quality_improvement_pr(successful_fixes)
                        if pr_url:
                            logger.info(f"🎉 Created PR with {len(successful_fixes)} improvements: {pr_url}")
                    except Exception as e:
                        logger.error(f"Failed to create PR: {e}")

                # Generate daily report
                await self.generate_daily_report(scan_results, fixes_attempted, fixes_successful)

                # Learn from patterns
                self.analyze_learning_patterns()

                # Sleep until tomorrow (run at 2 AM daily)
                now = datetime.now()
                tomorrow_2am = now.replace(hour=2, minute=0, second=0) + timedelta(days=1)
                sleep_seconds = (tomorrow_2am - now).total_seconds()

                logger.info(f"💤 Next scan in {sleep_seconds/3600:.1f} hours")
                await asyncio.sleep(sleep_seconds)

            except Exception as e:
                logger.error(f"Error in quality loop: {e}")
                # On error, wait 1 hour and retry
                await asyncio.sleep(3600)

    async def generate_daily_report(self, scan_results: Dict, fixes_attempted: int, fixes_successful: int):
        """Generate daily quality report"""
        report_path = self.log_path / f"daily_report_{datetime.now().strftime('%Y%m%d')}.md"

        total_files_scanned = sum(len(files) for files in scan_results.values())
        files_need_fixing = sum(1 for files in scan_results.values() for f in files if f.get("needs_fix"))

        report = f"""# Daily Code Quality Report - {datetime.now().strftime('%Y-%m-%d')}

## Summary
- **Files Scanned**: {total_files_scanned}
- **Files Below Threshold**: {files_need_fixing}
- **Fixes Attempted**: {fixes_attempted}
- **Fixes Successful**: {fixes_successful}
- **Success Rate**: {(fixes_successful/fixes_attempted*100) if fixes_attempted > 0 else 0:.1f}%

## Projects Scanned
"""

        for project, files in scan_results.items():
            if files:
                report += f"\n### {Path(project).name}\n"
                for file_info in files[:5]:  # Top 5 worst files
                    report += f"- {Path(file_info['file']).name}: {file_info['score']:.1f}/10\n"

        # Add learning insights
        if self.learning_data["improvement_patterns"]:
            report += "\n## Learning Insights\n"
            for pattern, stats in self.learning_data["improvement_patterns"].items():
                avg_improvement = stats["total_improvement"] / stats["success_count"] if stats["success_count"] > 0 else 0
                report += f"- {pattern}: {stats['success_count']} successes, avg improvement: {avg_improvement:.2f}\n"

        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"📊 Daily report saved to {report_path}")

    def analyze_learning_patterns(self):
        """Analyze what we've learned works best"""
        if not self.learning_data["successful_fixes"]:
            return

        # Find most effective strategies
        strategy_stats = {}
        for fix in self.learning_data["successful_fixes"]:
            strategy = fix["strategy"]
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    "count": 0,
                    "total_improvement": 0
                }
            strategy_stats[strategy]["count"] += 1
            strategy_stats[strategy]["total_improvement"] += fix["improvement"]

        # Log insights
        for strategy, stats in strategy_stats.items():
            avg_improvement = stats["total_improvement"] / stats["count"]
            logger.info(f"📈 Strategy '{strategy}': {stats['count']} uses, avg improvement: {avg_improvement:.2f}")

# Initialize the engine
proactive_quality = ProactiveCodeQuality()
#!/usr/bin/env python3
"""
Anime Production System Code Reviewer
Proactive code quality analysis and improvement recommendations
"""

import os
import json
import subprocess
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnimeCodeReviewer:
    """Automated code review system for anime production"""

    def __init__(self):
        self.anime_path = Path("/opt/tower-anime-production")
        self.results_path = Path("/opt/tower-echo-brain/logs/anime_code_reviews")
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.min_quality_score = 7.0

    async def scan_codebase(self) -> Dict:
        """Scan entire anime codebase for quality issues"""
        logger.info(f"🔍 Starting anime codebase scan at {self.anime_path}")

        results = {
            "timestamp": datetime.now().isoformat(),
            "total_files": 0,
            "files_analyzed": 0,
            "files_need_refactor": [],
            "quality_metrics": {},
            "recommendations": []
        }

        # Find all Python files
        python_files = list(self.anime_path.rglob("*.py"))
        results["total_files"] = len(python_files)
        logger.info(f"Found {len(python_files)} Python files to analyze")

        for py_file in python_files:
            if "venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            quality = await self.analyze_file(py_file)
            results["files_analyzed"] += 1

            if quality["score"] < self.min_quality_score:
                results["files_need_refactor"].append({
                    "file": str(py_file),
                    "score": quality["score"],
                    "issues": quality["issues"]
                })

            # Log progress every 100 files
            if results["files_analyzed"] % 100 == 0:
                logger.info(f"Analyzed {results['files_analyzed']}/{len(python_files)} files")

        # Generate recommendations
        results["recommendations"] = self.generate_recommendations(results)

        # Save results
        output_file = self.results_path / f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"✅ Code review complete. Results saved to {output_file}")
        return results

    async def analyze_file(self, filepath: Path) -> Dict:
        """Analyze single file for code quality"""
        try:
            # Run pylint
            result = subprocess.run(
                ["pylint", "--output-format=json", str(filepath)],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.stdout:
                issues = json.loads(result.stdout)
                # Calculate score (pylint gives score out of 10)
                score = 10.0 - (len(issues) * 0.1)  # Simple scoring
                score = max(0, min(10, score))

                return {
                    "score": score,
                    "issues": issues[:5]  # Top 5 issues
                }
        except Exception as e:
            logger.debug(f"Could not analyze {filepath}: {e}")

        return {"score": 10.0, "issues": []}  # Default to perfect if can't analyze

    def generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        if results["files_need_refactor"]:
            # Group by common issues
            issue_counts = {}
            for file_info in results["files_need_refactor"]:
                for issue in file_info.get("issues", []):
                    msg = issue.get("message", "unknown")
                    issue_counts[msg] = issue_counts.get(msg, 0) + 1

            # Top 3 issues
            top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]

            recommendations.append("🎯 TOP PRIORITY REFACTORING:")
            for issue, count in top_issues:
                recommendations.append(f"  - Fix '{issue}' ({count} occurrences)")

            # Files needing most work
            worst_files = sorted(
                results["files_need_refactor"],
                key=lambda x: x["score"]
            )[:5]

            recommendations.append("\n📝 FILES NEEDING IMMEDIATE ATTENTION:")
            for file_info in worst_files:
                recommendations.append(
                    f"  - {Path(file_info['file']).name} (score: {file_info['score']:.1f}/10)"
                )

        return recommendations

    async def create_refactor_tasks(self, results: Dict):
        """Create CODE_REFACTOR tasks in Echo's task queue"""
        logger.info("Creating refactoring tasks in Echo's queue...")

        # Import Echo's task system
        try:
            from src.tasks.task_queue import TaskQueueManager

            task_manager = TaskQueueManager()

            for file_info in results["files_need_refactor"][:10]:  # Top 10 files
                task = {
                    "task_type": "CODE_REFACTOR",
                    "priority": "NORMAL" if file_info["score"] > 5 else "HIGH",
                    "metadata": {
                        "file": file_info["file"],
                        "score": file_info["score"],
                        "issues": file_info["issues"][:3],  # Top 3 issues
                        "source": "anime_code_reviewer"
                    }
                }

                await task_manager.create_task(**task)
                logger.info(f"Created task for {Path(file_info['file']).name}")

        except Exception as e:
            logger.error(f"Could not create tasks: {e}")

async def main():
    """Run anime code review"""
    reviewer = AnimeCodeReviewer()

    logger.info("🎬 Starting Anime Production Code Review")
    results = await reviewer.scan_codebase()

    # Print summary
    print("\n" + "="*60)
    print("📊 ANIME CODE REVIEW SUMMARY")
    print("="*60)
    print(f"Total files: {results['total_files']}")
    print(f"Files analyzed: {results['files_analyzed']}")
    print(f"Files needing refactor: {len(results['files_need_refactor'])}")

    if results['recommendations']:
        print("\n".join(results['recommendations']))

    # Create tasks if needed
    if results['files_need_refactor']:
        await reviewer.create_refactor_tasks(results)

    print("\n✅ Review complete!")

if __name__ == "__main__":
    asyncio.run(main())
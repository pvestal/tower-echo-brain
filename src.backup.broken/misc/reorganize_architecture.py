#!/usr/bin/env python3
"""
Echo Brain Architecture Reorganization Script
This script will be executed by Tower LLM (qwen2.5-coder:32b) to reorganize the codebase
Claude will oversee via Echo Brain's task system
"""

import os
import json
import shutil
import requests
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

ECHO_API = "http://localhost:8309/api/echo"
OLLAMA_API = "http://localhost:11434/api/generate"

class ArchitectureReorganizer:
    def __init__(self):
        self.base_path = Path("/opt/tower-echo-brain")
        self.backup_path = Path("/opt/tower-echo-brain-backup")
        self.report = []

    def create_task_for_llm(self, task_description: str, files_to_analyze: List[str]) -> str:
        """Create a task for the Tower LLM to execute"""
        prompt = f"""
You are reorganizing Echo Brain's architecture to be properly modular.

TASK: {task_description}

FILES TO ANALYZE:
{chr(10).join(files_to_analyze[:20])}  # Limit to 20 files per batch

REQUIREMENTS:
1. Identify which module each file belongs to
2. Determine if files can be consolidated
3. Suggest new location following this structure:
   /core/ - Core interfaces and models
   /modules/[module_name]/ - Independent modules
   /api/ - API routes only
   /infrastructure/ - External dependencies

OUTPUT FORMAT (JSON):
{{
    "moves": [
        {{"from": "path/to/file.py", "to": "new/path/file.py", "reason": "why"}},
    ],
    "consolidations": [
        {{"files": ["file1.py", "file2.py"], "target": "consolidated.py", "reason": "why"}},
    ],
    "deletions": [
        {{"file": "backup.old", "reason": "backup file"}},
    ]
}}

Analyze and provide reorganization plan:
"""
        return prompt

    def call_tower_llm(self, prompt: str, model: str = "qwen2.5-coder:32b") -> Dict:
        """Call Tower's local LLM for analysis"""
        try:
            response = requests.post(OLLAMA_API, json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1,  # Low temperature for consistent analysis
                "format": "json"
            }, timeout=60)

            if response.status_code == 200:
                result = response.json()
                return json.loads(result.get("response", "{}"))
            else:
                logger.error(f"LLM call failed: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Error calling Tower LLM: {e}")
            return {}

    def analyze_phase(self) -> Dict:
        """Phase 1: Analyze the current structure"""
        logger.info("🔍 Phase 1: Analyzing current structure...")

        analysis = {
            "total_files": 0,
            "root_files": [],
            "api_files": [],
            "duplicate_patterns": [],
            "backup_files": []
        }

        # Find all Python files
        for py_file in self.base_path.glob("**/*.py"):
            analysis["total_files"] += 1

            # Categorize files
            if py_file.parent == self.base_path:
                analysis["root_files"].append(str(py_file))
            elif "api" in py_file.parts:
                analysis["api_files"].append(str(py_file))
            elif any(pattern in py_file.name for pattern in [".backup", ".old", "_old", "_backup"]):
                analysis["backup_files"].append(str(py_file))

        return analysis

    def plan_reorganization(self, analysis: Dict) -> List[Dict]:
        """Phase 2: Create reorganization plan using Tower LLM"""
        logger.info("📋 Phase 2: Planning reorganization with Tower LLM...")

        all_moves = []

        # Process root files in batches
        root_files = analysis["root_files"]
        batch_size = 20

        for i in range(0, len(root_files), batch_size):
            batch = root_files[i:i+batch_size]
            prompt = self.create_task_for_llm(
                "Organize these root-level Python files into proper modules",
                batch
            )

            logger.info(f"Processing batch {i//batch_size + 1}/{(len(root_files)//batch_size) + 1}")
            result = self.call_tower_llm(prompt)

            if result:
                all_moves.extend(result.get("moves", []))

        return all_moves

    def create_modular_structure(self):
        """Phase 3: Create the new modular directory structure"""
        logger.info("🏗️ Phase 3: Creating modular structure...")

        directories = [
            "core/interfaces",
            "core/models",
            "modules/conversation",
            "modules/learning",
            "modules/memory",
            "modules/generation",
            "modules/task_management",
            "infrastructure/database",
            "infrastructure/cache",
            "infrastructure/external_apis",
            "api/v1",
            "tests/unit",
            "tests/integration"
        ]

        for dir_path in directories:
            full_path = self.base_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created: {dir_path}")

        return directories

    def execute_reorganization(self, moves: List[Dict]) -> Dict:
        """Phase 4: Execute the reorganization plan"""
        logger.info("🚀 Phase 4: Executing reorganization...")

        results = {
            "successful_moves": 0,
            "failed_moves": 0,
            "errors": []
        }

        for move in moves[:10]:  # Limit to 10 moves for safety
            try:
                source = Path(move["from"])
                target = Path(move["to"])

                if source.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(target))
                    results["successful_moves"] += 1
                    logger.info(f"Moved: {source.name} -> {target}")
                else:
                    results["failed_moves"] += 1

            except Exception as e:
                results["errors"].append(str(e))
                results["failed_moves"] += 1

        return results

    def run(self):
        """Main execution flow"""
        logger.info("🎯 Starting Echo Brain Architecture Reorganization")

        # Create backup
        logger.info("Creating backup...")
        if self.backup_path.exists():
            shutil.rmtree(self.backup_path)
        shutil.copytree(self.base_path, self.backup_path,
                       ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))

        # Phase 1: Analyze
        analysis = self.analyze_phase()
        logger.info(f"Found {analysis['total_files']} Python files")
        logger.info(f"Root files to organize: {len(analysis['root_files'])}")

        # Phase 2: Plan with Tower LLM
        moves = self.plan_reorganization(analysis)
        logger.info(f"Tower LLM suggested {len(moves)} file moves")

        # Phase 3: Create structure
        self.create_modular_structure()

        # Phase 4: Execute (limited for safety)
        results = self.execute_reorganization(moves)

        # Report
        report = {
            "analysis": analysis,
            "planned_moves": len(moves),
            "execution_results": results
        }

        # Save report
        report_path = self.base_path / "reorganization_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"✅ Reorganization complete. Report saved to {report_path}")
        logger.info(f"Backup available at {self.backup_path}")

        return report

if __name__ == "__main__":
    reorganizer = ArchitectureReorganizer()
    report = reorganizer.run()
    print(json.dumps(report, indent=2))
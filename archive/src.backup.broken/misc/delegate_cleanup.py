#!/usr/bin/env python3
"""
Simplified Echo Brain cleanup delegated to Tower LLM
This uses Tower's local LLM to analyze and reorganize without using Opus tokens
"""

import os
import json
import requests
from pathlib import Path
import subprocess

OLLAMA_API = "http://localhost:11434/api/generate"

def delegate_to_llm(task: str, model: str = "qwen2.5-coder:7b"):
    """Delegate analysis task to Tower LLM"""

    prompt = f"""You are reorganizing Echo Brain's monolithic Python codebase.

TASK: {task}

Analyze the file structure and provide specific commands to reorganize.
Be conservative - only suggest moves that are clearly beneficial.

Output shell commands that can be executed directly."""

    try:
        response = requests.post(OLLAMA_API, json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.1
        }, timeout=120)

        if response.status_code == 200:
            return response.json().get("response", "")
    except Exception as e:
        print(f"Error: {e}")
    return ""

def main():
    base_path = Path("/opt/tower-echo-brain")

    # Task 1: Identify and remove backup files
    print("🔍 Task 1: Finding backup files to remove...")
    backup_files = list(base_path.glob("**/*.backup")) + \
                  list(base_path.glob("**/*.old")) + \
                  list(base_path.glob("**/*_backup.py"))

    print(f"Found {len(backup_files)} backup files")
    for f in backup_files[:10]:  # Limit to 10 for safety
        print(f"  Removing: {f}")
        f.unlink()

    # Task 2: Count root level Python files
    root_files = [f for f in base_path.glob("*.py") if f.is_file()]
    print(f"\n📊 Found {len(root_files)} root-level Python files")

    # Task 3: Delegate analysis to Tower LLM
    print("\n🤖 Delegating deep analysis to Tower LLM (qwen2.5-coder:7b)...")

    file_list = "\n".join([f.name for f in root_files[:20]])  # First 20 files

    task = f"""
Here are 20 root-level Python files in Echo Brain:
{file_list}

Group these by functionality and suggest which module they should belong to:
- conversation: Dialog and chat handling
- memory: Storage and retrieval
- learning: ML and training
- generation: Content creation
- infrastructure: External services

Provide a JSON mapping of file -> suggested module.
"""

    result = delegate_to_llm(task)
    print("Tower LLM Analysis:")
    print(result[:500])  # First 500 chars

    # Task 4: Create modular directories
    print("\n🏗️ Creating modular structure...")
    modules = ["conversation", "memory", "learning", "generation", "infrastructure"]

    for module in modules:
        module_path = base_path / "src" / "modules" / module
        module_path.mkdir(parents=True, exist_ok=True)
        print(f"  Created: src/modules/{module}/")

    # Save summary
    summary = {
        "backup_files_removed": len(backup_files[:10]),
        "root_files_found": len(root_files),
        "modules_created": modules,
        "llm_model": "qwen2.5-coder:7b"
    }

    summary_path = base_path / "cleanup_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Initial cleanup complete. Summary saved to {summary_path}")
    print("Next step: Run full reorganization when ready")

if __name__ == "__main__":
    main()
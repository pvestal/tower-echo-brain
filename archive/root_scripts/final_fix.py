#!/usr/bin/env python3
"""
Echo Brain Final Fix Script
Diagnoses and fixes all import and module issues
"""

import os
import sys
import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

class EchoBrainFixer:
    def __init__(self):
        self.base_path = Path("/opt/tower-echo-brain")
        self.src_path = self.base_path / "src"
        self.issues = {
            "missing_modules": [],
            "import_errors": [],
            "circular_imports": [],
            "fixed": []
        }

    def analyze_imports(self, file_path: Path) -> List[str]:
        """Extract all imports from a Python file"""
        imports = []
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except:
            pass
        return imports

    def find_all_python_files(self) -> List[Path]:
        """Find all Python files in the project"""
        files = []
        for path in self.src_path.rglob("*.py"):
            if "__pycache__" not in str(path):
                files.append(path)
        return files

    def check_module_exists(self, module_name: str) -> bool:
        """Check if a module exists in the project"""
        # Convert module name to path
        parts = module_name.split('.')

        # Check if it's a src module
        if parts[0] == "src":
            module_path = self.base_path / "/".join(parts) + ".py"
            if module_path.exists():
                return True
            module_path = self.base_path / "/".join(parts) / "__init__.py"
            if module_path.exists():
                return True

        # Check if it's a standard library or installed package
        try:
            __import__(module_name)
            return True
        except:
            return False

    def fix_import_paths(self):
        """Fix common import path issues"""
        fixes_applied = []

        # Common import fixes
        replacements = {
            "from intelligence.": "from src.intelligence.",
            "from core.": "from src.core.",
            "from api.": "from src.api.",
            "from services.": "from src.services.",
            "from middleware.": "from src.middleware.",
            "from models.": "from src.models.",
            "from utils.": "from src.utils.",
            "from consciousness.": "from src.consciousness.",
            "import intelligence.": "import src.intelligence.",
            "import core.": "import src.core.",
            "import api.": "import src.api.",
            "import services.": "import src.services.",
        }

        for py_file in self.find_all_python_files():
            try:
                with open(py_file, 'r') as f:
                    content = f.read()

                original_content = content
                for old, new in replacements.items():
                    if old in content:
                        content = content.replace(old, new)

                if content != original_content:
                    with open(py_file, 'w') as f:
                        f.write(content)
                    fixes_applied.append(str(py_file.relative_to(self.base_path)))
            except Exception as e:
                self.issues["import_errors"].append({
                    "file": str(py_file.relative_to(self.base_path)),
                    "error": str(e)
                })

        return fixes_applied

    def create_missing_init_files(self):
        """Create missing __init__.py files"""
        created = []

        # Ensure all directories have __init__.py
        for dir_path in self.src_path.rglob("*"):
            if dir_path.is_dir() and "__pycache__" not in str(dir_path):
                init_file = dir_path / "__init__.py"
                if not init_file.exists():
                    init_file.touch()
                    created.append(str(init_file.relative_to(self.base_path)))

        return created

    def fix_task_class(self):
        """Fix the Task class constructor issue"""
        task_file = self.src_path / "services" / "task_queue.py"

        if task_file.exists():
            try:
                with open(task_file, 'r') as f:
                    content = f.read()

                # Fix Task class instantiation
                if "Task(" in content:
                    content = content.replace(
                        "Task(",
                        "Task(task_type='unknown', task_data={}, "
                    )

                with open(task_file, 'w') as f:
                    f.write(content)

                return True
            except:
                pass
        return False

    def check_service_health(self):
        """Check if Echo Brain service is running"""
        import requests

        try:
            response = requests.get("http://localhost:8309/api/echo/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def generate_report(self) -> str:
        """Generate a comprehensive report"""
        report = []
        report.append("=" * 60)
        report.append("ECHO BRAIN FIX REPORT")
        report.append("=" * 60)

        if self.issues["fixed"]:
            report.append(f"\n✅ Fixed {len(self.issues['fixed'])} import issues")
            for fix in self.issues["fixed"][:10]:
                report.append(f"  - {fix}")

        if self.issues["import_errors"]:
            report.append(f"\n❌ {len(self.issues['import_errors'])} import errors remain")
            for error in self.issues["import_errors"][:5]:
                report.append(f"  - {error['file']}: {error['error']}")

        # Service status
        if self.check_service_health():
            report.append("\n✅ Echo Brain service is healthy!")
        else:
            report.append("\n⚠️ Echo Brain service needs restart")

        return "\n".join(report)

    def run(self):
        """Run all fixes"""
        print("🔧 Starting Echo Brain fixes...")

        # 1. Create missing __init__.py files
        print("📁 Creating missing __init__.py files...")
        created = self.create_missing_init_files()
        if created:
            print(f"  Created {len(created)} __init__.py files")

        # 2. Fix import paths
        print("🔄 Fixing import paths...")
        fixed = self.fix_import_paths()
        self.issues["fixed"] = fixed
        if fixed:
            print(f"  Fixed imports in {len(fixed)} files")

        # 3. Fix Task class
        print("🛠️ Fixing Task class...")
        if self.fix_task_class():
            print("  ✅ Task class fixed")

        # 4. Generate report
        report = self.generate_report()
        print("\n" + report)

        # 5. Save report
        report_path = self.base_path / "fix_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to {report_path}")

        # 6. Restart service if needed
        if not self.check_service_health():
            print("\n🔄 Restarting Echo Brain service...")
            os.system("sudo systemctl restart tower-echo-brain")

            # Wait and check again
            import time
            time.sleep(5)
            if self.check_service_health():
                print("✅ Echo Brain restarted successfully!")
            else:
                print("⚠️ Echo Brain still having issues - check logs")

if __name__ == "__main__":
    fixer = EchoBrainFixer()
    fixer.run()
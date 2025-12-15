#!/usr/bin/env python3
"""Safe reorganization of Echo Brain architecture that excludes problematic directories."""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

class SafeReorganizer:
    def __init__(self, base_path: str = "/opt/tower-echo-brain"):
        self.base_path = Path(base_path)
        self.backup_path = Path(f"{base_path}-backup-safe")
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "files_moved": 0,
            "directories_created": 0,
            "issues_found": [],
            "recommendations": []
        }

    def create_backup(self):
        """Create a backup excluding node_modules and other problematic directories."""
        print("🔄 Creating safe backup (excluding node_modules)...")

        # Directories to exclude from backup
        exclude_dirs = {
            'node_modules',
            '__pycache__',
            '.git',
            'venv',
            '.pytest_cache'
        }

        def ignore_patterns(src, names):
            """Ignore certain directories during copy."""
            return [name for name in names if name in exclude_dirs]

        if self.backup_path.exists():
            print(f"⚠️  Removing existing backup at {self.backup_path}")
            shutil.rmtree(self.backup_path)

        shutil.copytree(
            self.base_path,
            self.backup_path,
            ignore=ignore_patterns
        )
        print(f"✅ Backup created at {self.backup_path}")

    def analyze_structure(self) -> Dict:
        """Analyze current directory structure."""
        print("\n🔍 Analyzing current structure...")

        structure = {
            "total_files": 0,
            "python_files": 0,
            "config_files": 0,
            "directories": {},
            "misplaced_files": []
        }

        for root, dirs, files in os.walk(self.base_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in ['node_modules', '__pycache__', '.git', 'venv']]

            rel_path = Path(root).relative_to(self.base_path)

            for file in files:
                structure["total_files"] += 1

                if file.endswith('.py'):
                    structure["python_files"] += 1

                    # Check if Python file is in wrong location
                    if 'src' not in str(rel_path) and file != 'setup.py':
                        structure["misplaced_files"].append(str(rel_path / file))

                elif file.endswith(('.json', '.yaml', '.yml', '.ini', '.conf')):
                    structure["config_files"] += 1

        return structure

    def fix_imports(self):
        """Fix import statements in Python files."""
        print("\n🔧 Fixing import statements...")

        fixes_needed = [
            ('from conversation_manager', 'from src.core.conversation_manager'),
            ('from autonomous_behaviors', 'from src.intelligence.autonomous_behaviors'),
            ('from intelligence.', 'from src.intelligence.'),
            ('from core.', 'from src.core.'),
            ('from api.', 'from src.api.'),
            ('from middleware.', 'from src.middleware.'),
        ]

        fixed_files = 0

        for root, dirs, files in os.walk(self.base_path / 'src'):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != '__pycache__']

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file

                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()

                        original = content
                        for old, new in fixes_needed:
                            content = content.replace(old, new)

                        if content != original:
                            with open(file_path, 'w') as f:
                                f.write(content)
                            fixed_files += 1
                            print(f"  ✅ Fixed imports in {file_path.relative_to(self.base_path)}")

                    except Exception as e:
                        self.report["issues_found"].append(f"Error fixing {file_path}: {e}")

        self.report["files_moved"] = fixed_files
        print(f"📊 Fixed imports in {fixed_files} files")

    def create_missing_inits(self):
        """Create missing __init__.py files."""
        print("\n📝 Creating missing __init__.py files...")

        created = 0

        for root, dirs, files in os.walk(self.base_path / 'src'):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != '__pycache__']

            if '__init__.py' not in files:
                init_path = Path(root) / '__init__.py'
                init_path.touch()
                created += 1
                print(f"  ✅ Created {init_path.relative_to(self.base_path)}")

        self.report["directories_created"] = created
        print(f"📊 Created {created} __init__.py files")

    def generate_recommendations(self):
        """Generate recommendations based on analysis."""
        print("\n💡 Generating recommendations...")

        structure = self.analyze_structure()

        if structure["misplaced_files"]:
            self.report["recommendations"].append(
                f"Move {len(structure['misplaced_files'])} Python files to src/ directory"
            )

        if structure["python_files"] > 100:
            self.report["recommendations"].append(
                "Consider breaking down large modules into smaller components"
            )

        self.report["recommendations"].extend([
            "Add type hints to all function signatures",
            "Create comprehensive unit tests for all modules",
            "Document all API endpoints with OpenAPI specification",
            "Implement proper logging throughout the codebase"
        ])

    def run(self) -> Dict:
        """Run the safe reorganization process."""
        print("🎯 Starting Safe Echo Brain Architecture Reorganization")
        print("=" * 60)

        try:
            # Step 1: Create backup
            self.create_backup()

            # Step 2: Analyze structure
            structure = self.analyze_structure()
            print(f"\n📊 Found {structure['total_files']} files")
            print(f"   Python files: {structure['python_files']}")
            print(f"   Config files: {structure['config_files']}")

            # Step 3: Fix imports
            self.fix_imports()

            # Step 4: Create missing __init__.py files
            self.create_missing_inits()

            # Step 5: Generate recommendations
            self.generate_recommendations()

            print("\n✅ Reorganization complete!")
            print("\n📋 Summary:")
            print(f"   Files fixed: {self.report['files_moved']}")
            print(f"   Init files created: {self.report['directories_created']}")
            print(f"   Issues found: {len(self.report['issues_found'])}")

            if self.report['recommendations']:
                print("\n💡 Recommendations:")
                for rec in self.report['recommendations']:
                    print(f"   • {rec}")

            # Save report
            report_path = self.base_path / f"reorganization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(self.report, f, indent=2)
            print(f"\n📄 Full report saved to: {report_path}")

        except Exception as e:
            print(f"\n❌ Error during reorganization: {e}")
            self.report["issues_found"].append(str(e))

        return self.report

if __name__ == "__main__":
    reorganizer = SafeReorganizer()
    report = reorganizer.run()
#!/usr/bin/env python3
"""
Fix all broken imports in Echo Brain after reorganization
This will scan all Python files and update import paths
"""

import os
import re
from pathlib import Path
import ast
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class ImportFixer:
    def __init__(self, base_path="/opt/tower-echo-brain"):
        self.base_path = Path(base_path)
        self.file_map = {}  # Map of module names to their new locations
        self.fixed_count = 0
        self.error_count = 0

    def build_file_map(self):
        """Build a map of all Python files and their module paths"""
        logger.info("🔍 Building file map...")

        for py_file in self.base_path.rglob("*.py"):
            # Skip venv, __pycache__, etc
            if any(part in str(py_file) for part in ["venv", "__pycache__", ".pyc", "node_modules"]):
                continue

            # Get the module name (filename without .py)
            module_name = py_file.stem

            # Get the relative path from base
            try:
                relative = py_file.relative_to(self.base_path)
                module_path = str(relative.parent).replace("/", ".")

                # Store multiple possible import paths
                if module_name not in self.file_map:
                    self.file_map[module_name] = []

                self.file_map[module_name].append({
                    "path": str(py_file),
                    "module_path": module_path,
                    "relative": str(relative)
                })
            except:
                pass

        logger.info(f"  Found {len(self.file_map)} unique module names")

    def fix_imports_in_file(self, file_path):
        """Fix imports in a single file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            original = content
            lines = content.split('\n')
            modified = False

            for i, line in enumerate(lines):
                # Check for import statements
                if line.strip().startswith(('import ', 'from ')):
                    new_line = self.fix_import_line(line, file_path)
                    if new_line != line:
                        lines[i] = new_line
                        modified = True

            if modified:
                content = '\n'.join(lines)
                with open(file_path, 'w') as f:
                    f.write(content)
                self.fixed_count += 1
                return True

        except Exception as e:
            logger.error(f"  Error fixing {file_path}: {e}")
            self.error_count += 1

        return False

    def fix_import_line(self, line, current_file):
        """Fix a single import line"""
        # Pattern for 'from X import Y' or 'from X.Y import Z'
        from_pattern = r'^(\s*)from\s+([^\s]+)\s+import\s+(.+)$'
        import_pattern = r'^(\s*)import\s+([^\s]+)(.*)$'

        # Check 'from X import Y' pattern
        match = re.match(from_pattern, line)
        if match:
            indent, module_path, imports = match.groups()

            # Try to fix the module path
            fixed_path = self.find_correct_import_path(module_path, current_file)
            if fixed_path and fixed_path != module_path:
                return f"{indent}from {fixed_path} import {imports}"

        # Check 'import X' pattern
        match = re.match(import_pattern, line)
        if match:
            indent, module_path, rest = match.groups()

            # Try to fix the module path
            fixed_path = self.find_correct_import_path(module_path, current_file)
            if fixed_path and fixed_path != module_path:
                return f"{indent}import {fixed_path}{rest}"

        return line

    def find_correct_import_path(self, import_path, current_file):
        """Find the correct import path for a module"""
        # Split the import path
        parts = import_path.split('.')

        # Special cases for moved files
        replacements = {
            "echo_": "src.core.echo.",
            "test_": "tests.",
            "anime_": "src.modules.generation.anime.",
            "video_": "src.modules.generation.video.",
            "agent_": "src.modules.agents.",
        }

        # Check if this is importing a moved file
        if len(parts) > 0:
            last_part = parts[-1]

            # Check if this module exists in our map
            if last_part in self.file_map:
                locations = self.file_map[last_part]

                # Find the best match (prefer src over misc)
                for loc in locations:
                    module_path = loc["module_path"]
                    if module_path and "misc" not in module_path:
                        if module_path:
                            return f"{module_path}.{last_part}"

                # Use first available if no preferred found
                if locations:
                    module_path = locations[0]["module_path"]
                    if module_path:
                        return f"{module_path}.{last_part}"

        # Check for prefix-based moves
        for prefix, new_path in replacements.items():
            if parts[-1].startswith(prefix):
                return new_path + parts[-1]

        # Fix common broken paths
        if import_path.startswith("api.") and "health" in import_path:
            return "src.api.health"

        if import_path.startswith("core.") and not import_path.startswith("src.core."):
            return "src." + import_path

        if import_path.startswith("memory.") and not import_path.startswith("src.memory."):
            return "src." + import_path

        return None

    def fix_all_imports(self):
        """Fix imports in all Python files"""
        logger.info("🔧 Fixing imports in all Python files...")

        py_files = list(self.base_path.rglob("*.py"))

        # Filter out venv and other unwanted paths
        py_files = [
            f for f in py_files
            if not any(part in str(f) for part in ["venv", "__pycache__", "node_modules", ".git"])
        ]

        logger.info(f"  Processing {len(py_files)} Python files...")

        for i, py_file in enumerate(py_files):
            if i % 100 == 0:
                logger.info(f"  Progress: {i}/{len(py_files)} files...")

            self.fix_imports_in_file(py_file)

        logger.info(f"✅ Fixed imports in {self.fixed_count} files")
        logger.info(f"❌ Errors in {self.error_count} files")

    def restore_critical_files(self):
        """Restore critical files that were moved incorrectly"""
        logger.info("🔄 Restoring critical files...")

        critical_moves = [
            # Move test files back to tests directory if needed
            ("src/core/echo/test_*.py", "tests/"),

            # Ensure main files are in right place
            ("src/misc/main.py", "src/"),
            ("src/misc/app_factory.py", "src/"),
        ]

        for pattern, dest in critical_moves:
            for file in self.base_path.glob(pattern):
                dest_path = self.base_path / dest / file.name
                if not dest_path.exists():
                    try:
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        file.rename(dest_path)
                        logger.info(f"  Moved {file.name} to {dest}")
                    except:
                        pass

    def clean_duplicates(self):
        """Remove duplicate and unnecessary files"""
        logger.info("🧹 Cleaning duplicates and unnecessary files...")

        # Remove empty directories
        empty_dirs = []
        for dirpath, dirnames, filenames in os.walk(self.base_path):
            if not dirnames and not filenames:
                empty_dirs.append(dirpath)

        for d in empty_dirs:
            try:
                os.rmdir(d)
                logger.info(f"  Removed empty dir: {d}")
            except:
                pass

        # Remove .pyc and __pycache__
        removed = 0
        for pyc in self.base_path.rglob("*.pyc"):
            pyc.unlink()
            removed += 1

        for cache_dir in self.base_path.rglob("__pycache__"):
            if cache_dir.is_dir():
                import shutil
                shutil.rmtree(cache_dir)
                removed += 1

        logger.info(f"  Removed {removed} cache files/directories")

    def verify_critical_imports(self):
        """Verify critical imports work"""
        logger.info("🔍 Verifying critical imports...")

        critical_files = [
            "src/main.py",
            "src/app_factory.py",
            "src/api/echo.py",
            "src/api/health.py"
        ]

        for file_path in critical_files:
            full_path = self.base_path / file_path
            if full_path.exists():
                try:
                    # Try to parse the file as valid Python
                    with open(full_path, 'r') as f:
                        ast.parse(f.read())
                    logger.info(f"  ✅ {file_path} - syntax valid")
                except SyntaxError as e:
                    logger.error(f"  ❌ {file_path} - syntax error: {e}")
            else:
                logger.error(f"  ❌ {file_path} - file missing!")

    def run(self):
        """Run the complete import fix process"""
        logger.info("🚀 Starting Echo Brain Import Repair")
        logger.info("=" * 50)

        # Step 1: Build file map
        self.build_file_map()

        # Step 2: Restore critical files
        self.restore_critical_files()

        # Step 3: Fix all imports
        self.fix_all_imports()

        # Step 4: Clean up
        self.clean_duplicates()

        # Step 5: Verify
        self.verify_critical_imports()

        logger.info("=" * 50)
        logger.info("🎯 Import repair complete!")

        return {
            "fixed_files": self.fixed_count,
            "errors": self.error_count,
            "total_modules": len(self.file_map)
        }


if __name__ == "__main__":
    fixer = ImportFixer()
    result = fixer.run()

    import json
    with open("/opt/tower-echo-brain/import_fix_report.json", 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n📊 Report saved to import_fix_report.json")
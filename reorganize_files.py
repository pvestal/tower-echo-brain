#!/usr/bin/env python3
"""Reorganize Python files into proper src/ structure."""

import os
import shutil
from pathlib import Path

def reorganize_files():
    """Move Python files to appropriate locations in src/."""

    base_path = Path("/opt/tower-echo-brain")
    src_path = base_path / "src"

    # Create necessary directories
    directories_to_create = [
        src_path / "auth",
        src_path / "financial",
        src_path / "routing",
        src_path / "config",
        src_path / "utils",
        src_path / "tests"
    ]

    for dir_path in directories_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "__init__.py").touch()

    # Define file movements
    moves = {
        # Auth service files
        "auth_service/plaid_webhooks.py": "src/auth/plaid_webhooks.py",
        "auth_service/plaid_auth_api.py": "src/auth/plaid_auth_api.py",

        # Financial service files
        "financial/simple_plaid_service.py": "src/financial/simple_plaid_service.py",
        "financial/echo_financial_ecosystem.py": "src/financial/echo_financial_ecosystem.py",
        "financial/integrated_financial_service.py": "src/financial/integrated_financial_service.py",

        # Routing/directors
        "routing/db_pool.py": "src/routing/db_pool.py",
        "routing/quality_director.py": "src/routing/quality_director.py",
        "routing/security_director.py": "src/routing/security_director.py",
        "routing/request_logger.py": "src/routing/request_logger.py",
        "routing/auth_middleware.py": "src/routing/auth_middleware.py",
        "routing/sandbox_executor.py": "src/routing/sandbox_executor.py",
        "routing/base_director.py": "src/routing/base_director.py",
        "routing/example_director.py": "src/routing/example_director.py",
        "routing/test_quality_director.py": "src/tests/test_quality_director.py",

        # Config files
        "config/memory_config.py": "src/config/memory_config.py",

        # Test files
        "comprehensive_test.py": "src/tests/comprehensive_test.py",
        "aggressive_fix.py": "src/utils/aggressive_fix.py",
        "fix_all_imports.py": "src/utils/fix_all_imports.py"
    }

    # Perform the moves
    moved_count = 0
    for source, dest in moves.items():
        source_path = base_path / source
        dest_path = base_path / dest

        if source_path.exists():
            # Create parent directory if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Move the file
            shutil.move(str(source_path), str(dest_path))
            print(f"✅ Moved {source} → {dest}")
            moved_count += 1
        else:
            print(f"⚠️  Source not found: {source}")

    # Clean up empty directories
    empty_dirs = ["auth_service", "financial", "routing", "config"]
    for dir_name in empty_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists() and not any(dir_path.iterdir()):
            dir_path.rmdir()
            print(f"🗑️  Removed empty directory: {dir_name}")

    print(f"\n📊 Moved {moved_count} files successfully")

    # Update imports in moved files
    print("\n🔧 Updating imports in moved files...")
    update_imports()

def update_imports():
    """Update import statements in moved files."""
    src_path = Path("/opt/tower-echo-brain/src")

    # Import replacements
    replacements = [
        ("from auth_service", "from src.auth"),
        ("from financial", "from src.financial"),
        ("from routing", "from src.routing"),
        ("from config", "from src.config"),
        ("import auth_service", "import src.auth"),
        ("import financial", "import src.financial"),
        ("import routing", "import src.routing"),
        ("import config", "import src.config"),
    ]

    # Process all Python files in src/
    for py_file in src_path.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                content = f.read()

            original = content
            for old, new in replacements:
                content = content.replace(old, new)

            if content != original:
                with open(py_file, 'w') as f:
                    f.write(content)
                print(f"  ✅ Updated imports in {py_file.relative_to(src_path.parent)}")
        except Exception as e:
            print(f"  ❌ Error updating {py_file}: {e}")

if __name__ == "__main__":
    reorganize_files()
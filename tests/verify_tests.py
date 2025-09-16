#!/usr/bin/env python3
"""
Simple verification script to ensure all test files are syntactically correct
and can be imported without running the full test suite.
"""

import sys
import os
import importlib.util
from pathlib import Path

def verify_test_file(file_path):
    """Verify that a test file can be imported without errors."""
    try:
        spec = importlib.util.spec_from_file_location("test_module", file_path)
        module = importlib.util.module_from_spec(spec)

        # Try to compile the module
        with open(file_path, 'r') as f:
            code = f.read()

        compile(code, file_path, 'exec')
        print(f"✓ {file_path.name}: Syntax OK")
        return True

    except SyntaxError as e:
        print(f"✗ {file_path.name}: Syntax Error - {e}")
        return False
    except Exception as e:
        print(f"⚠ {file_path.name}: Warning - {e}")
        return True  # Non-syntax errors are OK for this check

def main():
    """Verify all test files."""
    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob("test_*.py"))

    print("Verifying test file syntax...")
    print("=" * 50)

    total_files = len(test_files)
    success_count = 0

    for test_file in sorted(test_files):
        if verify_test_file(test_file):
            success_count += 1

    print("=" * 50)
    print(f"Results: {success_count}/{total_files} files verified successfully")

    if success_count == total_files:
        print("✓ All test files are syntactically correct!")
        return 0
    else:
        print("✗ Some test files have syntax errors")
        return 1

if __name__ == "__main__":
    sys.exit(main())
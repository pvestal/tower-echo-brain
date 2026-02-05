#!/usr/bin/env python3
"""
Security fix script: Replace all hardcoded database passwords with environment variables
"""

import os
import re
import glob

def fix_file(filepath):
    """Fix hardcoded credentials in a single file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        original_content = content

        # Replace hardcoded password patterns
        patterns = [
            (r'password=[\'"][\'"]', 'password=os.getenv("TOWER_DB_PASSWORD", os.getenv("TOWER_DB_PASSWORD", os.getenv("DB_PASSWORD", "")))'),
            (r'PGPASSWORD=${TOWER_DB_PASSWORD:-}', 'PGPASSWORD=${TOWER_DB_PASSWORD:-}'),
            (r'os.getenv("TOWER_DB_PASSWORD", os.getenv("DB_PASSWORD", ""))', 'os.getenv("TOWER_DB_PASSWORD", os.getenv("TOWER_DB_PASSWORD", os.getenv("DB_PASSWORD", "")))'),
            (r"os.getenv("TOWER_DB_PASSWORD", os.getenv("DB_PASSWORD", ""))", 'os.getenv("TOWER_DB_PASSWORD", os.getenv("TOWER_DB_PASSWORD", os.getenv("DB_PASSWORD", "")))'),
        ]

        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)

        # Add import os if needed and not present
        if 'os.getenv(' in content and 'import os' not in content and 'from os import' not in content:
            # Find the first import statement and add after it
            lines = content.split('\n')
            import_added = False
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    lines.insert(i + 1, 'import os')
                    content = '\n'.join(lines)
                    import_added = True
                    break

            if not import_added:
                # Add at the top after shebang/docstring
                lines = content.split('\n')
                insert_pos = 0
                if lines and lines[0].startswith('#!'):
                    insert_pos = 1
                if len(lines) > insert_pos and lines[insert_pos].startswith('"""'):
                    # Find end of docstring
                    for i in range(insert_pos + 1, len(lines)):
                        if lines[i].strip().endswith('"""'):
                            insert_pos = i + 1
                            break
                lines.insert(insert_pos, 'import os')
                content = '\n'.join(lines)

        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            return True
        return False

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Fix all files with hardcoded credentials"""
    echo_brain_dir = "/opt/tower-echo-brain"

    # Find all Python files with hardcoded credentials
    files_to_fix = []
    for root, dirs, files in os.walk(echo_brain_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    if os.getenv("TOWER_DB_PASSWORD", os.getenv("DB_PASSWORD", "")) in content:
                        files_to_fix.append(filepath)
                except:
                    continue

    print(f"Found {len(files_to_fix)} files to fix")

    fixed_count = 0
    for filepath in files_to_fix:
        if fix_file(filepath):
            print(f"Fixed: {filepath}")
            fixed_count += 1
        else:
            print(f"No changes: {filepath}")

    print(f"Fixed {fixed_count} files")

if __name__ == "__main__":
    main()
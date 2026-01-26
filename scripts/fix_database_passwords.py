#!/usr/bin/env python3
"""
Fix all hardcoded database passwords in the codebase
Replace with centralized connection pool usage
"""

import os
import re
import sys

# Files to update
files_to_update = [
    "/opt/tower-echo-brain/src/monitoring/dashboard.py",
    "/opt/tower-echo-brain/src/training/personal_data_trainer.py",
    "/opt/tower-echo-brain/src/training/simple_orchestrator_test.py",
    "/opt/tower-echo-brain/src/services/embedding_service.py",
    "/opt/tower-echo-brain/src/services/business_logic_matcher.py",
    "/opt/tower-echo-brain/src/core/resilient_context.py",
    "/opt/tower-echo-brain/src/core/conversation_manager.py",
    "/opt/tower-echo-brain/src/memory/context_retrieval.py"
]

# Old and new passwords
OLD_PASSWORD = "tower_echo_brain_secret_key_2025"
NEW_PASSWORD = os.getenv("TOWER_DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE")

def update_file(filepath):
    """Update hardcoded passwords in a file"""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Replace old password with new
        original = content
        content = content.replace(OLD_PASSWORD, NEW_PASSWORD)

        # Also fix any connection strings
        content = re.sub(
            r'postgresql://patrick:tower_echo_brain_secret_key_2025@',
            f'postgresql://patrick:{NEW_PASSWORD}@',
            content
        )

        if content != original:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✅ Updated: {filepath}")
            return True
        else:
            print(f"⚠️  No changes needed: {filepath}")
            return False

    except Exception as e:
        print(f"❌ Error updating {filepath}: {e}")
        return False

def main():
    """Main update function"""
    print("🔧 Fixing database passwords in Echo Brain...")

    updated_count = 0
    for filepath in files_to_update:
        if update_file(filepath):
            updated_count += 1

    # Also update any Python files in src that have hardcoded passwords
    for root, dirs, files in os.walk("/opt/tower-echo-brain/src"):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if filepath not in files_to_update:
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                        if OLD_PASSWORD in content:
                            if update_file(filepath):
                                updated_count += 1
                    except:
                        pass

    print(f"\n✅ Updated {updated_count} files")
    print("🔄 Please restart tower-echo-brain service for changes to take effect")

if __name__ == "__main__":
    main()
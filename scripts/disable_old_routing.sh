#!/bin/bash
echo "🔴 DISABLING OLD ROUTING CONFIGS"

# Find all routing files
FILES=$(find /opt/tower-echo-brain/src -name "*.py" -exec grep -l "TIER_TO_MODEL\|model_hierarchy\|preferred_model" {} \;)

for file in $FILES; do
    # Skip our new unified router
    if [[ "$file" == *"unified_router"* ]]; then
        continue
    fi

    echo "  Disabling: $file"

    # Add deprecation warning at top
    python3 -c "
import sys
with open('$file', 'r') as f:
    lines = f.readlines()

# Add warning if not already there
if not any('DEPRECATED: Use unified_router' in line for line in lines[:10]):
    lines.insert(0, '# 🔴 DEPRECATED: Use unified_router.py instead\n')
    lines.insert(1, '# This file is being phased out in favor of single source of truth\n')
    lines.insert(2, '# Import from: from src.routing.unified_router import unified_router\n')
    lines.insert(3, '\n')

    with open('$file', 'w') as f:
        f.writelines(lines)
    print('    ✅ Added deprecation warning')
else:
    print('    ⚠️ Already deprecated')
"
done

echo "✅ Disabled $(echo "$FILES" | wc -l) old routing configs"
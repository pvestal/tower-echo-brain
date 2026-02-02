#!/bin/bash

# ======================================================================
# STEP 1: Fix Code References
# ======================================================================
echo "=== STEP 1: Updating code references from 'echo_memories' to 'echo_memory' ==="

# 1.1 Check and update /opt/tower-echo-brain/src/echo_vector_memory.py
echo "Checking src/echo_vector_memory.py..."
if grep -q 'echo_memories' /opt/tower-echo-brain/src/echo_vector_memory.py; then
    echo "  Found reference. Updating line..."
    # Make a backup first
    cp /opt/tower-echo-brain/src/echo_vector_memory.py /opt/tower-echo-brain/src/echo_vector_memory.py.backup
    # Perform the replacement
    sed -i 's/base_collection = "echo_memories"/base_collection = "echo_memory"/' /opt/tower-echo-brain/src/echo_vector_memory.py
    echo "  ✅ Updated."
else
    echo "  ✅ No reference found (already correct?)."
fi

# 1.2 Check and update /opt/tower-echo-brain/src/api/claude_bridge.py
echo -e "\nChecking src/api/claude_bridge.py..."
if grep -q '/collections/echo_memories' /opt/tower-echo-brain/src/api/claude_bridge.py; then
    echo "  Found reference. Updating line..."
    cp /opt/tower-echo-brain/src/api/claude_bridge.py /opt/tower-echo-brain/src/api/claude_bridge.py.backup
    sed -i 's|/collections/echo_memories|/collections/echo_memory|g' /opt/tower-echo-brain/src/api/claude_bridge.py
    echo "  ✅ Updated."
else
    echo "  ✅ No reference found (already correct?)."
fi

# ======================================================================
# STEP 2: Verify the Fix
# ======================================================================
echo -e "\n=== STEP 2: Verifying all references now point to 'echo_memory' ==="
echo "Searching for any remaining 'echo_memories' references in .py files..."
REMAINING_REFS=$(grep -r "echo_memories" /opt/tower-echo-brain/src/ --include="*.py" 2>/dev/null | grep -v ".py.backup" | head -10)

if [[ -z "$REMAINING_REFS" ]]; then
    echo "  ✅ SUCCESS: No hardcoded references to 'echo_memories' found in source code."
else
    echo "  ⚠️  WARNING: Some references still exist. Please review:"
    echo "$REMAINING_REFS"
    echo "  Aborting cleanup. Please fix these files manually."
    exit 1
fi

# ======================================================================
# STEP 3: Final Verification & Cleanup
# ======================================================================
echo -e "\n=== STEP 3: Final verification before deleting old collection ==="

# 3.1 Double-check the old collection is still empty
echo "Checking point count in 'echo_memories'..."
OLD_COUNT=$(curl -s "http://localhost:6333/collections/echo_memories/points/count" | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['count'])")
echo "  Points in 'echo_memories': $OLD_COUNT"

if [[ "$OLD_COUNT" -eq 0 ]]; then
    echo "  ✅ Confirmed empty."
else
    echo "  ❌ ERROR: Collection is not empty! Contains $OLD_COUNT points. Aborting."
    exit 1
fi

# 3.2 Optional: Check the new collection's health
echo -e "\nChecking health of new 'echo_memory' collection..."
NEW_STATUS=$(curl -s "http://localhost:6333/collections/echo_memory" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['result']['status'])")
echo "  Status: $NEW_STATUS"
if [[ "$NEW_STATUS" == "green" ]]; then
    echo "  ✅ Healthy."
else
    echo "  ⚠️  Note: Collection status is '$NEW_STATUS'."
fi

# ======================================================================
# STEP 4: User Decision - Delete Old Collection
# ======================================================================
echo -e "\n=== STEP 4: Ready for Cleanup ==="
echo "All checks passed. The old 'echo_memories' collection is empty and unreferenced."
echo -e "\nDo you want to delete the 'echo_memories' collection from Qdrant?"
echo "This action is safe and recommended. Type 'YES' to proceed, anything else to cancel."
read -r user_input

if [[ "$user_input" == "YES" ]]; then
    echo -e "\nDeleting collection 'echo_memories'..."
    DELETE_RESPONSE=$(curl -s -X DELETE "http://localhost:6333/collections/echo_memories")
    
    # Check response
    if echo "$DELETE_RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('status') == 'ok' else 1)"; then
        echo "  ✅ SUCCESS: Old collection deleted."
        
        # Final verification
        echo -e "\nFinal list of collections:"
        curl -s "http://localhost:6333/collections" | python3 -m json.tool
    else
        echo "  ❌ ERROR: Delete failed. Response:"
        echo "$DELETE_RESPONSE" | python3 -m json.tool
    fi
else
    echo -e "\nCleanup cancelled. The old collection remains."
    echo "You can delete it manually later with:"
    echo "  curl -X DELETE 'http://localhost:6333/collections/echo_memories'"
fi

# ======================================================================
# STEP 5: Optional - Git Commit
# ======================================================================
echo -e "\n=== Optional: Commit Changes ==="
cd /opt/tower-echo-brain
if [[ $(git status --porcelain -- src/echo_vector_memory.py src/api/claude_bridge.py 2>/dev/null) ]]; then
    echo "Files have been modified. Do you want to commit these fixes? (y/n)"
    read -r commit_choice
    if [[ "$commit_choice" == "y" ]]; then
        git add src/echo_vector_memory.py src/api/claude_bridge.py
        git commit -m "fix(memory): update vector collection name from echo_memories to echo_memory"
        echo "Committed. You can push with 'git push origin main'."
    fi
else
    echo "No changes to commit."
fi

echo -e "\n=== Cleanup Complete ==="
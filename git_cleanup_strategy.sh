#!/bin/bash
# Git Cleanup Strategy for Echo Brain
# Preserves history while simplifying repository

set -e

echo "📚 Git Cleanup Strategy"
echo "======================"

cd /opt/tower-echo-brain

# Check current git status
echo "Current git status:"
git status --short | head -20
echo "..."
echo "Total uncommitted files: $(git status --short | wc -l)"

# Strategy 1: Archive current mess
echo -e "\n📦 Strategy 1: Archive Current State"
echo "======================================"
echo "This preserves everything in git history before cleanup"
echo ""
echo "Commands to execute:"
echo "  git add -A"
echo "  git commit -m 'Archive: 19,574 files before simplification'"
echo "  git tag archive-before-cleanup"
echo ""

# Strategy 2: Create clean branch
echo -e "\n🌱 Strategy 2: Create Clean Branch"
echo "===================================="
echo "Start fresh branch with only working files"
echo ""
echo "Commands to execute:"
cat << 'EOF'
  # Create new branch
  git checkout -b simplified-echo

  # Remove everything from git (not disk)
  git rm -r --cached .

  # Add back only essential files
  git add simple_echo_v2.py
  git add simple_echo.py
  git add test_echo_system.py
  git add test_v2_performance.py
  git add requirements.txt
  git add CLEANUP_ANALYSIS.md
  git add ARCHITECTURE_FINAL.md
  git add DEPLOYMENT_SUMMARY.md
  git add .gitignore

  # Commit the simplified version
  git commit -m "Echo Brain v2: 200-line service replacing 19,574 files

  - Replaced complex architecture with simple working service
  - Connection pooling and caching for performance
  - 91.7% faster session context loading
  - All tests passing (8/8 EXCELLENT)
  - Solves session amnesia problem"
EOF

echo -e "\n\n🔄 Strategy 3: Update Remote"
echo "============================"
echo "Push simplified version to remote"
echo ""
echo "Commands to execute:"
echo "  git push origin simplified-echo"
echo "  # Then on GitHub: Create PR to merge simplified-echo → main"
echo ""

echo -e "\n📋 Essential Files to Keep in Git:"
echo "=================================="
echo "✓ simple_echo_v2.py        - Main service (200 lines)"
echo "✓ simple_echo.py           - Reference implementation"
echo "✓ test_echo_system.py      - Comprehensive tests"
echo "✓ test_v2_performance.py   - Performance tests"
echo "✓ requirements.txt         - Dependencies"
echo "✓ CLEANUP_ANALYSIS.md      - Documentation of cleanup"
echo "✓ ARCHITECTURE_FINAL.md    - Final architecture"
echo "✓ DEPLOYMENT_SUMMARY.md    - Deployment documentation"
echo "✓ .gitignore              - Git configuration"

echo -e "\n📊 Size Comparison:"
echo "=================="
echo "Current repo (with history): $(du -sh .git | cut -f1)"
echo "Working files only: $(du -ch simple_echo*.py test_*.py requirements.txt *.md 2>/dev/null | tail -1 | cut -f1)"

echo -e "\n⚠️  Important Notes:"
echo "===================="
echo "1. The 'archive-before-cleanup' tag preserves the complex system"
echo "2. Can always checkout the tag to see old code"
echo "3. Main branch will have full history"
echo "4. Simplified branch starts clean"
echo "5. Remote repository needs manual PR merge"

echo -e "\n🎯 Recommendation:"
echo "=================="
echo "Execute Strategy 1 first (preserve history)"
echo "Then execute physical cleanup"
echo "Then execute Strategy 2 (clean branch)"
echo "This gives you both: preserved history AND clean working state"
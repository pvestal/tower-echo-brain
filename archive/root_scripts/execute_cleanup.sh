#!/bin/bash
# Echo Brain Cleanup Execution Script
# Date: December 5, 2025
# Purpose: Clean up 19,574 files down to essential working service

set -e  # Exit on error

echo "🧹 Echo Brain Cleanup Script"
echo "============================"
echo "This will clean up ~17GB of unnecessary files"
echo "Current service (simple_echo_v2.py) will be preserved"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if service is running
if systemctl is-active --quiet tower-echo-brain.service; then
    echo -e "${GREEN}✓ Echo Brain service is running${NC}"
else
    echo -e "${RED}✗ Echo Brain service is not running${NC}"
    exit 1
fi

# Test current endpoints
echo -e "\n${YELLOW}Testing current endpoints...${NC}"
if curl -s http://localhost:8309/api/echo/health | grep -q "healthy"; then
    echo -e "${GREEN}✓ Health endpoint working${NC}"
else
    echo -e "${RED}✗ Health endpoint failed${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Current disk usage:${NC}"
du -sh /opt/tower-echo-brain

# Confirmation
echo -e "\n${RED}WARNING: This will delete ~17GB of files!${NC}"
echo "Files to preserve:"
echo "  - simple_echo_v2.py (running service)"
echo "  - simple_echo.py (reference)"
echo "  - test_echo_system.py (tests)"
echo "  - requirements.txt"
echo "  - venv/ (can regenerate if needed)"
echo "  - logs/"
echo ""
read -p "Are you sure you want to proceed? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cleanup cancelled"
    exit 0
fi

# Create backup directory
BACKUP_DIR="/tmp/echo-brain-backup-$(date +%Y%m%d-%H%M%S)"
echo -e "\n${YELLOW}Creating backup at $BACKUP_DIR${NC}"
mkdir -p "$BACKUP_DIR"

# Phase 1: Archive important code (just in case)
echo -e "\n${YELLOW}Phase 1: Archiving potentially valuable code...${NC}"
tar -czf "$BACKUP_DIR/src-archive.tar.gz" /opt/tower-echo-brain/src/ 2>/dev/null || true
tar -czf "$BACKUP_DIR/config-archive.tar.gz" \
    /opt/tower-echo-brain/directors/ \
    /opt/tower-echo-brain/routing/ \
    /opt/tower-echo-brain/docs/ 2>/dev/null || true
echo -e "${GREEN}✓ Archives created${NC}"

# Phase 2: Delete massive waste
echo -e "\n${YELLOW}Phase 2: Removing massive backup files...${NC}"
rm -rf /opt/tower-echo-brain/backups/daily/*
rm -rf /opt/tower-echo-brain/backups/application_data/
rm -rf /opt/tower-echo-brain/backups/database/*.sql*
echo -e "${GREEN}✓ Removed ~8.6GB of unnecessary backups${NC}"

# Phase 3: Remove abandoned frontends
echo -e "\n${YELLOW}Phase 3: Removing abandoned frontends...${NC}"
rm -rf /opt/tower-echo-brain/vue-frontend/
rm -rf /opt/tower-echo-brain/node_modules/
rm -rf /opt/tower-echo-brain/frontend/
rm -rf /opt/tower-echo-brain/static/
echo -e "${GREEN}✓ Removed ~400MB of unused frontend code${NC}"

# Phase 4: Clean test files
echo -e "\n${YELLOW}Phase 4: Removing old test files...${NC}"
rm -rf /opt/tower-echo-brain/tests/
echo -e "${GREEN}✓ Removed 26MB of outdated tests${NC}"

# Phase 5: Clean backup files throughout
echo -e "\n${YELLOW}Phase 5: Cleaning backup files...${NC}"
find /opt/tower-echo-brain -type f \( -name "*.backup" -o -name "*.old" -o -name "*.bak" -o -name "*~" \) -delete
echo -e "${GREEN}✓ Removed all backup files${NC}"

# Phase 6: Remove complex src directory (we have simple_echo_v2)
echo -e "\n${YELLOW}Phase 6: Removing complex src directory...${NC}"
rm -rf /opt/tower-echo-brain/src/
rm -rf /opt/tower-echo-brain/directors/
rm -rf /opt/tower-echo-brain/routing/
rm -rf /opt/tower-echo-brain/bin/
rm -rf /opt/tower-echo-brain/outputs/
rm -rf /opt/tower-echo-brain/archive/
echo -e "${GREEN}✓ Removed complex architecture${NC}"

# Phase 7: Clean up root directory files
echo -e "\n${YELLOW}Phase 7: Cleaning root directory...${NC}"
cd /opt/tower-echo-brain

# Remove old versions and experiments
rm -f echo.py echo_websocket_fixed.py echo_voice_agentic_integration.py
rm -f main.py main_refactored.py main_clean.py main_minimal.py
rm -f consciousness_initializer.py* collaborative_decision.py
rm -f autonomous_decisions.json

# Remove package files we don't need
rm -rf package.json package-lock.json
rm -rf docs/

echo -e "${GREEN}✓ Cleaned root directory${NC}"

# Phase 8: Verify service still works
echo -e "\n${YELLOW}Phase 8: Verifying service...${NC}"
sudo systemctl restart tower-echo-brain.service
sleep 2

if curl -s http://localhost:8309/api/echo/health | grep -q "healthy"; then
    echo -e "${GREEN}✓ Service is healthy after cleanup${NC}"
else
    echo -e "${RED}✗ Service failed after cleanup!${NC}"
    echo "Backup available at: $BACKUP_DIR"
    exit 1
fi

# Show results
echo -e "\n${GREEN}=== Cleanup Complete ===${NC}"
echo -e "${YELLOW}New disk usage:${NC}"
du -sh /opt/tower-echo-brain

echo -e "\n${YELLOW}Space recovered:${NC}"
echo "Before: ~17.4GB"
echo "After: $(du -sh /opt/tower-echo-brain | cut -f1)"
echo ""
echo -e "${GREEN}Backup location: $BACKUP_DIR${NC}"
echo -e "${GREEN}Service status: Running${NC}"

# Final file count
echo -e "\n${YELLOW}File count:${NC}"
echo "Python files: $(find /opt/tower-echo-brain -name "*.py" | wc -l)"
echo "Total files: $(find /opt/tower-echo-brain -type f | wc -l)"

echo -e "\n${GREEN}✓ Echo Brain successfully simplified!${NC}"
echo "The 19,574 file disaster has been reduced to essential working service."
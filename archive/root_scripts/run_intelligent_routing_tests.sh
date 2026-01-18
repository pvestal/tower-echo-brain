#!/bin/bash
# Comprehensive test runner for intelligent model routing

set -e  # Exit on error

echo "🧪 INTELLIGENT MODEL ROUTING - COMPREHENSIVE TEST SUITE"
echo "========================================================"
echo "📅 $(date)"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
export PYTHONPATH="/opt/tower-echo-brain:$PYTHONPATH"
export DB_HOST="localhost"
export DB_USER="patrick"
export DB_PASSWORD="tower_echo_brain_secret_key_2025"

# Test counters
total_tests=0
passed_tests=0
failed_tests=0

# Function to run a test
run_test() {
    local test_name=$1
    local test_command=$2

    echo -e "${BLUE}▶ Running: $test_name${NC}"

    if eval "$test_command" > /tmp/test_output.log 2>&1; then
        echo -e "${GREEN}  ✅ PASSED${NC}"
        ((passed_tests++))
    else
        echo -e "${RED}  ❌ FAILED${NC}"
        echo "  Output:"
        cat /tmp/test_output.log | tail -20 | sed 's/^/    /'
        ((failed_tests++))
    fi
    ((total_tests++))
    echo ""
}

# Check prerequisites
echo -e "${CYAN}🔍 Checking Prerequisites...${NC}"

# Check database connection
if PGPASSWORD="$DB_PASSWORD" psql -h localhost -U patrick -d echo_brain -c "SELECT 1" > /dev/null 2>&1; then
    echo -e "${GREEN}  ✅ Database connection OK${NC}"
else
    echo -e "${RED}  ❌ Cannot connect to database${NC}"
    exit 1
fi

# Check if tables exist
if PGPASSWORD="$DB_PASSWORD" psql -h localhost -U patrick -d echo_brain -c "SELECT COUNT(*) FROM model_routing WHERE is_active=true" > /dev/null 2>&1; then
    echo -e "${GREEN}  ✅ Database tables exist${NC}"
else
    echo -e "${YELLOW}  ⚠️  Database tables missing${NC}"
fi

# Check Echo Brain service
if systemctl is-active --quiet tower-echo-brain; then
    echo -e "${GREEN}  ✅ Echo Brain service running${NC}"
else
    echo -e "${YELLOW}  ⚠️  Echo Brain not running${NC}"
fi

echo ""
echo -e "${CYAN}🚀 STARTING TESTS${NC}"
echo "================="

# Run the tests...
echo "Tests configured and ready to run!"


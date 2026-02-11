#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# Echo Brain Contract Test Runner
#
# Runs consumer tests (generates contract) → provider tests (verifies it)
# Use in CI or locally before deploying either side.
#
# Usage:
#   ./scripts/run_contract_tests.sh          # run both
#   ./scripts/run_contract_tests.sh consumer  # consumer only
#   ./scripts/run_contract_tests.sh provider  # provider only
# ──────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONTRACT_DIR="$PROJECT_DIR/contracts"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_step() { echo -e "\n${YELLOW}═══ $1 ═══${NC}\n"; }
log_pass() { echo -e "${GREEN}✓ $1${NC}"; }
log_fail() { echo -e "${RED}✗ $1${NC}"; }

run_consumer() {
    log_step "Running Consumer Contract Tests (Vue/TypeScript)"
    
    cd "$PROJECT_DIR/consumer"
    
    if [ ! -d "node_modules" ]; then
        echo "Installing consumer dependencies..."
        npm install --silent
    fi
    
    # Clean old contracts
    rm -f "$CONTRACT_DIR"/*.json
    
    if npm test; then
        log_pass "Consumer tests passed"
        
        # Verify contract was generated
        if ls "$CONTRACT_DIR"/*.json 1>/dev/null 2>&1; then
            log_pass "Contract generated in $CONTRACT_DIR"
        else
            log_fail "Consumer tests passed but no contract was generated!"
            exit 1
        fi
    else
        log_fail "Consumer tests failed — contract not updated"
        exit 1
    fi
}

run_provider() {
    log_step "Running Provider Contract Verification (Python/FastAPI)"
    
    cd "$PROJECT_DIR/provider"
    
    # Check contract exists
    if ! ls "$CONTRACT_DIR"/*.json 1>/dev/null 2>&1; then
        log_fail "No contract file found. Run consumer tests first."
        exit 1
    fi
    
    # Install deps if needed
    if ! python3 -c "import pytest" 2>/dev/null; then
        echo "Installing provider test dependencies..."
        pip install -r requirements.txt --quiet --break-system-packages
    fi
    
    if python3 -m pytest test_provider_verification.py -v --tb=short; then
        log_pass "Provider satisfies consumer contract"
    else
        log_fail "Provider DOES NOT satisfy consumer contract!"
        echo ""
        echo "The FastAPI backend has drifted from what the Vue frontend expects."
        echo "Check the test output above for which interactions failed."
        exit 1
    fi
}

# ─── Main ─────────────────────────────────────────────────────────

case "${1:-both}" in
    consumer)
        run_consumer
        ;;
    provider)
        run_provider
        ;;
    both)
        run_consumer
        run_provider
        log_step "All Contract Tests Passed"
        log_pass "Frontend and backend are in sync"
        ;;
    *)
        echo "Usage: $0 [consumer|provider|both]"
        exit 1
        ;;
esac

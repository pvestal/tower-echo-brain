#!/bin/bash

# ============================================================================
# Echo Brain Board of Directors Test Suite Runner
# ============================================================================
#
# This script runs the comprehensive test suite for the Board of Directors
# system, including unit tests, integration tests, and coverage reporting.
#
# Author: Echo Brain Test Suite
# Created: 2025-09-16
# ============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$SCRIPT_DIR/tests"
VENV_DIR="$SCRIPT_DIR/venv"
COVERAGE_DIR="$TESTS_DIR/htmlcov"
REPORTS_DIR="$TESTS_DIR/reports"

# Test execution flags
RUN_UNIT_TESTS=true
RUN_INTEGRATION_TESTS=true
RUN_API_TESTS=true
RUN_SLOW_TESTS=false
GENERATE_COVERAGE=true
GENERATE_HTML_REPORT=true
VERBOSE=false
PARALLEL=false
PROFILE=false

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================================${NC}"
    echo
}

print_section() {
    echo -e "${CYAN}--- $1 ---${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${PURPLE}ℹ $1${NC}"
}

show_help() {
    cat << EOF
Echo Brain Board of Directors Test Suite Runner

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -u, --unit-only         Run only unit tests
    -i, --integration-only  Run only integration tests
    -a, --api-only          Run only API tests
    -s, --slow              Include slow tests
    -f, --fast              Skip slow tests (default)
    -c, --no-coverage       Skip coverage reporting
    -r, --no-html           Skip HTML report generation
    -v, --verbose           Verbose output
    -p, --parallel          Run tests in parallel (requires pytest-xdist)
    -P, --profile           Enable performance profiling
    -m, --markers MARKERS   Run tests with specific markers (e.g., "unit and not slow")
    -k, --keyword KEYWORD   Run tests matching keyword
    -x, --stop-on-fail      Stop on first failure
    -l, --last-failed       Run only tests that failed in the last run
    --clean                 Clean previous test artifacts
    --install-deps          Install test dependencies
    --check-deps            Check if test dependencies are installed

Examples:
    $0                          # Run all tests with coverage
    $0 -u -v                    # Run unit tests with verbose output
    $0 -i --slow                # Run integration tests including slow ones
    $0 -m "security or quality" # Run tests marked as security or quality
    $0 -k "test_consensus"       # Run tests with "consensus" in the name
    $0 --clean --install-deps   # Clean and reinstall dependencies

Test Categories:
    unit         - Fast, isolated unit tests
    integration  - End-to-end integration tests
    api          - API endpoint tests
    slow         - Performance and stress tests
    database     - Database connectivity tests
    async_test   - Asynchronous functionality tests
    security     - Security-related tests
    quality      - Code quality tests
    consensus    - Consensus algorithm tests

EOF
}

check_dependencies() {
    print_section "Checking Dependencies"

    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi

    # Check if virtual environment exists
    if [[ ! -d "$VENV_DIR" ]]; then
        print_warning "Virtual environment not found at $VENV_DIR"
        print_info "Run with --install-deps to create virtual environment"
        return 1
    fi

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Check if pytest is installed
    if ! python -c "import pytest" &> /dev/null; then
        print_warning "pytest not found in virtual environment"
        return 1
    fi

    print_success "Dependencies check passed"
    return 0
}

install_dependencies() {
    print_section "Installing Test Dependencies"

    # Create virtual environment if it doesn't exist
    if [[ ! -d "$VENV_DIR" ]]; then
        print_info "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip

    # Install test dependencies
    print_info "Installing test dependencies..."
    pip install pytest>=6.0
    pip install pytest-asyncio>=0.15.0
    pip install pytest-cov>=2.10.0
    pip install pytest-html>=3.0.0
    pip install pytest-timeout>=1.4.0
    pip install pytest-mock>=3.0.0
    pip install pytest-xdist>=2.0.0  # For parallel testing
    pip install pytest-benchmark>=3.4.0  # For performance testing
    pip install pytest-profiling>=1.7.0  # For profiling
    pip install coverage>=5.0
    pip install psutil>=5.7.0  # For resource monitoring

    # Install application dependencies
    if [[ -f "$SCRIPT_DIR/requirements.txt" ]]; then
        print_info "Installing application dependencies..."
        pip install -r "$SCRIPT_DIR/requirements.txt"
    fi

    print_success "Dependencies installed successfully"
}

clean_artifacts() {
    print_section "Cleaning Test Artifacts"

    # Remove cache directories
    find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$SCRIPT_DIR" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

    # Remove coverage files
    rm -f "$SCRIPT_DIR/.coverage" 2>/dev/null || true
    rm -rf "$COVERAGE_DIR" 2>/dev/null || true

    # Remove reports
    rm -rf "$REPORTS_DIR" 2>/dev/null || true
    rm -f "$TESTS_DIR/test.log" 2>/dev/null || true
    rm -f "$TESTS_DIR/coverage.xml" 2>/dev/null || true
    rm -f "$TESTS_DIR/report.html" 2>/dev/null || true

    # Remove compiled Python files
    find "$SCRIPT_DIR" -name "*.pyc" -delete 2>/dev/null || true
    find "$SCRIPT_DIR" -name "*.pyo" -delete 2>/dev/null || true

    print_success "Test artifacts cleaned"
}

setup_test_environment() {
    print_section "Setting Up Test Environment"

    # Create necessary directories
    mkdir -p "$REPORTS_DIR"
    mkdir -p "$COVERAGE_DIR"

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Set environment variables for testing
    export TESTING=true
    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
    export DB_HOST="localhost"
    export DB_NAME="test_echo_brain"
    export DB_USER="test_user"
    export DB_PASSWORD="test_password"

    print_success "Test environment configured"
}

run_tests() {
    local test_args=()
    local markers=()

    print_section "Running Tests"

    # Build test arguments
    if [[ "$RUN_UNIT_TESTS" == true && "$RUN_INTEGRATION_TESTS" == false && "$RUN_API_TESTS" == false ]]; then
        markers+=("unit")
    elif [[ "$RUN_INTEGRATION_TESTS" == true && "$RUN_UNIT_TESTS" == false && "$RUN_API_TESTS" == false ]]; then
        markers+=("integration")
    elif [[ "$RUN_API_TESTS" == true && "$RUN_UNIT_TESTS" == false && "$RUN_INTEGRATION_TESTS" == false ]]; then
        markers+=("api")
    fi

    if [[ "$RUN_SLOW_TESTS" == false ]]; then
        markers+=("not slow")
    fi

    # Add marker selection
    if [[ ${#markers[@]} -gt 0 ]]; then
        local marker_expr=$(IFS=" and "; echo "${markers[*]}")
        test_args+=("-m" "$marker_expr")
    fi

    # Add custom markers if specified
    if [[ -n "$CUSTOM_MARKERS" ]]; then
        test_args+=("-m" "$CUSTOM_MARKERS")
    fi

    # Add keyword filter if specified
    if [[ -n "$KEYWORD_FILTER" ]]; then
        test_args+=("-k" "$KEYWORD_FILTER")
    fi

    # Add verbosity
    if [[ "$VERBOSE" == true ]]; then
        test_args+=("-v" "-s")
    fi

    # Add coverage
    if [[ "$GENERATE_COVERAGE" == true ]]; then
        test_args+=(
            "--cov=directors"
            "--cov=board_api"
            "--cov=echo_board_integration"
            "--cov-report=term-missing"
            "--cov-report=xml:$TESTS_DIR/coverage.xml"
        )

        if [[ "$GENERATE_HTML_REPORT" == true ]]; then
            test_args+=("--cov-report=html:$COVERAGE_DIR")
        fi
    fi

    # Add HTML report
    if [[ "$GENERATE_HTML_REPORT" == true ]]; then
        test_args+=("--html=$REPORTS_DIR/report.html" "--self-contained-html")
    fi

    # Add parallel execution
    if [[ "$PARALLEL" == true ]]; then
        test_args+=("-n" "auto")
    fi

    # Add profiling
    if [[ "$PROFILE" == true ]]; then
        test_args+=("--profile")
    fi

    # Add stop on failure
    if [[ "$STOP_ON_FAIL" == true ]]; then
        test_args+=("-x")
    fi

    # Add last failed
    if [[ "$LAST_FAILED" == true ]]; then
        test_args+=("--lf")
    fi

    # Change to script directory
    cd "$SCRIPT_DIR"

    # Run the tests
    print_info "Executing: pytest ${test_args[*]} $TESTS_DIR"
    echo

    if pytest "${test_args[@]}" "$TESTS_DIR"; then
        print_success "All tests passed!"
        return 0
    else
        print_error "Some tests failed!"
        return 1
    fi
}

generate_summary() {
    print_section "Test Summary"

    # Coverage summary
    if [[ "$GENERATE_COVERAGE" == true && -f "$TESTS_DIR/coverage.xml" ]]; then
        print_info "Coverage report generated: $TESTS_DIR/coverage.xml"

        if [[ -d "$COVERAGE_DIR" ]]; then
            print_info "HTML coverage report: $COVERAGE_DIR/index.html"
        fi

        # Extract coverage percentage
        if command -v coverage &> /dev/null; then
            source "$VENV_DIR/bin/activate"
            local coverage_pct=$(coverage report --show-missing | tail -1 | grep -oE '[0-9]+%' || echo "N/A")
            print_info "Total coverage: $coverage_pct"
        fi
    fi

    # HTML report
    if [[ "$GENERATE_HTML_REPORT" == true && -f "$REPORTS_DIR/report.html" ]]; then
        print_info "HTML test report: $REPORTS_DIR/report.html"
    fi

    # Test log
    if [[ -f "$TESTS_DIR/test.log" ]]; then
        print_info "Test log: $TESTS_DIR/test.log"
    fi

    echo
    print_success "Test execution completed!"
}

# ============================================================================
# Main Script
# ============================================================================

main() {
    print_header "Echo Brain Board of Directors Test Suite"

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -u|--unit-only)
                RUN_UNIT_TESTS=true
                RUN_INTEGRATION_TESTS=false
                RUN_API_TESTS=false
                shift
                ;;
            -i|--integration-only)
                RUN_UNIT_TESTS=false
                RUN_INTEGRATION_TESTS=true
                RUN_API_TESTS=false
                shift
                ;;
            -a|--api-only)
                RUN_UNIT_TESTS=false
                RUN_INTEGRATION_TESTS=false
                RUN_API_TESTS=true
                shift
                ;;
            -s|--slow)
                RUN_SLOW_TESTS=true
                shift
                ;;
            -f|--fast)
                RUN_SLOW_TESTS=false
                shift
                ;;
            -c|--no-coverage)
                GENERATE_COVERAGE=false
                shift
                ;;
            -r|--no-html)
                GENERATE_HTML_REPORT=false
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -p|--parallel)
                PARALLEL=true
                shift
                ;;
            -P|--profile)
                PROFILE=true
                shift
                ;;
            -m|--markers)
                CUSTOM_MARKERS="$2"
                shift 2
                ;;
            -k|--keyword)
                KEYWORD_FILTER="$2"
                shift 2
                ;;
            -x|--stop-on-fail)
                STOP_ON_FAIL=true
                shift
                ;;
            -l|--last-failed)
                LAST_FAILED=true
                shift
                ;;
            --clean)
                clean_artifacts
                exit 0
                ;;
            --install-deps)
                install_dependencies
                exit 0
                ;;
            --check-deps)
                if check_dependencies; then
                    print_success "All dependencies are properly installed"
                    exit 0
                else
                    print_error "Missing dependencies. Run with --install-deps to install."
                    exit 1
                fi
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use -h or --help for usage information"
                exit 1
                ;;
        esac
    done

    # Check dependencies
    if ! check_dependencies; then
        print_warning "Installing missing dependencies..."
        install_dependencies
    fi

    # Setup test environment
    setup_test_environment

    # Run tests
    if run_tests; then
        test_result=0
    else
        test_result=1
    fi

    # Generate summary
    generate_summary

    exit $test_result
}

# Run main function
main "$@"
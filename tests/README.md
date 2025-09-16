# Echo Brain Board of Directors Test Suite

This comprehensive test suite provides thorough testing coverage for the Board of Directors system, including all five specialized directors, API endpoints, consensus algorithms, database operations, and end-to-end integration scenarios.

## Test Suite Overview

### Test Structure

```
tests/
├── README.md                 # This documentation
├── conftest.py              # Pytest configuration and fixtures
├── pytest.ini              # Pytest settings and markers
├── verify_tests.py          # Syntax verification utility
├── test_directors.py        # Tests for all 5 specialized directors
├── test_board_api.py        # API endpoint and FastAPI tests
├── test_consensus.py        # Consensus algorithm tests
├── test_database_pool.py    # Database connection pooling tests
└── test_integration.py      # End-to-end integration tests
```

### Test Categories

The test suite is organized using pytest markers for easy filtering:

- **unit**: Fast, isolated unit tests
- **integration**: End-to-end integration tests
- **api**: API endpoint tests
- **slow**: Performance and stress tests
- **database**: Database connectivity tests
- **async_test**: Asynchronous functionality tests
- **security**: Security-related tests
- **quality**: Code quality tests
- **consensus**: Consensus algorithm tests

## Running Tests

### Quick Start

```bash
# Run all tests with coverage
./run_tests.sh

# Run only unit tests
./run_tests.sh --unit-only

# Run integration tests including slow ones
./run_tests.sh --integration-only --slow

# Run specific test categories
./run_tests.sh -m "security or quality"

# Run tests matching a keyword
./run_tests.sh -k "consensus"
```

### Test Runner Options

The `run_tests.sh` script provides comprehensive options:

```bash
# Basic options
./run_tests.sh -h                    # Show help
./run_tests.sh -v                    # Verbose output
./run_tests.sh -x                    # Stop on first failure
./run_tests.sh -l                    # Run only last failed tests

# Test selection
./run_tests.sh --unit-only           # Only unit tests
./run_tests.sh --integration-only    # Only integration tests
./run_tests.sh --api-only            # Only API tests
./run_tests.sh --slow                # Include slow tests
./run_tests.sh -m "unit and not slow" # Custom marker expression

# Reporting
./run_tests.sh --no-coverage         # Skip coverage reporting
./run_tests.sh --no-html            # Skip HTML report generation

# Performance
./run_tests.sh --parallel            # Run tests in parallel
./run_tests.sh --profile            # Enable performance profiling

# Setup
./run_tests.sh --install-deps        # Install test dependencies
./run_tests.sh --check-deps          # Check if dependencies are installed
./run_tests.sh --clean              # Clean test artifacts
```

### Manual pytest Execution

You can also run tests directly with pytest:

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_directors.py

# Run with coverage
pytest tests/ --cov=directors --cov=board_api --cov-report=html

# Run with markers
pytest tests/ -m "unit and not slow"

# Run specific test
pytest tests/test_directors.py::TestSecurityDirector::test_sql_injection_detection
```

## Test Coverage

### Directors Module Tests (test_directors.py)

Tests all five specialized directors:

**SecurityDirector**:
- SQL injection detection
- XSS vulnerability identification
- Secure code approval
- Authentication security analysis

**QualityDirector**:
- Code complexity detection
- Naming convention analysis
- Documentation requirements
- Best practices compliance

**PerformanceDirector**:
- Inefficient algorithm detection
- Database query optimization
- Performance bottleneck identification

**EthicsDirector**:
- Privacy violation detection
- Bias identification in algorithms
- Compliance checking

**UXDirector**:
- Accessibility issue detection
- Usability concern identification
- User experience evaluation

### API Tests (test_board_api.py)

Tests FastAPI endpoints:

- Task submission endpoints
- Decision retrieval endpoints
- Task listing and filtering
- Feedback submission
- WebSocket connections
- Authentication integration
- Error handling scenarios

### Consensus Tests (test_consensus.py)

Tests director consensus algorithms:

- Unanimous approval scenarios
- Mixed recommendation handling
- High threshold consensus requirements
- Confidence-weighted consensus
- Task routing to appropriate directors
- Conflict resolution strategies
- Performance metrics tracking

### Database Tests (test_database_pool.py)

Tests database connection management:

- Connection pool creation and configuration
- Thread-safe connection acquisition
- Context manager usage
- Connection cleanup and resource management
- Error handling and recovery
- Performance under concurrent load

### Integration Tests (test_integration.py)

Tests end-to-end system functionality:

- Complete evaluation workflows
- Real-world code analysis scenarios
- Async operation handling
- Error recovery and resilience
- Multi-director coordination
- System performance under load

## Test Data and Fixtures

### Common Fixtures (conftest.py)

- **mock_user**: Authenticated user with full permissions
- **limited_user**: User with restricted permissions
- **sample_task_data**: Realistic task data for testing
- **vulnerable_code_sample**: Code samples with security issues
- **quality_issues_sample**: Code samples with quality problems
- **mock_database_pool**: Mock database connections
- **performance_monitor**: Performance monitoring utilities

### Environment Setup

Tests use isolated environments with:

- Mock database connections
- Test-specific configuration
- Clean environment variables
- Temporary file handling
- Logging capture utilities

## Coverage Requirements

The test suite aims for comprehensive coverage:

- **Critical Components**: ≥90% coverage
  - Director evaluation logic
  - Consensus algorithms
  - Security detection patterns

- **Supporting Components**: ≥70% coverage
  - API endpoints
  - Database operations
  - Integration workflows

- **Overall Target**: ≥80% total coverage

## Performance Testing

### Load Testing Scenarios

The test suite includes performance tests for:

- Concurrent task evaluation
- Database connection pooling under load
- WebSocket connection handling
- Memory usage patterns
- Response time requirements

### Benchmarking

Performance benchmarks are established for:

- Individual director evaluation times (< 1 second)
- Consensus calculation (< 2 seconds)
- API response times (< 500ms)
- Database query performance (< 100ms)

## Security Testing

### Vulnerability Detection Tests

Tests verify detection of:

- SQL injection patterns
- Cross-site scripting (XSS)
- Authentication bypasses
- Data exposure vulnerabilities
- Input validation failures

### Security Test Data

Includes realistic examples of:

- Vulnerable authentication code
- Unsafe database queries
- Poor input handling
- Privacy violations
- Weak cryptographic implementations

## Error Handling and Resilience

### Error Scenarios Tested

- Network connectivity failures
- Database unavailability
- Partial director failures
- Invalid input handling
- Resource exhaustion
- Timeout conditions

### Recovery Testing

Verifies system recovery from:

- Temporary service outages
- Database connection losses
- Memory pressure
- Concurrent access conflicts

## Continuous Integration

### Test Automation

The test suite is designed for CI/CD integration:

- Fast feedback for unit tests (< 30 seconds)
- Comprehensive integration tests (< 5 minutes)
- Automated coverage reporting
- Performance regression detection

### Test Reporting

Generates multiple report formats:

- HTML coverage reports
- XML coverage for CI tools
- JUnit XML for test results
- Performance profiling reports

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH=/opt/tower-echo-brain:$PYTHONPATH
```

**Database Connection Issues**:
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Verify test database exists
psql -h localhost -U postgres -c "CREATE DATABASE test_echo_brain;"
```

**Dependency Issues**:
```bash
# Reinstall test dependencies
./run_tests.sh --install-deps

# Check virtual environment
source venv/bin/activate
pip list
```

### Debug Mode

Run tests with debug output:

```bash
# Enable verbose logging
pytest tests/ -v -s --log-cli-level=DEBUG

# Run single test with debugging
pytest tests/test_directors.py::TestSecurityDirector::test_sql_injection_detection -v -s
```

## Contributing

### Adding New Tests

When adding new tests:

1. Follow existing naming conventions
2. Use appropriate pytest markers
3. Include both positive and negative test cases
4. Add fixtures for reusable test data
5. Document complex test scenarios
6. Ensure tests are deterministic and isolated

### Test Standards

- Each test should be independent
- Use descriptive test names
- Include docstrings for complex tests
- Mock external dependencies
- Test both success and failure paths
- Verify error messages and status codes

## Maintenance

### Regular Tasks

- Review and update test data monthly
- Monitor test execution times
- Update dependency versions quarterly
- Review coverage reports weekly
- Validate performance benchmarks

### Test Health Monitoring

The test suite includes health checks for:

- Test execution time trends
- Coverage percentage tracking
- Flaky test identification
- Performance regression detection

---

## Quick Reference

### Essential Commands

```bash
# Install and verify setup
./run_tests.sh --install-deps
./run_tests.sh --check-deps

# Run comprehensive test suite
./run_tests.sh

# Development workflow
./run_tests.sh --unit-only -v          # Quick feedback
./run_tests.sh --integration-only      # Full integration
./run_tests.sh -l                      # Re-run failures

# CI/CD workflow
./run_tests.sh -m "not slow"           # Fast CI tests
./run_tests.sh --slow                  # Nightly comprehensive tests
```

### Coverage Targets

- Unit Tests: 100+ test cases
- Integration Tests: 20+ scenarios
- API Tests: 50+ endpoint variations
- Performance Tests: 10+ load scenarios
- Security Tests: 25+ vulnerability patterns

This comprehensive test suite ensures the reliability, security, and performance of the Echo Brain Board of Directors system.
# Echo Brain Anime Generation Testing Framework

A comprehensive testing framework for validating the Echo Brain anime generation system with realistic performance metrics, visual validation, and continuous integration support.

## Overview

This testing framework addresses the critical gaps identified in the anime production system by focusing on **ACTUAL functionality validation** rather than aspirational performance metrics. The framework includes:

- **Visual Validation** using LLaVA model for character consistency
- **Performance Benchmarking** with realistic thresholds
- **Database Consistency** testing with ACID compliance
- **Integration Testing** across all service dependencies
- **CI/CD Workflows** with GitHub Actions automation

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL database access
- ComfyUI service running on port 8188
- Echo Brain service running on port 8309
- Anime Production API running on port 8328
- (Optional) LLaVA model for visual validation

### Installation

```bash
# Install testing dependencies
cd /opt/tower-echo-brain
pip install pytest pytest-asyncio pytest-cov pytest-benchmark
pip install httpx websockets psycopg2-binary redis pillow opencv-python

# Create required directories
mkdir -p tests/logs tests/junit tests/data
```

### Running Tests

#### Quick Test Suite
```bash
# Run smoke tests (basic functionality)
python tests/run_tests.py smoke

# Run unit tests only
python tests/run_tests.py unit --fast

# Check service availability
python tests/run_tests.py --check-services
```

#### Comprehensive Testing
```bash
# Run all test suites
python tests/run_tests.py comprehensive

# Run specific test suites
python tests/run_tests.py integration --service=comfyui
python tests/run_tests.py visual
python tests/run_tests.py performance --fast
```

#### Using pytest directly
```bash
# Run with pytest
pytest tests/unit/ -v
pytest tests/integration/ -m "not slow"
pytest tests/performance/ --benchmark-only
```

## Test Structure

```
tests/
├── conftest.py                 # Pytest configuration and fixtures
├── pytest.ini                 # Pytest settings
├── run_tests.py               # Test runner script
├── test_plan.md              # Comprehensive test strategy
├── fixtures/                 # Test data and configurations
│   └── test_data.json        # Character, project, and scenario data
├── unit/                     # Component-level tests
│   └── test_database_operations.py
├── integration/              # Service integration tests
│   ├── test_character_generation.py
│   └── test_comfyui_integration.py
├── performance/              # Benchmarking and load tests
│   └── test_generation_speed.py
├── visual/                   # Visual validation tests
│   └── visual_validator.py
└── monitoring/               # System health tests
    └── test_service_health.py
```

## Test Categories

### 1. Unit Tests (`tests/unit/`)

**Focus**: Individual component validation
**Speed**: Fast (<30 seconds)
**Dependencies**: Minimal

- Database ACID compliance testing
- Authentication and authorization
- Configuration management
- Error handling validation

```bash
pytest tests/unit/ -m "not slow"
```

### 2. Integration Tests (`tests/integration/`)

**Focus**: Service communication and workflows
**Speed**: Medium (30-300 seconds)
**Dependencies**: Running services

- Character generation end-to-end workflows
- ComfyUI integration reliability
- API endpoint validation
- Cross-service communication

```bash
pytest tests/integration/ -v
```

### 3. Visual Validation Tests (`tests/visual/`)

**Focus**: Character consistency and quality
**Speed**: Slow (60-600 seconds)
**Dependencies**: LLaVA model

- Character appearance consistency across generations
- Anime style compliance validation
- Emotional expression accuracy
- Visual quality assessment

```bash
pytest tests/visual/ --tb=short
```

### 4. Performance Tests (`tests/performance/`)

**Focus**: System performance and resource usage
**Speed**: Very slow (300-1800 seconds)
**Dependencies**: All services, GPU

- Generation speed benchmarking
- Concurrent request handling
- Resource utilization monitoring
- API response time validation

```bash
pytest tests/performance/ --benchmark-only
```

## Realistic Performance Thresholds

The framework uses **realistic** performance targets based on actual hardware capabilities:

| Operation | Target | Maximum | Current Reality |
|-----------|--------|---------|-----------------|
| Image Generation | 45s | 60s | 8+ minutes* |
| Short Video (2s) | 90s | 120s | Unknown |
| Long Video (5s) | 240s | 300s | Unknown |
| API Response | 2s | 5s | Variable |
| Character Consistency | 9.0/10 | 8.5/10 min | Untested |

*Current system performance requires significant optimization

## Visual Validation with LLaVA

The framework includes AI-powered visual validation:

```python
# Character consistency testing
result = await validator.validate_image(
    image_path="generated_character.png",
    character_name="Ryuu",
    expected_emotion="determined",
    reference_description="Young warrior with spiky black hair"
)

# Validation scores (1-10)
print(f"Consistency: {result.consistency_score}")
print(f"Style: {result.style_score}")
print(f"Quality: {result.quality_score}")
print(f"Emotion match: {result.emotion_match}")
```

## Test Data and Fixtures

### Character Test Data

The framework includes comprehensive test characters:

- **Ryuu**: Fantasy warrior for traditional anime testing
- **Yuki**: Mage character for magical scene testing
- **Kai**: Cyberpunk character for futuristic testing

Each character includes:
- Visual feature specifications
- Personality traits and motivations
- Test scenarios and expected emotions
- Reference images for consistency validation

### Mock Configurations

- API response mocking for offline testing
- Database operation mocking for unit tests
- Service health simulation
- Error scenario simulation

## CI/CD Integration

### GitHub Actions Workflow

The framework includes a comprehensive CI/CD pipeline:

```yaml
# Automated test execution on:
- Push to main/develop branches
- Pull requests
- Nightly performance testing
- Manual workflow dispatch
```

### Test Stages

1. **Unit Tests** - Fast feedback (15 minutes)
2. **Database Tests** - Data integrity validation (20 minutes)
3. **Integration Tests** - Service communication (45 minutes)
4. **Visual Validation** - Character consistency (30 minutes)
5. **Performance Tests** - Benchmarking (60 minutes)
6. **Quality Checks** - Code coverage and linting (10 minutes)

## Configuration

### Environment Variables

```bash
# Test configuration
export TESTING=true
export DATABASE_URL=postgresql://user:pass@host/test_db
export COMFYUI_ENDPOINT=http://***REMOVED***:8188
export ANIME_API_ENDPOINT=http://***REMOVED***:8328
export ECHO_API_ENDPOINT=http://***REMOVED***:8309
export LLAVA_ENDPOINT=http://***REMOVED***:11434
```

### pytest Configuration

Key settings in `pytest.ini`:
- Async test support enabled
- Test markers for categorization
- Coverage reporting configured
- Timeout settings for long-running tests

## Common Use Cases

### 1. Pre-commit Testing
```bash
# Fast validation before code commits
python tests/run_tests.py unit --fast
```

### 2. Pull Request Validation
```bash
# Comprehensive testing for PRs
python tests/run_tests.py comprehensive
```

### 3. Performance Regression Testing
```bash
# Benchmark current vs previous performance
pytest tests/performance/ --benchmark-compare=baseline
```

### 4. Character Consistency Validation
```bash
# Test specific character across scenarios
pytest tests/integration/test_character_generation.py::test_ryuu_consistency
```

### 5. Service Health Monitoring
```bash
# Check all services are operational
python tests/run_tests.py --check-services
```

## Troubleshooting

### Common Issues

1. **LLaVA Service Unavailable**
   ```bash
   # Check Ollama service
   curl http://***REMOVED***:11434/api/tags

   # Visual tests will be skipped automatically
   ```

2. **Database Connection Errors**
   ```bash
   # Test database connectivity
   pg_isready -h ***REMOVED*** -U patrick

   # Check test database configuration
   echo $DATABASE_URL
   ```

3. **ComfyUI Not Responding**
   ```bash
   # Check ComfyUI service
   curl http://***REMOVED***:8188/system_stats

   # Restart ComfyUI if needed
   sudo systemctl restart comfyui
   ```

4. **GPU Memory Issues**
   ```bash
   # Check GPU memory usage
   nvidia-smi

   # Reduce batch sizes in test configurations
   ```

### Debug Mode

```bash
# Enable verbose logging
python tests/run_tests.py comprehensive --verbose

# Run single test with full output
pytest tests/integration/test_comfyui_integration.py::test_simple_workflow_execution -v -s
```

## Test Reports

### Automated Reporting

Test results are automatically collected and saved:
- JUnit XML format for CI/CD integration
- JSON reports with detailed metrics
- Coverage reports in HTML format
- Performance benchmarks with historical comparison

### Manual Report Generation

```bash
# Generate comprehensive test report
python tests/run_tests.py comprehensive --output=test_report.json

# View coverage report
python -m http.server 8080 -d tests/htmlcov/
```

## Development Guidelines

### Adding New Tests

1. **Choose appropriate test category** (unit, integration, visual, performance)
2. **Use existing fixtures** from `conftest.py`
3. **Follow naming conventions** (`test_*.py`)
4. **Add appropriate markers** (`@pytest.mark.slow`, etc.)
5. **Include docstrings** with test purpose and expected outcomes

### Test Data Management

- Add character data to `fixtures/test_data.json`
- Use temporary directories for test outputs
- Clean up resources in test teardown
- Mock external dependencies appropriately

### Performance Testing

- Set realistic thresholds based on hardware
- Include both target and maximum acceptable times
- Monitor resource usage (GPU, memory, disk)
- Compare against baselines for regression detection

## Contributing

1. Run existing tests to ensure no regressions
2. Add tests for new functionality
3. Update documentation for new test categories
4. Ensure CI/CD pipeline passes
5. Include performance impact assessment

## Support and Maintenance

For issues with the testing framework:
1. Check service availability first
2. Review test logs for specific error messages
3. Verify environment configuration
4. Update test data if business logic changes
5. Maintain realistic performance thresholds

---

**Testing Framework Status**: Production Ready
**Last Updated**: November 19, 2025
**Framework Version**: 1.0.0
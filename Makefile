# Makefile for Echo Brain Board of Directors CI/CD Pipeline
# Production-grade testing and automation

# ============================================================================
# Configuration
# ============================================================================

PYTHON := python3
PIP := pip3
PYTEST := pytest
VENV_DIR := venv
TEST_DIR := tests
SRC_DIRS := directors board_api.py echo_board_integration.py
COV_DIR := htmlcov
REPORTS_DIR := reports

# Test execution options
PYTEST_OPTS := -v --tb=short --color=yes
PYTEST_COV_OPTS := --cov=directors --cov=board_api --cov=echo_board_integration
PYTEST_COVERAGE_MIN := 80

# Parallel test execution (adjust based on CPU cores)
PYTEST_PARALLEL := -n auto

# ============================================================================
# Help Target
# ============================================================================

.PHONY: help
help: ## Show this help message
	@echo "Echo Brain Board of Directors - CI/CD Pipeline"
	@echo "=============================================="
	@echo ""
	@echo "Available targets:"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@echo "  make test              # Run all tests"
	@echo "  make test-unit         # Run only unit tests"
	@echo "  make test-integration  # Run only integration tests"
	@echo "  make coverage          # Generate coverage report"
	@echo "  make lint              # Run all code quality checks"
	@echo "  make security          # Run security scans"
	@echo "  make ci                # Run full CI pipeline"
	@echo ""

# ============================================================================
# Environment Setup
# ============================================================================

.PHONY: setup
setup: ## Set up development environment
	@echo "Setting up development environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip setuptools wheel
	$(VENV_DIR)/bin/pip install -r requirements.txt
	$(VENV_DIR)/bin/pip install -r test_requirements.txt
	@echo "Development environment setup complete!"

.PHONY: install-deps
install-deps: ## Install dependencies
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install -r test_requirements.txt

.PHONY: install-test-deps
install-test-deps: ## Install test dependencies only
	@echo "Installing test dependencies..."
	$(PIP) install -r test_requirements.txt

# ============================================================================
# Testing Targets
# ============================================================================

.PHONY: test
test: ## Run all tests
	@echo "Running all tests..."
	@mkdir -p $(REPORTS_DIR)
	$(PYTEST) $(PYTEST_OPTS) $(PYTEST_COV_OPTS) --cov-report=html:$(COV_DIR) \
		--cov-report=xml:$(REPORTS_DIR)/coverage.xml \
		--cov-report=json:$(REPORTS_DIR)/coverage.json \
		--cov-report=term-missing \
		--cov-fail-under=$(PYTEST_COVERAGE_MIN) \
		--html=$(REPORTS_DIR)/report.html \
		--json-report --json-report-file=$(REPORTS_DIR)/test_results.json \
		$(TEST_DIR)

.PHONY: test-unit
test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	$(PYTEST) $(PYTEST_OPTS) -m "unit or not integration" $(TEST_DIR)

.PHONY: test-integration
test-integration: ## Run integration tests only
	@echo "Running integration tests..."
	$(PYTEST) $(PYTEST_OPTS) -m "integration" $(TEST_DIR)

.PHONY: test-api
test-api: ## Run API tests only
	@echo "Running API tests..."
	$(PYTEST) $(PYTEST_OPTS) -m "api" $(TEST_DIR)

.PHONY: test-security
test-security: ## Run security tests only
	@echo "Running security tests..."
	$(PYTEST) $(PYTEST_OPTS) -m "security or auth" $(TEST_DIR)

.PHONY: test-performance
test-performance: ## Run performance tests
	@echo "Running performance tests..."
	$(PYTEST) $(PYTEST_OPTS) -m "performance" --benchmark-only $(TEST_DIR)

.PHONY: test-parallel
test-parallel: ## Run tests in parallel
	@echo "Running tests in parallel..."
	$(PYTEST) $(PYTEST_OPTS) $(PYTEST_PARALLEL) $(TEST_DIR)

.PHONY: test-fast
test-fast: ## Run fast tests only (exclude slow tests)
	@echo "Running fast tests..."
	$(PYTEST) $(PYTEST_OPTS) -m "not slow" $(TEST_DIR)

.PHONY: test-slow
test-slow: ## Run slow tests only
	@echo "Running slow tests..."
	$(PYTEST) $(PYTEST_OPTS) -m "slow" $(TEST_DIR)

.PHONY: test-watch
test-watch: ## Run tests in watch mode
	@echo "Running tests in watch mode..."
	$(PYTEST) $(PYTEST_OPTS) --testmon $(TEST_DIR)

# ============================================================================
# Coverage Targets
# ============================================================================

.PHONY: coverage
coverage: ## Generate coverage report
	@echo "Generating coverage report..."
	@mkdir -p $(REPORTS_DIR)
	$(PYTEST) $(PYTEST_COV_OPTS) --cov-report=html:$(COV_DIR) \
		--cov-report=xml:$(REPORTS_DIR)/coverage.xml \
		--cov-report=json:$(REPORTS_DIR)/coverage.json \
		--cov-report=term-missing \
		$(TEST_DIR)
	@echo "Coverage report generated in $(COV_DIR)/index.html"

.PHONY: coverage-badge
coverage-badge: ## Generate coverage badge
	coverage-badge -o $(REPORTS_DIR)/coverage.svg

.PHONY: coverage-open
coverage-open: coverage ## Open coverage report in browser
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(COV_DIR)/index.html; \
	elif command -v open > /dev/null; then \
		open $(COV_DIR)/index.html; \
	else \
		echo "Coverage report available at $(COV_DIR)/index.html"; \
	fi

# ============================================================================
# Code Quality Targets
# ============================================================================

.PHONY: lint
lint: lint-flake8 lint-black lint-isort lint-mypy ## Run all linting checks

.PHONY: lint-flake8
lint-flake8: ## Run flake8 linting
	@echo "Running flake8..."
	flake8 $(SRC_DIRS) $(TEST_DIR) --max-line-length=100 --extend-ignore=E203,W503

.PHONY: lint-black
lint-black: ## Check code formatting with black
	@echo "Checking code formatting with black..."
	black --check --diff $(SRC_DIRS) $(TEST_DIR)

.PHONY: lint-isort
lint-isort: ## Check import sorting with isort
	@echo "Checking import sorting with isort..."
	isort --check-only --diff $(SRC_DIRS) $(TEST_DIR)

.PHONY: lint-mypy
lint-mypy: ## Run mypy type checking
	@echo "Running mypy type checking..."
	mypy $(SRC_DIRS) --ignore-missing-imports --no-strict-optional

.PHONY: format
format: format-black format-isort ## Auto-format code

.PHONY: format-black
format-black: ## Format code with black
	@echo "Formatting code with black..."
	black $(SRC_DIRS) $(TEST_DIR)

.PHONY: format-isort
format-isort: ## Sort imports with isort
	@echo "Sorting imports with isort..."
	isort $(SRC_DIRS) $(TEST_DIR)

# ============================================================================
# Security Targets
# ============================================================================

.PHONY: security
security: security-bandit security-safety ## Run all security scans

.PHONY: security-bandit
security-bandit: ## Run bandit security linting
	@echo "Running bandit security scan..."
	@mkdir -p $(REPORTS_DIR)
	bandit -r $(SRC_DIRS) -f json -o $(REPORTS_DIR)/bandit.json || true
	bandit -r $(SRC_DIRS) -f txt -o $(REPORTS_DIR)/bandit.txt || true
	bandit -r $(SRC_DIRS)

.PHONY: security-safety
security-safety: ## Check dependencies for known vulnerabilities
	@echo "Checking dependencies for vulnerabilities..."
	@mkdir -p $(REPORTS_DIR)
	safety check --json --output $(REPORTS_DIR)/safety.json || true
	safety check

.PHONY: security-semgrep
security-semgrep: ## Run semgrep security analysis
	@echo "Running semgrep security analysis..."
	@mkdir -p $(REPORTS_DIR)
	semgrep --config=auto --json --output=$(REPORTS_DIR)/semgrep.json $(SRC_DIRS) || true
	semgrep --config=auto $(SRC_DIRS)

# ============================================================================
# Database Targets
# ============================================================================

.PHONY: test-db-setup
test-db-setup: ## Set up test database
	@echo "Setting up test database..."
	# This would typically set up a test database
	# For now, we'll use SQLite for testing
	@echo "Using SQLite for testing..."

.PHONY: test-db-teardown
test-db-teardown: ## Tear down test database
	@echo "Tearing down test database..."
	rm -f test_*.db

# ============================================================================
# CI/CD Pipeline Targets
# ============================================================================

.PHONY: ci
ci: clean install-test-deps lint security test ## Run full CI pipeline
	@echo "✅ CI pipeline completed successfully!"

.PHONY: ci-fast
ci-fast: clean install-test-deps lint test-fast ## Run fast CI pipeline
	@echo "✅ Fast CI pipeline completed successfully!"

.PHONY: pre-commit
pre-commit: lint-black lint-isort lint-flake8 test-fast ## Pre-commit hook checks
	@echo "✅ Pre-commit checks passed!"

.PHONY: pre-push
pre-push: lint security test ## Pre-push hook checks
	@echo "✅ Pre-push checks passed!"

# ============================================================================
# Documentation Targets
# ============================================================================

.PHONY: docs
docs: ## Generate documentation
	@echo "Generating documentation..."
	# This would generate documentation (e.g., with Sphinx)
	@echo "Documentation generation not yet implemented"

# ============================================================================
# Cleanup Targets
# ============================================================================

.PHONY: clean
clean: ## Clean up generated files
	@echo "Cleaning up..."
	rm -rf $(COV_DIR)
	rm -rf $(REPORTS_DIR)
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf .mypy_cache
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf */*/__pycache__
	rm -rf *.pyc
	rm -rf */*.pyc
	rm -rf */*/*.pyc
	rm -rf test_*.db
	rm -rf .tox
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info

.PHONY: clean-all
clean-all: clean ## Clean up everything including venv
	rm -rf $(VENV_DIR)

# ============================================================================
# Development Targets
# ============================================================================

.PHONY: dev-setup
dev-setup: setup install-hooks ## Set up development environment with hooks
	@echo "Development environment ready!"

.PHONY: install-hooks
install-hooks: ## Install git hooks
	@echo "Installing git hooks..."
	@if [ -d .git ]; then \
		echo "#!/bin/bash" > .git/hooks/pre-commit; \
		echo "make pre-commit" >> .git/hooks/pre-commit; \
		chmod +x .git/hooks/pre-commit; \
		echo "#!/bin/bash" > .git/hooks/pre-push; \
		echo "make pre-push" >> .git/hooks/pre-push; \
		chmod +x .git/hooks/pre-push; \
		echo "Git hooks installed successfully!"; \
	else \
		echo "Not a git repository. Skipping git hooks installation."; \
	fi

.PHONY: benchmark
benchmark: ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	$(PYTEST) --benchmark-only --benchmark-sort=mean $(TEST_DIR)

.PHONY: profile
profile: ## Profile test execution
	@echo "Profiling test execution..."
	$(PYTEST) --profile --profile-svg $(TEST_DIR)

# ============================================================================
# Utility Targets
# ============================================================================

.PHONY: check-deps
check-deps: ## Check for outdated dependencies
	@echo "Checking for outdated dependencies..."
	$(PIP) list --outdated

.PHONY: update-deps
update-deps: ## Update dependencies
	@echo "Updating dependencies..."
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) install --upgrade -r test_requirements.txt

.PHONY: freeze-deps
freeze-deps: ## Freeze current dependencies
	@echo "Freezing dependencies..."
	$(PIP) freeze > requirements_frozen.txt

.PHONY: validate-config
validate-config: ## Validate configuration files
	@echo "Validating configuration files..."
	@if [ -f tests/pytest.ini ]; then echo "✅ pytest.ini found"; else echo "❌ pytest.ini missing"; fi
	@if [ -f test_requirements.txt ]; then echo "✅ test_requirements.txt found"; else echo "❌ test_requirements.txt missing"; fi
	@if [ -f requirements.txt ]; then echo "✅ requirements.txt found"; else echo "❌ requirements.txt missing"; fi

# ============================================================================
# Default Target
# ============================================================================

.DEFAULT_GOAL := help
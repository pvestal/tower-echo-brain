# Echo Brain CI/CD Complete Guide

## Overview

This document provides a comprehensive guide to the CI/CD pipeline and deployment capabilities for Echo Brain at `/opt/tower-echo-brain`.

## Repository Structure

```
/opt/tower-echo-brain/
├── .github/
│   ├── workflows/
│   │   ├── ci-cd.yml                    # Main CI/CD pipeline
│   │   ├── comprehensive_testing.yml    # Comprehensive testing
│   │   ├── deployment.yml              # Production deployment
│   │   └── release.yml                 # Release automation
│   ├── branch-protection.json          # Branch protection config
│   └── pull_request_template.md        # PR template
├── scripts/
│   └── deployment/
│       └── deploy.sh                   # Deployment script
├── tests/
│   ├── framework/
│   │   ├── test_markers.py             # Test markers
│   │   └── fixtures.py                 # Shared fixtures
│   └── test_critical_functionality.py  # Critical tests
└── pytest.ini                         # Pytest configuration
```

## CI/CD Pipeline Components

### 1. Main CI/CD Pipeline (`ci-cd.yml`)

**Triggers:**
- Push to `main` branch
- Pull requests to `main` branch

**Jobs:**
- **Code Quality**: Black formatting, Pylint, MyPy, Ruff security checks
- **Testing**: Unit tests with PostgreSQL and Redis services
- **Security Scan**: Trivy security scanning with SARIF output
- **Deploy Notification**: Manual deployment instructions for Tower

**Services:**
- PostgreSQL 16 (test database)
- Redis 7 (caching)

### 2. Comprehensive Testing Pipeline (`comprehensive_testing.yml`)

**Triggers:**
- Push to `main`, `develop`, `feature/*` branches
- Pull requests to `main`, `develop`
- Daily scheduled runs at 2 AM UTC
- Manual workflow dispatch with test type selection

**Test Types:**
- **Unit Tests**: Multi-Python version (3.9, 3.10, 3.11) with coverage
- **Integration Tests**: Database and Redis integration
- **Performance Tests**: Configurable load levels (light/medium/heavy)
- **AI Model Tests**: AI-specific functionality testing
- **Security Tests**: Bandit, Safety, Semgrep scanning
- **Regression Tests**: Baseline comparison

**Features:**
- Parallel test execution
- Coverage reporting with Codecov
- Artifact collection for all test results
- Performance baseline tracking
- PR comment integration with results

### 3. Production Deployment Pipeline (`deployment.yml`)

**Triggers:**
- Push to `main` branch
- Tags matching `v*`
- Manual workflow dispatch with environment selection

**Jobs:**
- **Pre-deployment Validation**: Critical test verification
- **Backup Creation**: Production state backup
- **Deployment**: Manual deployment instructions
- **Rollback**: Emergency rollback procedures
- **Notification**: Deployment status reporting

**Manual Deployment Process:**
```bash
cd /opt/tower-echo-brain
git pull origin main
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart tower-echo-brain
```

### 4. Release Automation (`release.yml`)

**Triggers:**
- Tags matching `v*`
- Manual workflow dispatch with version type

**Features:**
- Automatic changelog generation
- GitHub release creation
- Version management (patch/minor/major)
- Release notes from git history

## Testing Framework

### Test Categories

**Critical Tests** (`@pytest.mark.critical`):
- Core functionality that must work for production
- Import validation
- Configuration loading
- Health endpoint availability
- Database schema validation

**Test Markers:**
- `@critical` - Must pass for deployment
- `@unit` - Fast unit tests
- `@integration` - Integration tests with services
- `@performance` - Performance benchmarks
- `@ai` - AI model functionality
- `@security` - Security validations

### Configuration

**pytest.ini** includes:
- Comprehensive marker definitions
- Async test support
- Coverage configuration
- Warning filters
- Test discovery patterns

### Fixtures

**Shared fixtures** (`tests/framework/fixtures.py`):
- Mock database connections
- Mock Redis connections
- Temporary configuration directories
- Mock Vault client
- Echo test configuration

## Deployment Scripts

### Automated Deployment (`scripts/deployment/deploy.sh`)

**Features:**
- Environment-specific deployment (staging/production)
- Automatic backup creation for production
- Dependency management
- Database migrations
- Service management
- Health checks
- Rollback capability

**Usage:**
```bash
# Production deployment
./scripts/deployment/deploy.sh production

# Staging deployment
./scripts/deployment/deploy.sh staging
```

**Deployment Process:**
1. Create backup (production only)
2. Stop service
3. Pull latest changes
4. Update dependencies
5. Run migrations
6. Start service
7. Verify health
8. Rollback if failed

## Branch Protection

### Main Branch Protection Rules

**Required Status Checks:**
- Setup and Code Quality
- Unit Tests (Python 3.9, 3.10, 3.11)
- Integration Tests
- Security Tests

**Pull Request Requirements:**
- 1 approving review required
- Dismiss stale reviews on update
- Require last push approval
- Conversation resolution required

**Restrictions:**
- No force pushes
- No deletions
- Block direct pushes to main

## Security

### Automated Security Scanning

**Tools Integrated:**
- **Trivy**: Container and filesystem vulnerability scanning
- **Bandit**: Python security linting
- **Safety**: Dependency vulnerability checking
- **Semgrep**: Static code analysis

**SARIF Integration:**
- Security results uploaded to GitHub Security tab
- Automated security alerts
- Dependency vulnerability tracking

### Secrets Management

**Vault Integration:**
- GitHub tokens stored in HashiCorp Vault
- Secure secret retrieval for CI/CD
- No plain text secrets in repository

## Performance Monitoring

### Baseline Tracking

**Metrics Collected:**
- Test execution times
- Performance benchmarks
- Memory usage patterns
- API response times

**Regression Detection:**
- Automated baseline comparison
- Performance degradation alerts
- Historical trend analysis

## Manual Operations

### Emergency Deployment

For critical fixes bypassing normal pipeline:

```bash
# On Tower server
cd /opt/tower-echo-brain

# Create emergency backup
sudo systemctl stop tower-echo-brain
cp -r . /opt/backups/emergency-$(date +%Y%m%d-%H%M%S)

# Deploy fix
git pull origin main
source venv/bin/activate
pip install -r requirements.txt

# Start service
sudo systemctl start tower-echo-brain

# Verify
curl -s https://***REMOVED***/api/echo/health
```

### Rollback Procedure

```bash
# Stop service
sudo systemctl stop tower-echo-brain

# Restore from backup
LATEST_BACKUP=$(ls -t /opt/backups/echo-brain/ | head -1)
rm -rf /opt/tower-echo-brain
cp -r "/opt/backups/echo-brain/$LATEST_BACKUP" /opt/tower-echo-brain

# Restart service
sudo systemctl start tower-echo-brain
```

## Monitoring and Alerts

### Health Checks

**Endpoint:** `https://***REMOVED***/api/echo/health`

**Automated Monitoring:**
- Service status verification
- Database connectivity
- Redis connectivity
- Model availability

### Log Monitoring

**Locations:**
- Service logs: `journalctl -u tower-echo-brain`
- Application logs: `/opt/tower-echo-brain/logs/`
- Deployment logs: `/tmp/echo-brain-deployment.log`

## Development Workflow

### Feature Development

1. **Create Feature Branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Development:**
   - Write code with tests
   - Run local tests: `pytest tests/`
   - Commit changes with conventional commits

3. **Pull Request:**
   - Push feature branch
   - Create PR to `main`
   - Address review feedback
   - Wait for CI/CD pipeline completion

4. **Merge:**
   - Squash and merge after approvals
   - Automatic deployment to production

### Hotfix Workflow

1. **Create Hotfix Branch:**
   ```bash
   git checkout -b hotfix/critical-fix
   ```

2. **Quick Fix:**
   - Minimal changes for critical issue
   - Emergency testing only
   - Fast-track PR process

3. **Emergency Deployment:**
   - Use workflow dispatch for immediate deployment
   - Manual verification on production

## Troubleshooting

### Common Issues

**CI/CD Pipeline Failures:**
- Check GitHub Actions logs
- Verify test database connectivity
- Review dependency conflicts
- Check security scan results

**Deployment Issues:**
- Verify service status: `systemctl status tower-echo-brain`
- Check application logs
- Validate configuration files
- Test database connectivity

**Performance Degradation:**
- Review performance test results
- Check baseline metrics
- Monitor resource usage
- Analyze application logs

## Best Practices

### Code Quality
- Maintain test coverage above 80%
- Follow Python PEP 8 standards
- Use type hints throughout codebase
- Write comprehensive docstrings

### Testing
- Write tests before code (TDD)
- Include integration tests for APIs
- Performance test critical paths
- Mock external dependencies

### Deployment
- Always backup before production deployment
- Test deployment scripts in staging
- Verify health checks after deployment
- Monitor logs during deployment

### Security
- Regular dependency updates
- Security scan review
- Vault secret rotation
- Access control management

## Continuous Improvement

### Metrics to Track
- Deployment frequency
- Lead time for changes
- Mean time to recovery
- Change failure rate

### Regular Reviews
- Weekly CI/CD performance review
- Monthly security audit
- Quarterly architecture review
- Annual disaster recovery testing

## Support and Escalation

### Contact Information
- **Primary**: Echo Brain autonomous monitoring
- **Backup**: Manual intervention required
- **Emergency**: Direct server access

### Documentation Updates
This guide should be updated with any changes to:
- CI/CD pipeline modifications
- Deployment procedure changes
- New testing requirements
- Security policy updates

---

**Last Updated:** November 12, 2025
**Version:** 1.0
**Maintainer:** Echo Brain CI/CD System

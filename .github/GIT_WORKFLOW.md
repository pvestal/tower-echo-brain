# Git Workflow for Tower Echo Brain

## Branch Protection Strategy

### Protected Branches
- `main` - Production-ready code
- `feature/*` - Feature development branches
- `fix/*` - Bug fix branches

### Commit Standards
- Use conventional commit format: `feat:`, `fix:`, `refactor:`, `docs:`
- Include detailed description for architectural changes
- Reference issues/tickets when applicable
- Include Co-Authored-By for AI-assisted development

### Merge Requirements
- All changes to `main` should be via Pull Requests
- Require at least 1 review for production changes
- All tests must pass before merging
- Maintain clean, linear history when possible

### Tagging Strategy
- Use semantic versioning: `v{major}.{minor}.{patch}`
- Tag major architectural milestones
- Include detailed release notes in tag annotations
- Format: `v2.0.0-comprehensive-learning` for major features

### Example Workflow
```bash
# Create feature branch
git checkout -b feature/new-learning-component

# Make changes and commit
git add .
git commit -m "feat: add new learning component

Detailed description of changes and their impact"

# Push to remote
git push origin feature/new-learning-component

# Create Pull Request via GitHub interface
# Merge after review and testing
```

### Current Major Milestones
- `v2.0.0-comprehensive-learning` - Multi-source learning architecture
- Future: `v3.0.0` - Enhanced autonomous capabilities

## Repository Health
- Regular automated testing via GitHub Actions
- Code quality checks with pre-commit hooks
- Dependency security scanning
- Documentation updates with major features

This workflow ensures code quality, traceability, and proper collaboration
for the Echo Brain comprehensive learning system.
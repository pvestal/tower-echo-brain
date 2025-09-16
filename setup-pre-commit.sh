#!/bin/bash
# Setup Pre-commit Hooks for Echo Brain Board of Directors
# Production-grade code quality enforcement

set -e

echo "🔧 Setting up pre-commit hooks for Echo Brain Board of Directors..."

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "📦 Installing pre-commit..."
    pip install pre-commit
fi

# Install the git hook scripts
echo "⚙️  Installing pre-commit hooks..."
pre-commit install

# Install hooks for different git events
echo "🎯 Installing hooks for different git events..."
pre-commit install --hook-type pre-commit
pre-commit install --hook-type pre-push
pre-commit install --hook-type commit-msg

# Create secrets baseline if it doesn't exist
if [ ! -f .secrets.baseline ]; then
    echo "🔐 Creating secrets baseline..."
    detect-secrets scan --baseline .secrets.baseline
fi

# Run hooks on all files to verify setup
echo "✅ Running hooks on all files to verify setup..."
pre-commit run --all-files

echo ""
echo "🎉 Pre-commit hooks setup completed successfully!"
echo ""
echo "Pre-commit hooks are now active and will run automatically on:"
echo "  - Every commit (pre-commit hooks)"
echo "  - Every push (pre-push hooks)"
echo ""
echo "You can also run hooks manually:"
echo "  pre-commit run --all-files    # Run all hooks on all files"
echo "  pre-commit run <hook-id>      # Run specific hook"
echo "  pre-commit run --files <file> # Run hooks on specific files"
echo ""
echo "To bypass hooks temporarily (not recommended):"
echo "  git commit --no-verify"
echo "  git push --no-verify"
echo ""
echo "To update hooks to latest versions:"
echo "  pre-commit autoupdate"
echo ""
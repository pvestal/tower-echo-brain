# Echo Brain Repository Cleanup Plan

## Current Issues
- 547MB of backups directory (unnecessary)
- Multiple archive/backup directories totaling ~560MB
- Legacy src.backup.broken directory
- Cache directories scattered throughout
- Test files mixed with production code

## Directories to Remove/Archive

### To Remove Completely (560MB+)
- `backups/` - 547MB of old backups
- `src.backup.broken/` - 2.2MB legacy broken code
- `archive/` - 9.8MB old archives
- `test_results/` - 12KB old test results
- `.pytest_cache/` - 72KB pytest cache
- `.ruff_cache/` - 52KB ruff cache
- All `__pycache__` directories
- All `.pyc` files

### Files to Remove
- `src/app_factory.py.backup.1767546193`
- `src/consciousness_initializer.py.fake_backup`
- `src/manual_gpu_test.py`
- All `*.backup*`, `*.old`, `*.tmp` files

## Clean Directory Structure

```
tower-echo-brain/
├── src/                    # Core application code
│   ├── api/               # API endpoints
│   ├── capabilities/      # Echo Brain capabilities
│   ├── core/             # Core logic
│   ├── db/               # Database management
│   ├── orchestrators/    # Task orchestration
│   ├── managers/         # Service managers
│   ├── middleware/       # Request middleware
│   ├── services/         # External services
│   ├── utils/            # Utilities
│   ├── main.py          # Entry point
│   ├── app_factory.py   # App configuration
│   └── startup.py       # Initialization
├── tests/                 # Test suite
│   ├── test_*.py        # Test files
│   └── *.md             # Test documentation
├── scripts/              # Utility scripts
├── docs/                 # Documentation
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
├── README.md            # Project documentation
└── LICENSE              # License file
```

## Cleanup Commands

```bash
# 1. Remove all cache directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null

# 2. Remove backup directories
rm -rf backups/
rm -rf archive/
rm -rf src.backup.broken/
rm -rf test_results/

# 3. Remove backup files
find . -name "*.backup*" -delete
find . -name "*.old" -delete
find . -name "*.tmp" -delete
find . -name "*_old.py" -delete
find . -name "*_backup.py" -delete

# 4. Remove specific legacy files
rm -f src/app_factory.py.backup.1767546193
rm -f src/consciousness_initializer.py.fake_backup
rm -f src/manual_gpu_test.py
```

## .gitignore Updates

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.mypy_cache/
.ruff_cache/

# Environment
venv/
ENV/
.env
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
*.log
logs/

# Backups
*.backup*
*.old
*.tmp
*_backup.*
*_old.*

# OS
.DS_Store
Thumbs.db

# Project specific
backups/
archive/
test_results/
src.backup*/
```

## Benefits After Cleanup
- Repository size reduced by ~560MB (50%+ reduction)
- Clear, organized directory structure
- No legacy/broken code confusion
- Faster git operations
- Professional repository appearance

## Estimated Cleanup Results
- **Before:** ~1.1GB repository
- **After:** ~540MB repository
- **Space Saved:** 560MB+
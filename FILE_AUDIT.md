# Echo Brain File Audit Report
Generated: 2026-02-02

## Repository Status
- **Remote**: git@github.com:pvestal/tower-echo-brain.git
- **Branch**: main
- **Latest Commit**: 8ae54306 (missing context)
- **Modified Files**: Multiple files with pending changes

## Active System Components

### Working Endpoints (Verified)
1. **Core**: `/`, `/health`, `/docs`, `/redoc`
2. **System API**: `/api/system/*` (health, metrics, status)
3. **Echo API**: `/api/echo/*` (health, brain, status, models)
4. **Conversations**: `/api/conversations/health`
5. **MCP**: `/mcp`, `/mcp/health`
6. **Moltbook**: `/api/echo/moltbook/*` (5 endpoints)
7. **Agents**: `/api/agents/status`

### Active Routers (Currently Mounted)
- `echo_frontend_router.py` - Frontend compatibility endpoints
- `system_router.py` - System monitoring and health
- `conversation_minimal_router.py` - Conversation search
- `moltbook_router.py` - Moltbook integration

### Inactive Routers (Not Mounted)
- `diagnostic_router.py`
- `integration_router.py`
- `dashboard_router.py`
- `monitoring_router.py`
- `production_router.py`
- `echo_frontend.py` (duplicate of echo_frontend_router.py?)

## Root Directory Analysis

### Categories of Files

#### OAuth/Authentication Files (17 files)
**Purpose**: Google Calendar OAuth integration attempts
- `exchange_*.py` - Various OAuth code exchange attempts
- `oauth_*.py` - OAuth configuration files
- `*_token.py` - Token generation/refresh scripts
- **Status**: Mostly obsolete, can be archived

#### Test/Debug Files (8 files)
- `test_*.py` - Various test scripts
- `debug_*.py` - Debug utilities
- `verify_*.py` - Verification scripts
- **Status**: Can move to tests/ directory

#### Documentation Files (14 files)
- System status and structure documentation
- OAuth setup guides
- Tower sync documentation
- **Keep**: README.md, CONTEXT.md, ACTUAL_STATUS.md
- **Archive**: OAuth-related docs

#### Configuration Files (Essential - KEEP)
- `.env` - Environment variables
- `.gitignore` - Git ignore patterns
- `requirements.txt` - Python dependencies
- `models_config.yaml` - Model configurations
- `projects_config.json` - Project settings
- `health_monitor.json` - Health monitoring config
- `.moltbook_config.json` - Moltbook configuration
- `.moltbook_credentials.json` - Moltbook credentials

#### System Files (KEEP)
- `.vault-token` - Vault authentication
- `.encryption_key` - Encryption key
- `.qdrant-initialized` - Qdrant init marker
- `.diagnosis_in_progress` - Diagnostic marker

#### Archive Files
- `*.zip` - Backup archives
- `*.deb` - Installation packages
- `*.log` - Log files

## Duplicates Found
- `exchange_oauth_code.py` appears twice (MD5 match)
- `echo_frontend.py` vs `echo_frontend_router.py` (check purpose)

## Recommended Actions

### 1. Create Directory Structure
```
/opt/tower-echo-brain/
├── src/               # Keep as-is (active code)
├── scripts/           # Keep as-is (operational scripts)
├── docs/              # Consolidate documentation
├── tests/             # Move test files here
├── archive/           # Already exists
│   └── oauth/         # Move OAuth attempts here
├── config/            # Already exists
└── logs/              # Already exists
```

### 2. Files to Move (Not Delete)
- **To archive/oauth/**: All OAuth-related .py files in root
- **To tests/**: All test_*.py, debug_*.py files
- **To docs/**: Non-essential .md files
- **To archive/**: .zip, .deb files

### 3. Files to Keep in Root
- `.env`, `.env.consistent`
- `.gitignore`
- `README.md`
- `requirements.txt`
- Hidden system files (.*-initialized, .vault-token, etc.)
- Configuration files (.json, .yaml)

### 4. Files Needing Review
- `majicmix_download.log` (3.2MB - check if needed)
- Duplicate router files (echo_frontend.py vs echo_frontend_router.py)
- Multiple "final" OAuth scripts (which is actually final?)

## Service Dependencies
- **PostgreSQL**: echo_brain database (42 tables)
- **Qdrant**: 315,222 vectors in echo_memory collection
- **Redis**: Context caching
- **Ollama**: Embeddings generation

## Git Status Summary
- **Deleted files**: 20 files marked for deletion
- **Modified files**: 6 files with changes
- **Action needed**: Review deletions before committing

## Next Steps
1. Review this audit
2. Move files to appropriate directories (not delete)
3. Update imports if files are moved
4. Test all endpoints after reorganization
5. Commit with detailed message
6. Push to GitHub
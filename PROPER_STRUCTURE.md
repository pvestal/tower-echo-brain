# Echo Brain Proper Structure Analysis

## What Actually Belongs Where

### ✅ **NEEDS TO BE IN ROOT**
```
/opt/tower-echo-brain/
├── src/              # Source code
├── venv/             # Virtual environment
├── logs/             # Log files
├── config/           # Configuration (qdrant-config.yaml used by running service)
├── scripts/          # Active scripts (extract_facts_background.py is RUNNING)
├── mcp_server/       # MCP server (ACTIVELY RUNNING)
├── model-manifests/  # Model definitions (NEEDED by models_manager.py)
├── .env              # Environment variables
├── .git/             # Git repository
├── requirements.txt  # Dependencies
└── README.md         # Documentation
```

### ⚠️ **QUESTIONABLE BUT MAY BE NEEDED**
- `cache/` - May be auto-created by services, but could be in /tmp
- `models/` - Code references it but actually uses /mnt/1TB-storage
- `docs/` - Documentation could be in archive but might be needed

### ❌ **DEFINITELY ARCHIVE**
- `frontend/` - No frontend service running
- `nginx/` - Not a web server
- `static/` - No static file serving
- `outputs/` - Random outputs
- `database/` - Only has SQL schemas, not actual DB
- `bin/` - Random binaries
- Old cleanup scripts
- Test files
- Duplicate services

## What's Actually Running

1. **Main Service** (port 8309)
   - Uses: `src/main.py`
   - Needs: src/, venv/, logs/

2. **MCP Server**
   - Process: `/opt/tower-echo-brain/mcp_server/main.py`
   - Needs: mcp_server/

3. **Background Script**
   - Process: `scripts/extract_facts_background.py`
   - Needs: scripts/

4. **Model Manager**
   - Code: `src/api/models_manager.py`
   - Needs: `model-manifests/` directory

5. **Qdrant**
   - Config: `config/qdrant-config.yaml`
   - Needs: config/

## The Real Problem

I was moving things without checking:
- If they're referenced in code
- If they're actively running
- If they're needed for functionality

Model management IS a feature that should work. The model-manifests directory contains important model definitions for anime production.
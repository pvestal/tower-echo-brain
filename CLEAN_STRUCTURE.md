# Echo Brain - ACTUALLY Clean Structure

## Root Directory (12 items only)
```
/opt/tower-echo-brain/
├── src/              # All source code
├── venv/             # Virtual environment
├── logs/             # Log files
├── config/           # Configuration (qdrant config, etc.)
├── scripts/          # Active scripts (fact extraction, etc.)
├── mcp_server/       # MCP server (actively running)
├── archive/          # Everything else organized
├── .env              # Environment variables
├── .encryption_key   # Encryption key
├── .vault-token      # Vault authentication
├── .qdrant-initialized # Qdrant marker
├── .claude/          # Claude configuration
├── .git/             # Git repository
├── .github/          # GitHub actions
├── .gitignore        # Git ignore
├── README.md         # Documentation
└── requirements.txt  # Python requirements
```

## What's Running
- Main service: `src/main.py` (port 8309)
- MCP server: `mcp_server/main.py`
- Background: `scripts/extract_facts_background.py`
- Improvement: `src/improvement/continuous_learning.py`

## Archive Organization
```
archive/
├── definitely_not_needed/  # Confirmed unnecessary
├── old_cleanup_scripts/    # Previous cleanup attempts
├── sql_schemas/           # Database schemas
└── [other organized folders]
```

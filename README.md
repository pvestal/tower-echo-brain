# Echo Brain - Advanced AI Memory System

## Overview
Echo Brain is Patrick's personal AI assistant and memory system with 315,000+ stored memories, providing semantic search, pattern recognition, and autonomous learning capabilities.

## Status
- **Version**: 4.0.0
- **Service**: Active on port 8309
- **Vectors**: 315,222 in Qdrant
- **Database**: 42 tables in PostgreSQL
- **API Coverage**: 93% endpoints operational

## Quick Start

```bash
# Check service status
sudo systemctl status tower-echo-brain

# View logs
sudo journalctl -u tower-echo-brain -f

# Test API
curl http://localhost:8309/health
```

## Architecture

### Core Components
- **FastAPI**: Main application framework
- **Qdrant**: Vector database (315K+ embeddings)
- **PostgreSQL**: Relational data storage
- **Redis**: Context caching
- **Ollama**: Embeddings generation

## Project Structure
```
/opt/tower-echo-brain/
├── src/              # Source code
│   ├── main.py       # Entry point (port 8309)
│   ├── routers/      # API endpoints
│   ├── services/     # Business logic
│   └── integrations/ # External services
├── scripts/          # Utility scripts
├── docs/             # Documentation
├── tests/            # Test files
├── venv/             # Virtual environment
├── logs/             # Log files
└── archive/          # Archived code
```

## API Endpoints

### Working Endpoints (93% Coverage)
- `GET /` - Service info
- `GET /health` - Health check
- `GET /docs` - API documentation
- `/api/system/*` - System monitoring
- `/api/echo/*` - Echo Brain features
- `/api/conversations/*` - Memory search
- `/mcp` - MCP server integration
- `/api/echo/moltbook/*` - Agent network

## Features

### Memory System
- 315,222 vector embeddings
- Semantic search across all domains
- Pattern recognition
- Real-time ingestion

### Integrations
- **Moltbook**: Agent social network
- **Claude MCP**: Native tool support
- **GitHub**: Repository sync
- **Tower Services**: Full ecosystem

## Deployment

```bash
# Restart service
sudo systemctl restart tower-echo-brain

# Check health
curl http://localhost:8309/health

# View documentation
open http://localhost:8309/docs
```

## Recent Updates
- ✅ File structure reorganized
- ✅ Moltbook integration added
- ✅ MCP server implemented
- ✅ Documentation updated

## Repository
GitHub: [pvestal/tower-echo-brain](https://github.com/pvestal/tower-echo-brain)

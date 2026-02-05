# Echo Brain - Unified AI Knowledge System

## Overview
Echo Brain is a personal AI assistant with a unified knowledge layer that combines:
- **24,657** vector embeddings in Qdrant
- **6,129** structured facts in PostgreSQL
- **13,630** conversation messages
- **22** hardcoded core facts for reliability

## Status
- **Version**: 0.4.0 (Unified Knowledge Layer)
- **Service**: Active on port 8309
- **Frontend**: Vue3 dashboard at `/echo-brain/`
- **API**: All endpoints under `/api/echo/*`
- **Database**: PostgreSQL `echo_brain`

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
- **FastAPI**: Backend API framework
- **Vue3 + TypeScript**: Frontend dashboard
- **UnifiedKnowledgeLayer**: Combines all knowledge sources
- **Qdrant**: Vector database (24,657 embeddings, 1024 dims)
- **PostgreSQL**: Facts (6,129) and conversations (13,630)
- **Ollama**: LLM inference (mistral:7b, deepseek-r1:8b)
- **mxbai-embed-large**: Embedding model (1024 dimensions)

## Project Structure
```
/opt/tower-echo-brain/
├── src/                          # Source code
│   ├── main.py                   # FastAPI entry (port 8309)
│   ├── api/endpoints/            # API routers
│   │   └── echo_main_router.py   # Consolidated endpoints
│   ├── core/                     # Core components
│   │   └── unified_knowledge.py  # Knowledge layer
│   └── integrations/             # External services
│       └── mcp_service.py        # MCP integration
├── frontend/                     # Vue3 dashboard
│   ├── src/views/                # Dashboard views
│   └── dist/                     # Built static files
├── scripts/                      # Utility scripts
└── venv/                         # Virtual environment
```

## API Endpoints

### API Endpoints (ALL under `/api/echo/`)
- `GET /health` - Service health check
- `GET /docs` - OpenAPI documentation
- `POST /api/echo/ask` - Main Q&A with unified knowledge
- `POST /api/echo/intelligence/think` - Multi-stage reasoning
- `POST /api/echo/memory/search` - Vector similarity search
- `POST /api/echo/conversations/search` - Search conversations
- `GET /api/echo/facts` - Retrieve structured facts
- `GET /api/echo/metrics` - System metrics
- `GET /api/echo/logs` - Service logs
- `POST /mcp` - Model Context Protocol integration

## Features

### Unified Knowledge Layer
- Combines facts, vectors, and conversations in single responses
- Core facts hardcoded (port 8309, endpoints, architecture)
- Parallel retrieval from all sources
- Transparent source attribution

### Knowledge Sources
- **Vectors**: 24,657 embeddings from conversations/code
- **Facts**: 6,129 structured subject-predicate-object facts
- **Conversations**: 13,630 Claude conversation messages
- **Core Facts**: 22 essential system facts always available

### Frontend Dashboard
- Real-time system monitoring
- Interactive Q&A interface with debug info
- Source visualization (color-coded by type)
- Conversation history tracking

## Deployment

```bash
# Restart service
sudo systemctl restart tower-echo-brain

# Check health
curl http://localhost:8309/health

# View documentation
open http://localhost:8309/docs
```

## Recent Updates (Feb 2026)
- ✅ Unified Knowledge Layer implemented
- ✅ All endpoints consolidated under `/api/echo/`
- ✅ Frontend shows debug info and sources
- ✅ Fixed `/intelligence/think` memory retrieval
- ✅ Expanded core facts from 7 to 22
- ✅ Vector search using HTTP API directly

## Repository
GitHub: [pvestal/tower-echo-brain](https://github.com/pvestal/tower-echo-brain)

# Echo Brain - System Documentation

## 📋 Overview
Echo Brain is a full-stack personal AI assistant platform providing a unified dashboard for interacting with local LLMs, memory systems, and various AI services. It features a Vue.js frontend with a FastAPI backend, offering 48 RESTful and MCP endpoints.

**Core Purpose:** Centralized interface for querying, managing, and monitoring local AI inference, memory, and automation services.

## 🏗️ System Architecture

### High-Level Diagram
```
[User Browser]
        ↓ (HTTPS)
[Cloudflare Tunnel] ← Temporary: `trycloudflare.com` / Target: `vestal-garcia.duckdns.org`
        ↓ (HTTP)
[Nginx @ localhost:443] → Proxies `/echo-brain/` and `/api/`
        ↓
[FastAPI Backend @ localhost:8309] ← Core application
        ↓
[Service Layer] → PostgreSQL, Ollama, Qdrant, ComfyUI, MCP
```

### Component Stack
| Component | Version/Model | Purpose | Port/Access |
| :--- | :--- | :--- | :--- |
| **Frontend** | Vue.js 3 + Router + Pinia | Dashboard UI | Served via Nginx |
| **Backend** | FastAPI (Python 3.12) | API server & logic | `localhost:8309` |
| **Reverse Proxy** | Nginx 1.24.0 | Static files & API routing | `localhost:443/80` |
| **Primary LLM** | `mistral:7b` (via Ollama) | Core reasoning & Q&A | `localhost:11434` |
| **Embedding Model** | `mxbai-embed-large` | Semantic embeddings | `localhost:11434` |
| **Vector Database** | Qdrant 1.11 | Semantic memory storage | `localhost:6333` |
| **Relational DB** | PostgreSQL 16 | Conversation history, logs | `localhost:5432` |
| **Image Pipeline** | ComfyUI | Image generation workflows | `localhost:8188` |
| **External Access** | Cloudflare Tunnel | Secure public endpoint | `trycloudflare.com` |
| **GPU** | NVIDIA RTX 3060 (12GB) | ComfyUI inference | - |
| **GPU** | AMD RX 9070 XT | Ollama inference | - |

## ⚙️ Installation & Configuration

### 1. Prerequisites
Ensure the following are installed and running on the host (`tower`):
```bash
# Core services
sudo systemctl status postgresql ollama qdrant comfyui

# Node.js & Python
node --version  # v18+
python --version  # Python 3.10+

# Check GPU availability
nvidia-smi  # For NVIDIA GPU
rocm-smi    # For AMD GPU
```

### 2. Repository & Dependencies
```bash
git clone <repository-url> /opt/tower-echo-brain
cd /opt/tower-echo-brain

# Backend Python dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend dependencies
cd frontend
npm install
npm run build  # Outputs to `dist/` for Nginx to serve
```

### 3. Core Service Configuration

**PostgreSQL Database:**
```sql
-- Database and user setup
CREATE DATABASE echo_brain;
CREATE USER patrick WITH PASSWORD 'RP78eIrW7cI2jYvL5akt1yurE';
GRANT ALL PRIVILEGES ON DATABASE echo_brain TO patrick;

-- Core tables (auto-created on first run)
-- conversations, echo_conversations, vector_content, facts, ingestion_tracking
```

**Ollama Models:**
```bash
# Primary reasoning model
ollama pull mistral:7b

# Embedding model for vector search
ollama pull mxbai-embed-large

# Verify models are loaded
ollama list
```

**Qdrant Collections:**
```bash
# Primary collection (1024 dimensions)
echo_memory - 61,932 vectors

# Other collections
claude_conversations (384 dims)
echo_memories
scene_embeddings
```

**Environment Variables:**
```bash
# Backend (.env)
DATABASE_URL="postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost/echo_brain"
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="mistral:7b"
QDRANT_URL="http://localhost:6333"
COMFYUI_URL="http://localhost:8188"

# Frontend (.env.production)
VITE_API_URL=  # Empty to use relative paths
```

### 4. Nginx Configuration
The site is served via HTTPS with SSL certificates from Let's Encrypt:

**Main Config (`/etc/nginx/sites-enabled/tower-https`):**
```nginx
server {
    listen 443 ssl http2;
    server_name vestal-garcia.duckdns.org;

    ssl_certificate /etc/letsencrypt/live/vestal-garcia.duckdns.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/vestal-garcia.duckdns.org/privkey.pem;

    # Echo Brain Dashboard
    location /echo-brain/ {
        alias /opt/tower-echo-brain/frontend/dist/;
        try_files $uri $uri/ /echo-brain/index.html;
        add_header Cache-Control "no-cache, no-store, must-revalidate";
    }

    # API Proxy - ALL Echo Brain endpoints
    location /api/ {
        proxy_pass http://127.0.0.1:8309/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
    }

    # MCP Server
    location /mcp {
        proxy_pass http://127.0.0.1:8309/mcp;
        proxy_set_header Host $host;
        proxy_buffering off;
    }
}
```

### 5. Systemd Service
**Service File (`/etc/systemd/system/tower-echo-brain.service`):**
```ini
[Unit]
Description=Tower Echo Brain API
After=network.target postgresql.service

[Service]
Type=simple
User=patrick
WorkingDirectory=/opt/tower-echo-brain
Environment="PATH=/opt/tower-echo-brain/venv/bin"
ExecStart=/opt/tower-echo-brain/venv/bin/python src/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Management:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable tower-echo-brain
sudo systemctl start tower-echo-brain
sudo systemctl status tower-echo-brain
sudo journalctl -u tower-echo-brain -f  # View logs
```

## 🚀 Operation & Access

### Starting the System
```bash
# 1. Start all dependent services
sudo systemctl start postgresql ollama qdrant comfyui

# 2. Start Echo Brain backend
sudo systemctl start tower-echo-brain

# 3. Ensure Nginx is running
sudo systemctl restart nginx

# 4. (Optional) Create Cloudflare tunnel for external access
cloudflared tunnel --url https://localhost --no-tls-verify
```

### Access Points
| Access Type | URL | Notes |
| :--- | :--- | :--- |
| **Local Dashboard** | `http://localhost:8309/echo-brain/` | Direct access |
| **Local API** | `http://localhost:8309/api/` | Backend API |
| **LAN Access** | `http://192.168.50.135/echo-brain/` | From network devices |
| **Public Domain** | `https://vestal-garcia.duckdns.org/echo-brain/` | Requires port forwarding |
| **Cloudflare Tunnel** | `https://*.trycloudflare.com/echo-brain/` | Temporary public access |

### Dashboard Views
The Vue.js dashboard provides six main views:

1. **Dashboard** - Real-time system health monitoring
   - Service status (PostgreSQL, Ollama, Qdrant, MCP, ComfyUI)
   - Resource usage (CPU, Memory, Disk, GPU)
   - Uptime and endpoint statistics

2. **Ask** - Primary Q&A interface
   - Natural language queries
   - Conversation history
   - Stream mode toggle
   - Related memories display

3. **Memory** - Semantic memory management
   - Search with custom limits
   - Ingest new content with metadata
   - View search results with similarity scores
   - Memory status and health checks

4. **Endpoints** - Interactive API testing
   - Test all 48 endpoints
   - Custom request builder
   - Response viewer
   - Test history tracking

5. **System** - Service control panel
   - Echo Brain Core operations
   - Intelligence Engine controls
   - Conversations database search
   - MCP Server operations
   - Moltbook integration
   - System diagnostics
   - Self-test suite

6. **Logs** - Real-time logging and history
   - Service filtering
   - Log level filtering
   - Search functionality
   - Real-time toggle (2s refresh)
   - Export as JSON
   - Activity history

## 🔧 API Overview

### Endpoint Categories (48 Total)
| Category | Count | Key Endpoints |
| :--- | :--- | :--- |
| **Health** | 5 | `/api/health/`, `/api/health/quick`, `/api/health/resources` |
| **System** | 14 | `/api/system/status`, `/api/system/diagnostics`, `/api/system/metrics` |
| **Memory** | 4 | `/api/memory/search`, `/api/memory/ingest`, `/api/memory/status` |
| **Intelligence** | 6 | `/api/intelligence/think`, `/api/intelligence/knowledge-map` |
| **Conversations** | 4 | `/api/conversations/search`, `/api/conversations/health` |
| **Echo** | 5 | `/api/echo/query`, `/api/echo/brain`, `/api/echo/status` |
| **Ask** | 2 | `/api/ask`, `/api/ask/stream` |
| **MCP** | 2 | `/mcp`, `/mcp/health` |
| **Self-test** | 2 | `/api/self-test/quick`, `/api/self-test/run` |
| **Moltbook** | 5 | `/api/echo/moltbook/status`, `/api/echo/moltbook/establish` |

**Interactive API Documentation:** Available at `http://localhost:8309/docs`

### Core API Examples
```bash
# Ask a question
curl -X POST "http://localhost:8309/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Echo Brain?"}'

# Search memory
curl -X POST "http://localhost:8309/api/memory/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "tower system", "limit": 5}'

# Get system health
curl "http://localhost:8309/api/health/"

# MCP search
curl -X POST "http://localhost:8309/mcp" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "search_memory", "arguments": {"query": "test"}}, "id": 1}'
```

## 🐛 Common Issues & Troubleshooting

| Issue | Likely Cause | Solution |
| :--- | :--- | :--- |
| **Dashboard blank/not rendering** | Vue router or build issue | Check browser console, rebuild with `npm run build` |
| **API returns 404** | Nginx proxy misconfiguration | Verify `/api/` location block in nginx config |
| **`POST /ask` times out** | Ollama model not loaded | Run `ollama ps` and `ollama pull mistral:7b` |
| **"Database not ready"** | PostgreSQL connection | Check credentials in `.env`, verify `echo_brain` database exists |
| **Cannot access externally** | Port forwarding or firewall | Check router port 443 forwarding, UFW rules |
| **NAT Loopback issue** | Router limitation | Add hosts entry: `192.168.50.135 vestal-garcia.duckdns.org` |
| **Cloudflare tunnel fails** | Local service down | Verify `curl http://localhost:8309/health` works |
| **Memory search returns empty** | Qdrant not running | Check `sudo systemctl status qdrant` |
| **GPU not being used** | Driver or config issue | Check `nvidia-smi` or `rocm-smi`, verify Ollama GPU support |

### Diagnostic Commands
```bash
# Check all services
sudo systemctl status tower-echo-brain postgresql ollama qdrant comfyui

# View logs
sudo journalctl -u tower-echo-brain -f
sudo tail -f /var/log/nginx/error.log

# Test endpoints locally
curl http://localhost:8309/api/health/
curl http://localhost:8309/api/system/diagnostics

# Check resource usage
htop
nvidia-smi  # GPU usage
df -h       # Disk space

# Database connection test
psql -h localhost -U patrick -d echo_brain -c "SELECT COUNT(*) FROM conversations;"
```

## 📁 Project Structure
```
/opt/tower-echo-brain/
├── src/
│   ├── main.py                     # FastAPI app entry point
│   ├── core/
│   │   ├── reasoning_engine.py     # Main logic for /ask endpoint
│   │   ├── pg_reasoning.py         # PostgreSQL context building
│   │   └── memory_manager.py       # Qdrant integration
│   ├── api/
│   │   └── endpoints/              # All route definitions
│   │       ├── system_router.py
│   │       ├── reasoning_router.py
│   │       ├── memory_router.py
│   │       ├── intelligence_router.py
│   │       └── ...
│   ├── services/
│   │   ├── health_service.py       # Unified health checks
│   │   ├── ollama_client.py        # LLM interface
│   │   └── qdrant_client.py        # Vector DB interface
│   └── models/                     # Pydantic schemas
├── frontend/                        # Vue.js 3 application
│   ├── src/
│   │   ├── views/                  # Dashboard views
│   │   │   ├── DashboardView.vue
│   │   │   ├── AskView.vue
│   │   │   ├── MemoryView.vue
│   │   │   ├── EndpointsView.vue
│   │   │   ├── SystemView.vue
│   │   │   └── LogsView.vue
│   │   ├── api/
│   │   │   └── echoApi.ts         # API client with all endpoints
│   │   ├── stores/                 # Pinia state management
│   │   └── router/                 # Vue Router config
│   └── dist/                       # Built assets (served by Nginx)
├── scripts/
│   ├── ingest_conversations.py     # Import Claude conversations
│   └── migrate_vectors.py          # Qdrant migration tools
├── requirements.txt                # Python dependencies
├── package.json                    # Node dependencies
└── DEPLOYMENT.md                   # This file
```

## 📊 Metrics & Monitoring

### Key Performance Indicators
- **Response Time:** `/api/ask` should respond within 5-30s depending on model
- **Memory Usage:** PostgreSQL + Qdrant should stay under 4GB RAM
- **Vector Search:** Should return results in <500ms for 60k vectors
- **Dashboard Load:** Should render within 2s on LAN

### Monitoring Commands
```bash
# Real-time metrics
curl http://localhost:8309/api/system/metrics

# Resource usage over time
curl http://localhost:8309/api/system/metrics/history

# Service-specific health
curl http://localhost:8309/api/health/ | jq '.services'
```

## 🔐 Security Considerations

### Current Security Measures
- SSL/TLS via Let's Encrypt certificates
- PostgreSQL password authentication
- UFW firewall with specific port allowances
- Nginx reverse proxy hiding backend ports
- No exposed database ports to internet

### Recommended Improvements
1. Add authentication middleware to API endpoints
2. Implement rate limiting on `/api/ask`
3. Use environment-specific secrets management
4. Enable CORS restrictions
5. Add request logging and anomaly detection

## 📝 Maintenance

### Regular Tasks
```bash
# Weekly: Clean old logs
sudo journalctl --vacuum-time=7d

# Monthly: Update Ollama models
ollama pull mistral:7b
ollama pull mxbai-embed-large

# Quarterly: Backup databases
pg_dump -U patrick echo_brain > backup_$(date +%Y%m%d).sql
qdrant-client export echo_memory

# As needed: Update SSL certificates (auto-renews via certbot)
sudo certbot renew --dry-run
```

### Upgrade Procedure
1. Backup current deployment
2. Test changes in development
3. Stop services: `sudo systemctl stop tower-echo-brain`
4. Pull updates and install dependencies
5. Run migrations if needed
6. Rebuild frontend: `npm run build`
7. Restart services: `sudo systemctl start tower-echo-brain`
8. Verify health: `curl http://localhost:8309/api/health/`

---

**Last Updated:** February 4, 2026
**Version:** 1.0.0
**Maintainer:** Patrick

This document reflects the system **as deployed**. Update this documentation when modifying core services, ports, credentials, or architecture.
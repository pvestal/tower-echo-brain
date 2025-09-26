# Tower Echo Brain - Advanced AI Orchestrator

## Overview

The Tower Echo Brain is a comprehensive AI-powered service that consolidates multiple Echo services into a single intelligent router with Board of Directors decision tracking. It implements dynamic model escalation from 1B to 70B parameters and includes sophisticated governance, conversation memory, and real-time brain visualization.

**Service Port**: 8309
**Production Location**: `/opt/tower-echo-brain/`
**Main Service File**: `echo.py`
**Service Name**: `tower-echo-brain.service`

## Architecture

### Core Components

1. **Echo Brain Unified Service** (`echo.py`)
   - Primary FastAPI service with intelligent model routing
   - Dynamic parameter escalation (1B → 70B models)
   - WebSocket streaming for real-time brain monitoring
   - Background task processing for model operations

2. **Board of Directors System** (`board_api.py`)
   - Governance system with approval workflows
   - Decision tracking and audit trails
   - Performance assessment and recommendations
   - Board member personality system (5 directors)

3. **Model Management System** (`model_manager.py`)
   - CRUD operations for 24+ Ollama models (280GB+ storage)
   - Background operations with progress tracking
   - Automatic resource management and cleanup
   - Model recommendation engine

4. **Decision Engine** (`model_decision_engine.py`)
   - Intelligent model selection based on query complexity
   - Performance monitoring and optimization
   - Context-aware escalation decisions
   - Quality assessment metrics

### Advanced Features

- **Conversation Memory**: PostgreSQL persistence with thought streams
- **Voice Integration**: Complete notification system with multiple characters
- **Expert Personalities**: Specialized AI agents for different domains
- **Testing Framework**: Universal service testing and debugging
- **Telegram Integration**: Bot interface for external communication
- **Veteran Support**: Specialized endpoints for veteran services
- **Agent Development**: Dynamic agent creation and management

## API Endpoints

### Core Echo Brain API

#### Query Processing
```bash
POST /api/echo/query
# Intelligent query processing with model escalation
{
  "query": "string",
  "context": {
    "user_id": "string",
    "domain": "string",
    "complexity": "low|medium|high|expert"
  },
  "model": "optional_specific_model"
}
```

#### Model Management
```bash
# List all available models
GET /api/echo/models/list

# Manage models (pull/update/remove)
POST /api/echo/models/manage
{
  "action": "pull|remove|update",
  "model": "model_name",
  "user_id": "string",
  "reason": "explanation for operation"
}

# Check operation status
GET /api/echo/models/status/{request_id}
```

#### Conversation & Memory
```bash
# Get conversation history
GET /api/echo/conversations?user_id={user_id}

# Get specific conversation
GET /api/echo/conversation/{conversation_id}

# Analyze thought streams
GET /api/echo/thoughts/{thought_id}
```

#### Real-time Brain Monitoring
```bash
# Get brain activity data
GET /api/echo/brain

# Server-sent events streaming
GET /api/echo/stream

# WebSocket connection
WS /api/echo/brain/ws
```

### Board of Directors API

#### Decision Management
```bash
# Submit decision for board review
POST /api/board/decisions
{
  "decision_type": "model_operation|system_change|policy_update",
  "details": {},
  "user_id": "string"
}

# Get decision status
GET /api/board/decisions/{decision_id}

# Board voting
POST /api/board/decisions/{decision_id}/vote
{
  "director_id": "string",
  "vote": "approve|reject|abstain",
  "reasoning": "string"
}
```

#### Board Analytics
```bash
# Get board performance metrics
GET /api/board/analytics

# Director activity summary
GET /api/board/directors/activity

# Decision audit trail
GET /api/board/audit?start_date={date}&end_date={date}
```

### Testing & Debugging

#### Universal Service Testing
```bash
# Test any Tower service
POST /api/echo/test/{service_name}

# Debug with comprehensive output
POST /api/echo/debug/{service_name}

# Tower status monitoring
GET /api/echo/tower/status
```

#### Performance Benchmarks
```bash
# Run performance tests
POST /api/echo/benchmark
{
  "test_type": "model_performance|decision_speed|memory_usage",
  "iterations": 100
}

# Get benchmark results
GET /api/echo/benchmark/results/{test_id}
```

### Expert Personalities

Available expert agents:
- **Security Expert**: Vulnerability assessment and security analysis
- **Creative Expert**: Design, UX, and creative solutions
- **Technical Expert**: Deep technical implementation guidance
- **Data Analyst**: Performance metrics and data analysis
- **System Architect**: Architecture planning and design
- **Debug Expert**: Error analysis and troubleshooting

```bash
POST /api/echo/expert/{expert_type}
{
  "query": "string",
  "context": {},
  "complexity": "standard|deep"
}
```

## Installation & Setup

### Prerequisites

- Python 3.8+
- PostgreSQL 13+
- Ollama service running
- Redis (for caching)
- NVIDIA GPU (recommended for larger models)

### Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Required environment variables:
POSTGRES_HOST=***REMOVED***
POSTGRES_USER=patrick
POSTGRES_DB=tower_consolidated
OLLAMA_HOST=http://127.0.0.1:11434
REDIS_URL=redis://127.0.0.1:6379
JWT_SECRET_KEY=your-secret-key
BOARD_DECISION_THRESHOLD=3
```

### Database Setup

```bash
# Initialize database schema
python -c "
import psycopg2
from board_database_schema import initialize_board_database
initialize_board_database()
"

# Run database migrations
python migrations/001_initial_setup.py
```

### Virtual Environment Setup

```bash
cd /opt/tower-echo-brain
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Systemd Service Management

### Service Configuration

```ini
# /etc/systemd/system/tower-echo-brain.service
[Unit]
Description=Tower Echo Brain Advanced AI Orchestrator
After=network.target postgresql.service

[Service]
Type=simple
User=patrick
Group=patrick
WorkingDirectory=/opt/tower-echo-brain
Environment=PATH=/opt/tower-echo-brain/venv/bin
ExecStart=/opt/tower-echo-brain/venv/bin/python echo.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Service Commands

```bash
# Start the service
sudo systemctl start tower-echo-brain

# Enable auto-start on boot
sudo systemctl enable tower-echo-brain

# Check service status
sudo systemctl status tower-echo-brain

# View logs
sudo journalctl -u tower-echo-brain -f

# Restart service
sudo systemctl restart tower-echo-brain

# Stop service
sudo systemctl stop tower-echo-brain
```

## Model Management

### Available Models (24+ models, 280GB+ storage)

#### Small Models (1B-3B parameters)
- `qwen2.5:1.5b` - Fast general purpose
- `llama3.2:1b` - Meta's small model
- `gemma2:2b` - Google's efficient model

#### Medium Models (7B-9B parameters)
- `llama3.1:8b` - Balanced performance
- `qwen2.5-coder:7b` - Code generation
- `mistral:7b` - General purpose
- `gemma2:9b` - Advanced reasoning

#### Large Models (30B+ parameters)
- `qwen2.5-coder:32b` - Advanced code analysis
- `mixtral:8x7b` - Expert mixture model
- `llama3.1:70b` - Maximum capability

### Model Operations

```bash
# Pull a new model (requires board approval for 70B+ models)
curl -X POST https://***REMOVED***/api/echo/models/manage \
  -H "Content-Type: application/json" \
  -d '{
    "action": "pull",
    "model": "llama3.1:70b",
    "user_id": "patrick",
    "reason": "Need powerful model for complex reasoning tasks"
  }' -k

# Remove unused model
curl -X POST https://***REMOVED***/api/echo/models/manage \
  -H "Content-Type: application/json" \
  -d '{
    "action": "remove",
    "model": "old-model:7b",
    "user_id": "patrick"
  }' -k

# List all models with metadata
curl https://***REMOVED***/api/echo/models/list -k
```

## Board of Directors System

### Board Members

1. **Aria Chen** - Technical Innovation Director
   - Focus: Performance optimization, system architecture
   - Voting weight: Strategic decisions

2. **Marcus Rodriguez** - Security & Compliance Director
   - Focus: Security assessment, privacy protection
   - Voting weight: Security decisions

3. **Dr. Sarah Kim** - User Experience Director
   - Focus: User satisfaction, interface design
   - Voting weight: UX decisions

4. **James Wilson** - Operations Director
   - Focus: System stability, resource management
   - Voting weight: Operational decisions

5. **Elena Petrov** - Innovation Director
   - Focus: New features, experimental capabilities
   - Voting weight: Innovation decisions

### Decision Types

- **Model Operations**: Requires 3/5 board approval for 70B+ models
- **System Changes**: Architecture modifications
- **Policy Updates**: Service behavior changes
- **Resource Allocation**: Hardware/storage decisions

## Development & Testing

### Running Tests

```bash
# Run all tests
./run_tests.sh

# Run specific test categories
python -m pytest tests/model_tests.py -v
python -m pytest tests/board_tests.py -v
python -m pytest tests/integration_tests.py -v

# Performance benchmarks
python test_performance_benchmarks.py

# Model decision tests
python test_model_decisions.py
```

### Code Quality

```bash
# Pre-commit hooks setup
./setup-pre-commit.sh

# Manual code quality check
python code_quality_validator.py

# Security analysis
python security_fixes.md
```

### Development Workflow

```bash
# Development server (with auto-reload)
uvicorn echo:app --reload --host 0.0.0.0 --port 8309

# Production server
python echo.py

# Background model operations
python model_manager.py --operation pull --model llama3.1:8b
```

## Monitoring & Observability

### Health Checks

```bash
# Service health
curl https://***REMOVED***/api/echo/health -k

# Model availability
curl https://***REMOVED***/api/echo/models/health -k

# Database connectivity
curl https://***REMOVED***/api/echo/db/health -k
```

### Metrics & Analytics

- **Model Performance**: Response times, accuracy metrics
- **Board Activity**: Decision frequency, approval rates
- **Resource Usage**: Memory, CPU, storage utilization
- **Conversation Analytics**: User engagement, query patterns

### Log Analysis

```bash
# Service logs
sudo journalctl -u tower-echo-brain -f

# Model operation logs
tail -f logs/model_operations.log

# Board decision logs
tail -f logs/board_decisions.log

# Performance logs
tail -f logs/performance.log
```

## Integration with Tower Services

### Authentication Integration

Uses Tower's centralized auth service (port 8088) for JWT authentication and user management.

### Database Integration

Shares `tower_consolidated` PostgreSQL database with other Tower services for unified data management.

### Service Communication

- **Knowledge Base**: Automatic article creation for important decisions
- **Anime Production**: Model recommendations for creative tasks
- **Apple Music**: BPM analysis and music intelligence
- **Testing Framework**: Universal service testing capabilities

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check port availability
netstat -tlnp | grep :8309

# Check Python environment
source /opt/tower-echo-brain/venv/bin/activate
python -c "import fastapi; print('FastAPI available')"

# Check database connection
psql -h ***REMOVED*** -U patrick -d tower_consolidated -c "SELECT 1;"
```

#### Model Operations Failing
```bash
# Check Ollama service
systemctl status ollama

# Test Ollama connection
curl http://127.0.0.1:11434/api/tags

# Check disk space (models require significant storage)
df -h
```

#### Board Decisions Stuck
```bash
# Check board database
psql -h ***REMOVED*** -U patrick -d tower_consolidated -c "
SELECT * FROM board_decisions WHERE status = 'pending' ORDER BY created_at DESC LIMIT 10;
"

# Manual board vote (if needed)
python -c "
from board_api import force_board_vote
force_board_vote('decision_id', 'approve', 'Manual intervention')
"
```

### Performance Optimization

#### Memory Management
```bash
# Monitor memory usage
python -c "
from model_manager import get_model_manager
manager = get_model_manager()
print(manager.get_memory_usage())
"

# Clear model cache
curl -X POST https://***REMOVED***/api/echo/models/cache/clear -k
```

#### Database Optimization
```bash
# Analyze query performance
psql -h ***REMOVED*** -U patrick -d tower_consolidated -c "
SELECT query, mean_exec_time, calls FROM pg_stat_statements
WHERE query LIKE '%echo%' ORDER BY mean_exec_time DESC LIMIT 10;
"

# Update statistics
psql -h ***REMOVED*** -U patrick -d tower_consolidated -c "ANALYZE;"
```

## Security Considerations

- **JWT Authentication**: All board endpoints require valid JWT tokens
- **Encryption**: Sensitive data encrypted using Fernet encryption
- **Vault Integration**: Credentials stored in HashiCorp Vault
- **Rate Limiting**: API endpoints protected against abuse
- **Audit Trails**: All board decisions and model operations logged

## Future Enhancements

- **Multi-GPU Support**: Parallel model execution
- **Federated Learning**: Distributed model training
- **Advanced Analytics**: Predictive performance modeling
- **Voice Interface**: Enhanced voice interaction capabilities
- **Mobile Integration**: React Native companion app

---

**Documentation Version**: 1.0
**Last Updated**: September 17, 2025
**Service Version**: Production-ready with comprehensive capabilities
**Maintainer**: Patrick Vestal (patrick.vestal@gmail.com)
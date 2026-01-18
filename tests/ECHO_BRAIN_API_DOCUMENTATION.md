# Echo Brain Complete API & Capability Documentation

## Overview
Echo Brain is an advanced AI system with autonomous capabilities, integrations, and self-improvement features. This document comprehensively catalogs ALL discovered endpoints, capabilities, and integrations.

## Base URL
`http://192.168.50.135:8309`

## Core API Endpoints

### Health & Status Endpoints

#### GET /health
Returns system health status
```json
{
  "status": "healthy",
  "uptime_seconds": 911,
  "version": "1.0.0",
  "service": "echo-brain"
}
```

#### GET /ready
Readiness probe for Kubernetes/orchestration

#### GET /alive
Liveness probe

#### GET /metrics
System metrics and performance data

### Echo Core Endpoints

#### POST /api/echo/chat
Main conversational interface
- **Request:**
```json
{
  "query": "string",
  "user_id": "string",
  "conversation_id": "string"
}
```
- **Response:** QueryResponse with reasoning, intent classification, and execution results

#### POST /api/echo/query
Alternative chat endpoint (same as /api/echo/chat)

#### GET /api/echo/status
Real-time Echo Brain status
```json
{
  "status": "active",
  "persona": "responsive",
  "recent_messages": [...],
  "stats_24h": {
    "conversations": 73,
    "messages": 80
  }
}
```

#### GET /api/echo/brain
Brain architecture and neural network status

#### GET /api/echo/thoughts/recent
Recent processing thoughts and reasoning chains

#### GET /api/echo/thoughts/{thought_id}
Specific thought details

#### GET /api/echo/conversation/{conversation_id}
Retrieve specific conversation history

#### GET /api/echo/conversations
List all conversations

### User Management

#### GET /api/echo/users/{username}
Get user profile and preferences

#### POST /api/echo/users/{username}/preferences
Update user preferences

### Orchestration & Coordination

#### GET /api/coordination/services
Service coordination status

#### GET /api/theater/agents
Agent theater management

#### GET /api/echo/oversight/dashboard
Oversight and monitoring dashboard

### Git & Version Control

#### GET /api/echo/git/status
Git repository status and recent commits

#### GET /api/echo/improvements/history
Self-improvement history

### Delegation & Solutions

#### /api/echo/delegation/*
Task delegation to specialized agents

#### /api/solutions/*
Solution generation and management

#### /api/codebase/*
Codebase analysis and indexing

#### /api/agents/*
Agent management and development

### Knowledge Management

#### /api/knowledge/*
Knowledge base CRUD operations
- GET /api/knowledge/items
- POST /api/knowledge/items
- PUT /api/knowledge/items/{id}
- DELETE /api/knowledge/items/{id}

### Preferences

#### /api/preferences/*
User preference management
- GET /api/preferences/profile
- POST /api/preferences/update
- GET /api/preferences/templates

### Vault Integration

#### /api/vault/*
HashiCorp Vault integration for secrets
- GET /api/vault/get/{path}
- POST /api/vault/set/{path}

### Integrations

#### /api/integrations/*
External service integrations
- GET /api/integrations/status
- POST /api/integrations/configure
- GET /api/integrations/list

### Photo & Vision

#### /api/photo/*
Photo comparison and vision capabilities

### Testing

#### /api/testing/*
System testing endpoints

### Models

#### /api/models/*
AI model management and routing

## Enterprise Integrations (When Available)

### Google Calendar
- **Router:** google_calendar_router
- **Status:** Conditional (google_calendar_available)
- **Endpoints:** Calendar event management, scheduling

### Home Assistant
- **Router:** home_assistant_router
- **Status:** Conditional (home_assistant_available)
- **Endpoints:** Smart home control, device management

### Telegram
- **Multiple Routers:**
  - telegram_router - Basic integration
  - general_telegram_router - General chat
  - telegram_executor_router - Command execution
  - enhanced_telegram_router - Image handling

### Notification System
- **Router:** notification_router
- **Status:** Conditional
- **Channels:** ntfy, Telegram, Email

### Media & Anime
- **Routers:**
  - anime_search_router - Semantic search
  - media_search_router - Media database search
  - anime_integration_router - Character integration
  - semantic_integration_router - Intelligent orchestration

### Database Metrics
- **Router:** db_metrics_router
- **Endpoints:** Connection pool monitoring, query performance

## Registered Capabilities

### 1. autonomous_repair
- **Type:** SELF_MODIFICATION
- **Description:** Autonomous system repair and service restart
- **Actions:**
  - Service restart (tower-* services)
  - System diagnosis
  - Health checks
- **Patterns:**
  - "restart [service]"
  - "fix [service]"
  - "repair [service]"
  - "diagnose system"

### 2. code_analysis
- **Type:** ANALYSIS
- **Description:** Code analysis and improvement suggestions
- **Actions:**
  - Analyze code files
  - Review code quality
  - Suggest improvements
- **Patterns:**
  - "analyze code [file]"
  - "review code [file]"
  - "improve code [file]"

### 3. service_monitoring
- **Type:** ANALYSIS
- **Description:** Monitor Tower services health
- **Actions:**
  - Check service status
  - Monitor health endpoints
  - Report service states
- **Patterns:**
  - "check services status"
  - "service health check"
  - "monitor services"
  - "what services are running"

### 4. send_notification
- **Type:** COMMUNICATION
- **Description:** Send notifications via various channels
- **Actions:**
  - Send ntfy notifications
  - Telegram messages
  - Email alerts
- **Patterns:**
  - "send notification [message]"
  - "notify [message]"
  - "alert [message]"

### 5. image_generation (Coordinator Pattern)
- **Type:** Not formally registered but pattern exists
- **Description:** Image/anime generation via ComfyUI
- **Patterns:**
  - "generate image [prompt]"
  - "create anime [prompt]"
  - "make image [prompt]"

## Capability Types (Enum)

```python
CODE_EXECUTION = "code_execution"
SELF_MODIFICATION = "self_modification"
FINANCIAL = "financial"
TRAINING = "training"
WEB_SCRAPING = "web_scraping"
API_INTEGRATION = "api_integration"
DATABASE = "database"
FILE_SYSTEM = "file_system"
COMMUNICATION = "communication"
ANALYSIS = "analysis"
```

## Database Tables

### echo_unified_interactions
- Stores all conversation interactions
- Fields: conversation_id, query, response, intent, timestamp, user_id

### echo_test_runs
- Test execution history
- Fields: test_run_id, timestamp, total_tests, passed, failed, success_rate

## Authentication & Security

- Most endpoints are open (no auth required)
- Vault endpoints require proper credentials
- Service control capabilities use sudo permissions

## Known Issues

1. **Database Column Issue:** Some tables use 'timestamp' instead of 'created_at'
2. **Service Monitoring:** Returns "Unknown error" occasionally
3. **Integration Endpoints:** Return 404 when integration not configured

## Testing Recommendations

### Critical Tests
1. Service restart verification
2. Database logging consistency
3. Intent classification accuracy
4. Capability execution success rates
5. Conversation memory persistence

### Integration Tests
1. Google Calendar event creation
2. Home Assistant device control
3. Notification delivery
4. Git operations
5. Vault secret management

## Performance Metrics

- Average response time: 0.5-2 seconds
- Capability execution: < 3 seconds
- Database queries: < 100ms
- Model routing: 50-200ms

## Future Enhancements

1. **Planned Capabilities:**
   - DeepSeek reasoning execution
   - LoRA character training
   - Financial integration (Plaid)
   - Docker container management

2. **API Improvements:**
   - GraphQL endpoint
   - WebSocket streaming
   - Batch processing
   - Rate limiting

## Usage Examples

### Restart a Service
```bash
curl -X POST http://localhost:8309/api/echo/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "restart tower-dashboard", "user_id": "admin"}'
```

### Send Notification
```bash
curl -X POST http://localhost:8309/api/echo/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "send notification: System update complete", "user_id": "admin"}'
```

### Check System Status
```bash
curl -X GET http://localhost:8309/api/echo/status
```

## Configuration

- **Port:** 8309
- **Host:** 0.0.0.0
- **Database:** PostgreSQL (echo_brain)
- **Message Queue:** Autonomous loop with priority queue
- **Vector Database:** Qdrant for embeddings

## Monitoring

- Logs: `sudo journalctl -u tower-echo-brain -f`
- Status: `sudo systemctl status tower-echo-brain`
- Metrics: `curl http://localhost:8309/metrics`

---

**Last Updated:** 2026-01-18
**Version:** 1.0.0
**Maintained By:** Echo Brain Development Team
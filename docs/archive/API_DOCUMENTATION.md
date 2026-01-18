# Tower AI Assist - API Documentation

## Overview

The Tower AI Assist system is a comprehensive AI-powered service that consolidates multiple Echo services into a single intelligent router with Board of Directors decision tracking. This documentation covers all service endpoints, authentication requirements, and integration patterns.

**Primary Service**: AI Assist Unified Service (Port 8309)
**Board System**: Board of Directors API with transparent decision tracking
**Personal Context**: Echo Personal Proxy for enhanced user experiences

---

## Table of Contents

1. [Authentication](#authentication)
2. [AI Assist Unified Service API](#echo-brain-unified-service-api)
3. [Board of Directors API](#board-of-directors-api)
4. [Echo Personal Proxy API](#echo-personal-proxy-api)
5. [WebSocket Endpoints](#websocket-endpoints)
6. [Error Handling](#error-handling)
7. [Example Usage](#example-usage)

---

## Authentication

### JWT Authentication

All Board API endpoints require JWT authentication. The system integrates with Tower's centralized auth service.

**Auth Service URL**: `http://192.168.50.135:8088`

#### Required Headers
```http
Authorization: Bearer <JWT_TOKEN>
```

#### JWT Token Structure
```json
{
  "user_id": "string",
  "username": "string",
  "roles": ["array", "of", "roles"],
  "permissions": ["array", "of", "permissions"],
  "exp": 1234567890
}
```

#### Permission Levels

| Permission | Description | Required Roles |
|------------|-------------|----------------|
| `board.submit_task` | Submit tasks to the board | `board_user`, `board_contributor` |
| `board.view_decisions` | View decision details | `board_user`, `board_viewer`, `board_contributor` |
| `board.provide_feedback` | Provide feedback on decisions | `board_contributor`, `board_reviewer` |
| `board.override_decisions` | Override board decisions | `board_admin`, `system_admin` |

#### Security Features
- 30-minute token expiration
- Strict role-based access control
- No fallback authentication (secure by default)
- WebSocket authentication via token query parameter

---

## AI Assist Unified Service API

**Base URL**: `http://localhost:8309/api/echo`

### Core Intelligence Endpoints

#### 1. Health Check
**GET** `/api/echo/health`

Returns system health status and capabilities.

**Response:**
```json
{
  "status": "healthy",
  "service": "AI Assist Unified",
  "intelligence_levels": ["tinyllama", "llama3.2", "mistral", "qwen2.5-coder", "llama3.1"],
  "specialized_models": [],
  "max_parameters": "70B",
  "timestamp": "2025-01-15T10:30:00Z",
  "board_system": {
    "status": "enabled",
    "decision_tracking": true,
    "user_preferences": true,
    "knowledge_management": true,
    "sandbox_execution": true
  }
}
```

#### 2. Process Query
**POST** `/api/echo/query`

Main conversational query processing with dynamic intelligence escalation.

**Request Body:**
```json
{
  "query": "string",
  "context": {},
  "intelligence_level": "auto",
  "user_id": "default",
  "conversation_id": "optional-uuid"
}
```

**Response:**
```json
{
  "response": "AI-generated response",
  "model_used": "llama3.1",
  "intelligence_level": "advanced",
  "processing_time": 2.5,
  "escalation_path": ["tinyllama", "llama3.2", "llama3.1"],
  "requires_clarification": false,
  "clarifying_questions": [],
  "conversation_id": "uuid",
  "intent": "technical_assistance",
  "confidence": 0.95
}
```

**Example:**
```bash
curl -X POST http://localhost:8309/api/echo/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I implement a REST API with FastAPI?",
    "intelligence_level": "auto",
    "user_id": "developer_123"
  }'
```

### Brain Visualization Endpoints

#### 3. Brain Activity
**GET** `/api/echo/brain`

Get current brain visualization state with neural activity.

**Response:**
```json
{
  "brain_visualization": {
    "active_regions": ["reasoning", "language", "technical"],
    "neural_firing_rate": 0.75,
    "cognitive_load": 0.6
  },
  "thought_history_count": 150,
  "active_neurons": 42,
  "service": "AI Assist Neural Visualization",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

#### 4. Thought Stream
**GET** `/api/echo/thoughts/{thought_id}`

Get detailed thought stream for a specific thought process.

**Response:**
```json
{
  "thought_id": "uuid",
  "thought_stream": [
    {
      "timestamp": "2025-01-15T10:30:00Z",
      "process": "initial_analysis",
      "content": "Analyzing user query..."
    }
  ],
  "neuron_count": 25,
  "service": "AI Assist Thought Visualization"
}
```

### Conversation Management

#### 5. Get Conversation
**GET** `/api/echo/conversation/{conversation_id}`

Retrieve conversation history and context.

**Response:**
```json
{
  "conversation_id": "uuid",
  "history": [
    {
      "user_query": "How do I deploy FastAPI?",
      "ai_response": "Here's how to deploy FastAPI...",
      "intent": "deployment_help",
      "timestamp": "2025-01-15T10:30:00Z"
    }
  ],
  "last_intent": "deployment_help",
  "interaction_count": 5
}
```

#### 6. List Conversations
**GET** `/api/echo/conversations?user_id=default&limit=10`

List recent conversations for a user.

**Response:**
```json
{
  "conversations": [
    {
      "conversation_id": "uuid",
      "created_at": "2025-01-15T10:30:00Z",
      "last_intent": "technical_assistance",
      "interaction_count": 3,
      "last_query": "How do I implement authentication..."
    }
  ]
}
```

### Execution and Testing

#### 7. Execute Task
**POST** `/api/echo/execute`

Execute a task based on conversational understanding (CI/CD integration).

**Request Body:**
```json
{
  "conversation_id": "uuid",
  "safe_mode": true
}
```

**Response:**
```json
{
  "conversation_id": "uuid",
  "intent": "code_modification",
  "status": "ready_for_execution",
  "steps": [
    "Analyze existing code structure",
    "Identify files to modify",
    "Implement changes with proper error handling",
    "Run tests to verify functionality",
    "Commit changes with descriptive message"
  ]
}
```

#### 8. Universal Testing
**POST** `/api/echo/test/{target}`

Run comprehensive testing on any target service.

**Request Body:**
```json
{
  "target": "comfyui",
  "test_type": "universal",
  "user_id": "default"
}
```

**Response:**
```json
{
  "success": true,
  "target": "comfyui",
  "test_results": {
    "connectivity": "pass",
    "response_time": 150,
    "status_code": 200,
    "json_valid": true
  },
  "service": "AI Assist Testing Framework",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

#### 9. Debug Service
**POST** `/api/echo/debug/{service}`

Debug a specific Tower service with comprehensive analysis.

**Response:**
```json
{
  "success": true,
  "debug_target": "comfyui",
  "network_analysis": {
    "port_status": "open",
    "connectivity": "reachable",
    "latency_ms": 25
  },
  "resource_analysis": {
    "cpu_usage": "12%",
    "memory_usage": "2.1GB",
    "disk_usage": "45%"
  },
  "service": "AI Assist Debug Tools",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Statistics and Analytics

#### 10. Usage Statistics
**GET** `/api/echo/stats`

Get usage statistics and model performance metrics.

**Response:**
```json
{
  "statistics": [
    {
      "model": "llama3.1",
      "usage_count": 150,
      "avg_processing_time": 2.3
    },
    {
      "model": "tinyllama",
      "usage_count": 500,
      "avg_processing_time": 0.8
    }
  ],
  "total_queries": 650
}
```

### Streaming Endpoints

#### 11. Stream Brain Activity
**GET** `/api/echo/stream`

Server-Sent Events stream of real-time brain activity.

**Response (SSE Stream):**
```
data: {"timestamp": "2025-01-15T10:30:00Z", "brain_state": {...}, "service": "AI Assist Stream"}

data: {"timestamp": "2025-01-15T10:30:01Z", "brain_state": {...}, "service": "AI Assist Stream"}
```

#### 12. Stream Query Processing
**POST** `/api/echo/stream-query`

Process query with real-time streaming of thought process.

**Request Body:**
```json
{
  "query": "Explain machine learning",
  "intelligence_level": "auto"
}
```

**Response (SSE Stream):**
```
data: {"type": "start", "conversation_id": "uuid", "query": "Explain machine learning"}

data: {"type": "intent", "intent": "educational", "confidence": 0.9}

data: {"type": "thinking_start", "thought_id": "uuid"}

data: {"type": "brain_activity", "brain_state": {...}}

data: {"type": "response", "response": "Machine learning is...", "model_used": "llama3.1"}

data: {"type": "complete"}
```

### Voice Integration

#### 13. Voice Notification
**POST** `/api/echo/voice/notify`

Send voice notification using unified voice service.

**Request Body:**
```json
{
  "message": "Task completed successfully",
  "character": "echo_default",
  "tone": "helpful",
  "priority": "normal"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Voice notification sent",
  "character": "echo_default",
  "processing_time": 0.5
}
```

#### 14. Voice Status Update
**POST** `/api/echo/voice/status`

Send voice status update for service changes.

**Request Body:**
```json
{
  "service_name": "comfyui",
  "status": "running",
  "details": "All systems operational"
}
```

#### 15. Voice Characters
**GET** `/api/echo/voice/characters`

Get available voice characters.

**Response:**
```json
{
  "success": true,
  "characters": ["echo_default", "tokyo_debt_desire", "sakura"],
  "voice_service_url": "http://127.0.0.1:8331"
}
```

### Tower Integration

#### 16. Tower Status
**GET** `/api/echo/tower/status`

Get comprehensive Tower services status.

**Response:**
```json
{
  "success": true,
  "command": "status",
  "output": "tower-dashboard: active\ntower-auth: active\n...",
  "processing_time": 1.2,
  "service": "AI Assist Tower Status",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

#### 17. Tower Health
**GET** `/api/echo/tower/health`

Get health status of all Tower services.

#### 18. Tower Commands
**POST** `/api/echo/tower/{command}`

Run any Tower framework command.

**Request Body:**
```json
{
  "args": ["--verbose"]
}
```

#### 19. Testing Capabilities
**GET** `/api/echo/testing/capabilities`

Get information about testing framework capabilities.

**Response:**
```json
{
  "framework_name": "Tower Testing Framework",
  "version": "1.0.0",
  "capabilities": {
    "universal_testing": {
      "description": "Comprehensive testing suite for any Tower service",
      "endpoint": "/api/echo/test/{target}",
      "supports": ["HTTP endpoints", "Network connectivity", "JSON responses"]
    },
    "debug_analysis": {
      "description": "Advanced debugging capabilities",
      "endpoint": "/api/echo/debug/{service}",
      "supports": ["Network debugging", "System resources", "Log analysis"]
    }
  },
  "services": {
    "dashboard": "8080",
    "anime": "8300",
    "agent-manager": "8301",
    "loan-search": "8302",
    "crypto-trader": "8303",
    "echo": "8309"
  }
}
```

---

## Board of Directors API

**Base URL**: `http://localhost:8309/api/board`

The Board of Directors system provides transparent AI decision tracking with user oversight capabilities.

### Task Management

#### 1. Submit Task
**POST** `/api/board/task`

**Authentication**: Required (`board.submit_task` permission)

Submit a task for board evaluation with transparent decision tracking.

**Request Body:**
```json
{
  "task_description": "Deploy new microservice to production",
  "user_id": "user_123",
  "priority": "high",
  "context": {
    "service_name": "payment-processor",
    "environment": "production"
  },
  "expected_completion_time": "2025-01-15T12:00:00Z"
}
```

**Response:**
```json
{
  "task_id": "uuid",
  "status": "submitted",
  "message": "Task submitted for board evaluation",
  "estimated_completion_time": "2025-01-15T10:35:00Z"
}
```

**Example:**
```bash
curl -X POST http://localhost:8309/api/board/task \
  -H "Authorization: Bearer ${JWT_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Optimize database queries for user dashboard",
    "priority": "normal",
    "context": {
      "database": "postgresql",
      "tables_affected": ["users", "dashboards", "analytics"]
    }
  }'
```

#### 2. Get Decision Details
**GET** `/api/board/decisions/{task_id}`

**Authentication**: Optional

Get detailed decision information for a specific task.

**Query Parameters:**
- `include_evidence`: boolean (default: true)
- `include_timeline`: boolean (default: true)
- `include_director_details`: boolean (default: true)

**Response:**
```json
{
  "task_id": "uuid",
  "current_status": "completed",
  "consensus_score": 0.85,
  "confidence_score": 0.92,
  "director_count": 5,
  "evidence_count": 12,
  "timeline": [
    {
      "id": "uuid",
      "timestamp": "2025-01-15T10:30:00Z",
      "status": "in_progress",
      "description": "Security Director reviewing task",
      "consensus_score": 0.7,
      "confidence": 0.8,
      "director_count": 3
    }
  ],
  "director_evaluations": [
    {
      "director_id": "security_director",
      "director_name": "Security Director",
      "recommendation": "approve",
      "confidence": 0.9,
      "risk_score": 0.2,
      "processing_time": 1.5,
      "timestamp": "2025-01-15T10:30:00Z",
      "reasoning": "Task follows security best practices",
      "evidence": [
        {
          "id": "uuid",
          "type": "security_scan",
          "source": "automated_scanner",
          "weight": 0.8,
          "confidence": 0.95,
          "reasoning": "No security vulnerabilities detected"
        }
      ]
    }
  ],
  "user_override_available": true,
  "estimated_completion": "2025-01-15T10:35:00Z"
}
```

#### 3. Submit User Feedback
**POST** `/api/board/feedback/{task_id}`

**Authentication**: Required (`board.provide_feedback` permission)

Submit user feedback or override for a task decision.

**Request Body:**
```json
{
  "feedback_type": "approve",
  "feedback_content": "Looks good to proceed",
  "override_recommendation": "approve_with_conditions",
  "reasoning": "Add additional monitoring during deployment"
}
```

**Response:**
```json
{
  "feedback_id": "uuid",
  "status": "accepted",
  "message": "Feedback recorded successfully",
  "decision_updated": true
}
```

### Board Status and Analytics

#### 4. Board Status
**GET** `/api/board/status`

**Authentication**: Optional

Get current board status and metrics.

**Response:**
```json
{
  "active_tasks": 3,
  "completed_tasks_today": 15,
  "average_consensus_score": 0.87,
  "average_processing_time": 2.3,
  "director_status": {
    "security_director": {
      "name": "Security Director",
      "specialization": "Security Analysis",
      "status": "active",
      "current_load": 2,
      "response_time": 1.5
    },
    "quality_director": {
      "name": "Quality Assurance Director",
      "specialization": "Quality Control",
      "status": "active",
      "current_load": 1,
      "response_time": 2.1
    }
  },
  "system_health": "healthy"
}
```

#### 5. Directors List
**GET** `/api/board/directors`

**Authentication**: Optional

Get list of all directors with their details.

**Response:**
```json
{
  "directors": [
    {
      "director_id": "security_director",
      "director_name": "Security Director",
      "specialization": "Security Analysis",
      "status": "active",
      "current_load": 2,
      "average_response_time": 1.5,
      "approval_rate": 0.85,
      "last_activity": "2025-01-15T10:30:00Z"
    }
  ],
  "total_count": 5,
  "active_count": 5
}
```

#### 6. Board Analytics
**GET** `/api/board/analytics?start_date=2025-01-01&end_date=2025-01-15`

**Authentication**: Optional

Get board analytics for specified date range.

**Response:**
```json
{
  "overall_stats": {
    "completed_tasks": 125,
    "avg_consensus": 0.87,
    "avg_processing_time": 2.1,
    "success_rate": 0.95
  },
  "director_performance": {
    "security_director": {
      "tasks_evaluated": 45,
      "avg_response_time": 1.5,
      "approval_rate": 0.85
    }
  },
  "time_series": [
    {
      "date": "2025-01-15",
      "tasks_completed": 8,
      "avg_consensus": 0.89
    }
  ]
}
```

#### 7. Board Health
**GET** `/api/board/health`

**Authentication**: None

Health check endpoint for board system.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00Z",
  "active_connections": 3,
  "active_tasks": 2
}
```

---

## Echo Personal Proxy API

**Base URL**: `http://localhost:8313/api`

The Personal Proxy enhances Echo responses with Patrick's personal context and preferences.

### Enhanced Query Processing

#### 1. Personal Query Processing
**POST** `/api/echo/query`

Process query with personal context enhancement.

**Request Body:**
```json
{
  "query": "What should I use for my next API project?",
  "intelligence_level": "professional",
  "conversation_id": "uuid"
}
```

**Response:**
```json
{
  "response": "Based on your expertise with FastAPI and Python, I recommend...",
  "model_used": "llama3.1",
  "intelligence_level": "professional",
  "processing_time": 2.1,
  "escalation_path": ["tinyllama", "llama3.1"],
  "personal_context": "enhanced",
  "conversation_id": "uuid"
}
```

#### 2. Personal Health Check
**GET** `/api/echo/health`

Health check with personal context status.

**Response:**
```json
{
  "status": "healthy",
  "service": "AI Assist Unified",
  "personal_context": "enabled",
  "patrick_profile": "loaded",
  "enhanced_responses": true,
  "intelligence_levels": ["tinyllama", "llama3.2", "mistral", "qwen2.5-coder", "llama3.1"]
}
```

#### 3. Get Personal Profile
**GET** `/api/personal/profile`

Get Patrick's personal profile and preferences.

**Response:**
```json
{
  "name": "Patrick",
  "role": "Senior Full-Stack Developer",
  "expertise": ["Python", "FastAPI", "Docker", "Kubernetes", "PostgreSQL", "React"],
  "current_projects": ["Tower AI System", "Financial Automation", "DevOps Optimization"],
  "communication_style": "Direct, technical, solution-focused",
  "preferences": {
    "response_style": "Technical detail matching senior developer expertise",
    "explanation_depth": "Assume advanced technical knowledge",
    "code_examples": "Always include when relevant",
    "communication": "Minimal small talk, direct solutions"
  }
}
```

---

## WebSocket Endpoints

### Board Real-Time Updates

#### Board WebSocket Connection
**WebSocket** `/api/board/ws?token={JWT_TOKEN}`

**Authentication**: Required (JWT token in query parameter)

Real-time updates for board activities and decisions.

**Connection Example:**
```javascript
const ws = new WebSocket('ws://localhost:8309/api/board/ws?token=' + jwtToken);

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Board update:', data);
};
```

**Message Types:**

1. **Welcome Message**
```json
{
  "type": "welcome",
  "user_id": "user_123",
  "message": "Connected to AI Assist Board",
  "permissions": ["board.submit_task", "board.view_decisions"]
}
```

2. **Task Submitted**
```json
{
  "type": "task_submitted",
  "task_id": "uuid",
  "user_id": "user_123",
  "description": "Deploy new microservice",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

3. **Director Started**
```json
{
  "type": "director_started",
  "task_id": "uuid",
  "director_id": "security_director",
  "director_name": "Security Director",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

4. **Director Completed**
```json
{
  "type": "director_completed",
  "task_id": "uuid",
  "director_id": "security_director",
  "director_name": "Security Director",
  "recommendation": "approve",
  "confidence": 0.9,
  "timestamp": "2025-01-15T10:30:00Z"
}
```

5. **Task Completed**
```json
{
  "type": "task_completed",
  "task_id": "uuid",
  "final_recommendation": "approve",
  "director_count": 5,
  "timestamp": "2025-01-15T10:30:00Z"
}
```

6. **User Override**
```json
{
  "type": "user_override",
  "task_id": "uuid",
  "override_type": "modify",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

7. **Error Handling**
```json
{
  "type": "error",
  "message": "Processing failed",
  "task_id": "uuid",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

**Client Commands:**

1. **Ping/Pong**
```json
// Send
{"type": "ping"}

// Receive
{"type": "pong"}
```

2. **Subscribe to Task**
```json
// Send
{"type": "subscribe", "task_id": "uuid"}

// Receive
{"type": "subscribed", "task_id": "uuid"}
```

---

## Error Handling

### HTTP Status Codes

| Code | Description | Example Response |
|------|-------------|------------------|
| 200 | Success | Standard successful response |
| 400 | Bad Request | Missing required parameters |
| 401 | Unauthorized | Invalid or missing JWT token |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Task or resource not found |
| 429 | Rate Limited | Too many requests |
| 500 | Internal Error | Server processing error |
| 503 | Service Unavailable | Dependent service down |

### Error Response Format

```json
{
  "detail": "Error description",
  "request_id": "uuid",
  "type": "error_type",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Common Error Examples

#### Authentication Error
```json
{
  "detail": "Invalid or expired token",
  "request_id": "uuid",
  "type": "authentication_error",
  "headers": {"WWW-Authenticate": "Bearer"}
}
```

#### Permission Error
```json
{
  "detail": "Insufficient permissions. Required: 'board.submit_task'",
  "request_id": "uuid",
  "type": "permission_error"
}
```

#### Task Not Found
```json
{
  "detail": "Task not found",
  "request_id": "uuid",
  "type": "not_found_error",
  "task_id": "uuid"
}
```

#### Service Unavailable
```json
{
  "detail": "Echo service unavailable",
  "request_id": "uuid",
  "type": "service_error",
  "service": "echo_brain"
}
```

---

## Example Usage

### Complete Workflow Example

#### 1. Submit Task to Board
```bash
# Submit a task for board evaluation
curl -X POST http://localhost:8309/api/board/task \
  -H "Authorization: Bearer ${JWT_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Implement new authentication system",
    "priority": "high",
    "context": {
      "technology": "FastAPI + JWT",
      "security_requirements": "enterprise"
    }
  }'
```

**Response:**
```json
{
  "task_id": "auth-task-123",
  "status": "submitted",
  "message": "Task submitted for board evaluation"
}
```

#### 2. Monitor Board Decision (WebSocket)
```javascript
const ws = new WebSocket('ws://localhost:8309/api/board/ws?token=' + jwtToken);

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);

    if (data.type === 'director_completed' && data.task_id === 'auth-task-123') {
        console.log(`${data.director_name}: ${data.recommendation} (${data.confidence})`);
    }

    if (data.type === 'task_completed' && data.task_id === 'auth-task-123') {
        console.log(`Final decision: ${data.final_recommendation}`);
    }
};
```

#### 3. Get Decision Details
```bash
curl -X GET "http://localhost:8309/api/board/decisions/auth-task-123" \
  -H "Authorization: Bearer ${JWT_TOKEN}"
```

#### 4. Process with AI Assist
```bash
curl -X POST http://localhost:8309/api/echo/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Generate FastAPI authentication code based on the approved task",
    "context": {
      "task_id": "auth-task-123",
      "board_approved": true
    }
  }'
```

### Personal Context Example

#### Enhanced Query with Personal Context
```bash
# Query through personal proxy
curl -X POST http://localhost:8313/api/echo/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What database should I use for my next project?",
    "intelligence_level": "professional"
  }'
```

**Enhanced Response:**
```json
{
  "response": "Based on your expertise with PostgreSQL and your current work on the Tower AI System, I recommend PostgreSQL for your next project. Given your DevOps optimization focus, consider...",
  "model_used": "llama3.1",
  "personal_context": "enhanced"
}
```

### Streaming Example

#### Real-Time Query Processing
```javascript
const eventSource = new EventSource('http://localhost:8309/api/echo/stream-query', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        query: "Explain microservices architecture",
        intelligence_level: "auto"
    })
});

eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);

    switch(data.type) {
        case 'start':
            console.log('Processing started:', data.conversation_id);
            break;
        case 'intent':
            console.log('Intent detected:', data.intent, 'Confidence:', data.confidence);
            break;
        case 'thinking_start':
            console.log('AI thinking process started');
            break;
        case 'brain_activity':
            console.log('Brain activity:', data.brain_state);
            break;
        case 'response':
            console.log('Response:', data.response);
            break;
        case 'complete':
            console.log('Processing complete');
            eventSource.close();
            break;
    }
};
```

### Testing Framework Example

#### Test Service Health
```bash
curl -X POST http://localhost:8309/api/echo/test/comfyui \
  -H "Content-Type: application/json" \
  -d '{
    "target": "comfyui",
    "test_type": "universal"
  }'
```

#### Debug Service Issues
```bash
curl -X POST http://localhost:8309/api/echo/debug/anime-production \
  -H "Content-Type: application/json"
```

### Voice Integration Example

#### Send Voice Notification
```bash
curl -X POST http://localhost:8309/api/echo/voice/notify \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Deployment completed successfully",
    "character": "echo_default",
    "tone": "helpful",
    "priority": "normal"
  }'
```

---

## Integration Patterns

### Tower Service Integration
- All services accessible via nginx proxy at `https://192.168.50.135/`
- Unified authentication with Tower auth service (port 8088)
- PostgreSQL database integration for persistence
- Redis caching for session management

### Service Dependencies
- **AI Assist** → Board API (decision tracking)
- **Board API** → Auth Service (JWT validation)
- **Personal Proxy** → AI Assist (enhanced responses)
- **Voice Service** → External TTS (synthesis)
- **Testing Framework** → All Tower services (monitoring)

### Database Schema
- `echo_unified_interactions`: Query logs and analytics
- `board_*`: Board decision tracking tables
- `conversation_history`: Chat history and context
- `user_preferences`: Personal settings and preferences

### Security Considerations
- JWT tokens expire in 30 minutes
- Role-based access control for board operations
- Safe shell execution with command filtering
- Rate limiting on all endpoints
- Audit logging for board decisions
- No fallback authentication (secure by default)

---

## Performance and Monitoring

### Response Time Targets
- Health checks: < 100ms
- Simple queries: < 1s (tinyllama)
- Complex queries: < 5s (llama3.1)
- Board decisions: < 30s
- WebSocket connections: < 50 concurrent

### Monitoring Endpoints
- `/api/echo/health` - Service health
- `/api/board/health` - Board system health
- `/api/echo/stats` - Usage statistics
- `/api/echo/tower/status` - All Tower services

### Logging
- All interactions logged to PostgreSQL
- Board decisions audit trail
- Error tracking with request IDs
- Performance metrics collection

---

This documentation provides comprehensive coverage of all Tower AI Assist service endpoints, authentication requirements, and integration patterns. The system supports dynamic intelligence scaling, transparent AI decision tracking, and real-time monitoring capabilities.
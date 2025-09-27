# AI Assist Quick Start Guide & Developer Tutorials

## Overview

Welcome to AI Assist - the Advanced AI Orchestrator with 24+ models, Board of Directors governance, and comprehensive testing capabilities. This guide will get you from zero to productive in under 15 minutes.

## Table of Contents

1. [5-Minute Quick Start](#5-minute-quick-start)
2. [Authentication Setup](#authentication-setup)
3. [Core Tutorials](#core-tutorials)
4. [Advanced Use Cases](#advanced-use-cases)
5. [Integration Examples](#integration-examples)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## 5-Minute Quick Start

### Step 1: Verify System Access (30 seconds)

```bash
# Test basic connectivity
curl -k https://***REMOVED***/api/echo/health

# Expected response:
{
  "status": "healthy",
  "service": "AI Assist Unified",
  "intelligence_levels": ["tinyllama", "llama3.2", "mistral", "qwen2.5-coder", "llama3.1"],
  "specialized_models": ["qwen2.5-coder:32b", "deepseek-coder-v2:16b"],
  "max_parameters": "70B",
  "timestamp": "2025-01-15T10:30:00Z",
  "board_system": {
    "status": "enabled",
    "decision_tracking": true
  }
}
```

### Step 2: Your First AI Query (1 minute)

```bash
# Simple query without authentication
curl -X POST https://***REMOVED***/api/echo/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain what AI Assist is in one sentence",
    "intelligence_level": "auto"
  }' | jq .

# Expected response includes:
{
  "response": "AI Assist is an advanced AI orchestrator...",
  "model_used": "llama3.1:8b",
  "intelligence_level": "intermediate",
  "processing_time": 2.3,
  "escalation_path": ["tinyllama", "llama3.1:8b"],
  "conversation_id": "uuid-1234"
}
```

### Step 3: Explore Available Models (1 minute)

```bash
# List all available AI models
curl https://***REMOVED***/api/echo/models/list | jq -r '.models[] | "\(.name) - \(.size) - \(.specialization)"'

# Example output:
# llama3.1:70b - 40GB - General Intelligence
# qwen2.5-coder:32b - 18GB - Code Generation
# mistral:7b - 4GB - Creative Writing
# tinyllama:1b - 600MB - Quick Responses
```

### Step 4: Test Service Integration (1 minute)

```bash
# Test integration with other Tower services
curl -X POST https://***REMOVED***/api/echo/test/comfyui \
  -H "Content-Type: application/json" \
  -d '{"target": "comfyui", "test_type": "universal"}' | jq .

# Check Tower services status
curl https://***REMOVED***/api/echo/tower/status | jq .
```

### Step 5: Monitor Real-time Brain Activity (1.5 minutes)

```bash
# Connect to brain activity stream (requires wscat: npm install -g wscat)
wscat -c wss://***REMOVED***/api/echo/stream

# Or use curl for Server-Sent Events
curl -N https://***REMOVED***/api/echo/stream
```

🎉 **Congratulations!** You've successfully:
- ✅ Connected to AI Assist
- ✅ Made your first AI query with automatic model selection
- ✅ Explored available models (1B to 70B parameters)
- ✅ Tested service integrations
- ✅ Connected to real-time monitoring

---

## Authentication Setup

For protected endpoints (Board of Directors, Model Management), you'll need authentication.

### Get Your JWT Token

```bash
# Step 1: Login to Tower Auth Service
curl -X POST https://***REMOVED***/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your_username",
    "password": "your_password"
  }'

# Response includes your JWT token:
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}

# Step 2: Set your token as environment variable
export JWT_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# Step 3: Test authenticated endpoint
curl -H "Authorization: Bearer $JWT_TOKEN" \
  https://***REMOVED***/api/echo/board/status
```

### Token Management Script

```bash
#!/bin/bash
# save as: ~/bin/echo-auth.sh

# Tower Auth Helper Script
AUTH_URL="https://***REMOVED***/api/auth"
TOKEN_FILE="$HOME/.echo-token"

login() {
    echo "Enter Tower credentials:"
    read -p "Username: " username
    read -s -p "Password: " password
    echo

    response=$(curl -s -X POST "$AUTH_URL/login" \
        -H "Content-Type: application/json" \
        -d "{\"username\": \"$username\", \"password\": \"$password\"}")

    token=$(echo "$response" | jq -r '.access_token // empty')

    if [ -n "$token" ] && [ "$token" != "null" ]; then
        echo "$token" > "$TOKEN_FILE"
        chmod 600 "$TOKEN_FILE"
        echo "✓ Login successful! Token saved to $TOKEN_FILE"
        echo "Usage: export JWT_TOKEN=\$(cat $TOKEN_FILE)"
    else
        echo "✗ Login failed: $(echo "$response" | jq -r '.detail // "Unknown error"')"
        exit 1
    fi
}

check() {
    if [ ! -f "$TOKEN_FILE" ]; then
        echo "No token file found. Run: $0 login"
        exit 1
    fi

    token=$(cat "$TOKEN_FILE")
    response=$(curl -s -H "Authorization: Bearer $token" "$AUTH_URL/verify")

    if echo "$response" | jq -e '.valid' >/dev/null 2>&1; then
        exp=$(echo "$response" | jq -r '.expires_at')
        echo "✓ Token is valid until: $exp"
        echo "export JWT_TOKEN=\"$token\""
    else
        echo "✗ Token is invalid or expired"
        rm -f "$TOKEN_FILE"
        echo "Run: $0 login"
        exit 1
    fi
}

case "$1" in
    login) login ;;
    check) check ;;
    *) echo "Usage: $0 {login|check}" ;;
esac

# Make executable: chmod +x ~/bin/echo-auth.sh
# Usage:
# ./echo-auth.sh login
# eval $(./echo-auth.sh check)
```

---

## Core Tutorials

### Tutorial 1: AI Query Processing with Intelligence Escalation

Learn how AI Assist automatically selects the best model for your query.

```bash
# Simple query - will use lightweight model
curl -X POST https://***REMOVED***/api/echo/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is 2+2?",
    "intelligence_level": "auto"
  }' | jq '.model_used, .escalation_path'

# Complex query - will escalate to powerful model
curl -X POST https://***REMOVED***/api/echo/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Design a microservices architecture for a high-throughput financial trading system with real-time risk management",
    "intelligence_level": "auto"
  }' | jq '.model_used, .escalation_path, .processing_time'

# Force specific intelligence level
curl -X POST https://***REMOVED***/api/echo/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain quantum computing",
    "intelligence_level": "genius",
    "context": {"domain": "physics", "audience": "experts"}
  }' | jq .
```

**Key Concepts:**
- **Intelligence Levels**: `basic`, `intermediate`, `advanced`, `expert`, `genius`, `auto`
- **Automatic Escalation**: System tries smaller models first, escalates if needed
- **Context Awareness**: Additional context improves response quality

### Tutorial 2: Conversation Management

AI Assist maintains conversation context across multiple queries.

```bash
# Start a conversation
response1=$(curl -s -X POST https://***REMOVED***/api/echo/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I need help designing a REST API for user management",
    "user_id": "developer_123"
  }')

# Extract conversation ID
conversation_id=$(echo "$response1" | jq -r '.conversation_id')
echo "Conversation ID: $conversation_id"

# Continue the conversation with context
curl -X POST https://***REMOVED***/api/echo/query \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"What authentication method would you recommend?\",
    \"conversation_id\": \"$conversation_id\",
    \"user_id\": \"developer_123\"
  }" | jq '.response'

# Get conversation history
curl "https://***REMOVED***/api/echo/conversation/$conversation_id" | jq .

# List all conversations for a user
curl "https://***REMOVED***/api/echo/conversations?user_id=developer_123&limit=10" | jq .
```

### Tutorial 3: Model Management with Board Approval

For large models (70B+ parameters), the Board of Directors system provides governance.

```bash
# List current models
curl https://***REMOVED***/api/echo/models/list | jq '.models[] | {name, size, parameters}'

# Request to install a large model (requires authentication)
curl -X POST https://***REMOVED***/api/echo/models/manage \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "action": "pull",
    "model": "llama3.1:70b",
    "user_id": "your_username",
    "reason": "Need advanced reasoning capabilities for complex analysis tasks"
  }' | jq .

# Monitor the board decision process
# (Replace TASK_ID with the actual task_id from above response)
curl https://***REMOVED***/api/echo/board/decisions/TASK_ID | jq .

# Connect to WebSocket for real-time board updates
wscat -c "wss://***REMOVED***/api/echo/board/ws?token=$JWT_TOKEN"
```

### Tutorial 4: Universal Testing Framework

AI Assist can test any service in the Tower ecosystem.

```bash
# Test ComfyUI image generation service
curl -X POST https://***REMOVED***/api/echo/test/comfyui \
  -H "Content-Type: application/json" \
  -d '{
    "target": "comfyui",
    "test_type": "universal",
    "comprehensive": true
  }' | jq .

# Debug a service with detailed analysis
curl -X POST https://***REMOVED***/api/echo/debug/anime-production \
  -H "Content-Type: application/json" | jq .

# Test multiple services in batch
services=("comfyui" "anime-production" "agent-manager" "loan-search")
for service in "${services[@]}"; do
    echo "Testing $service..."
    curl -s -X POST https://***REMOVED***/api/echo/test/$service \
      -H "Content-Type: application/json" \
      -d "{\"target\": \"$service\"}" | jq -r '.success, .test_results.connectivity'
done
```

### Tutorial 5: Real-time Monitoring & Brain Visualization

Monitor AI decision-making in real-time.

```bash
# Get current brain state
curl https://***REMOVED***/api/echo/brain | jq .

# Stream real-time brain activity
curl -N https://***REMOVED***/api/echo/stream | head -20

# Get detailed thought process for a specific query
# First, make a query and note the thought_id
response=$(curl -s -X POST https://***REMOVED***/api/echo/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze the performance implications of microservices"}')

thought_id=$(echo "$response" | jq -r '.thought_id // empty')
if [ -n "$thought_id" ]; then
    curl "https://***REMOVED***/api/echo/thoughts/$thought_id" | jq .
fi
```

### Tutorial 6: Voice Integration

AI Assist supports voice notifications for hands-free operation.

```bash
# Send a voice notification
curl -X POST https://***REMOVED***/api/echo/voice/notify \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Your deployment has completed successfully",
    "character": "echo_default",
    "tone": "helpful",
    "priority": "normal"
  }' | jq .

# Get available voice characters
curl https://***REMOVED***/api/echo/voice/characters | jq .

# Send urgent notification
curl -X POST https://***REMOVED***/api/echo/voice/notify \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Critical system alert: database connection lost",
    "character": "echo_default",
    "tone": "urgent",
    "priority": "urgent"
  }'
```

---

## Advanced Use Cases

### Use Case 1: AI-Powered DevOps Automation

Integrate AI Assist into your DevOps workflow for intelligent automation.

```bash
#!/bin/bash
# ai-devops-helper.sh - Intelligent DevOps Assistant

# Function to analyze deployment logs with AI
analyze_deployment() {
    local service="$1"
    local log_file="$2"

    echo "Analyzing deployment logs for $service..."

    # Get recent logs
    logs=$(tail -100 "$log_file" | jq -Rs .)

    # Send to AI Assist for analysis
    analysis=$(curl -s -X POST https://***REMOVED***/api/echo/query \
        -H "Content-Type: application/json" \
        -d "{
            \"query\": \"Analyze these deployment logs and identify any issues or recommendations: $logs\",
            \"intelligence_level\": \"expert\",
            \"context\": {
                \"service\": \"$service\",
                \"task\": \"log_analysis\",
                \"domain\": \"devops\"
            }
        }")

    echo "$analysis" | jq -r '.response'

    # Check if intervention is needed
    confidence=$(echo "$analysis" | jq -r '.confidence')
    if (( $(echo "$confidence < 0.7" | bc -l) )); then
        echo "Low confidence analysis - escalating to human review"
        # Send alert
        curl -X POST https://***REMOVED***/api/echo/voice/notify \
            -H "Content-Type: application/json" \
            -d '{
                "message": "Deployment analysis requires human review",
                "priority": "high"
            }'
    fi
}

# Function to get intelligent scaling recommendations
get_scaling_recommendations() {
    local service="$1"

    # Test service performance
    perf_data=$(curl -s -X POST https://***REMOVED***/api/echo/test/$service \
        -H "Content-Type: application/json" \
        -d '{"target": "'$service'", "test_type": "performance"}')

    # Get AI recommendations
    recommendation=$(curl -s -X POST https://***REMOVED***/api/echo/query \
        -H "Content-Type: application/json" \
        -d "{
            \"query\": \"Based on this performance data, provide scaling recommendations: $(echo "$perf_data" | jq -c .)\",
            \"intelligence_level\": \"expert\",
            \"context\": {
                \"service\": \"$service\",
                \"task\": \"performance_optimization\",
                \"domain\": \"infrastructure\"
            }
        }")

    echo "$recommendation" | jq -r '.response'
}

# Usage examples:
# analyze_deployment "api-service" "/var/log/deployment.log"
# get_scaling_recommendations "web-frontend"
```

### Use Case 2: Intelligent Code Review Assistant

Use AI Assist for AI-powered code review and suggestions.

```python
#!/usr/bin/env python3
# ai-code-reviewer.py

import requests
import json
import subprocess
import sys

class EchoBrainCodeReviewer:
    def __init__(self, base_url="https://***REMOVED***/api/echo"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.verify = False  # For self-signed certificates

    def review_diff(self, diff_content, language="python"):
        """Review git diff with AI analysis"""
        payload = {
            "query": f"""
            Please review this {language} code diff and provide:
            1. Potential bugs or issues
            2. Performance improvements
            3. Security concerns
            4. Best practice recommendations
            5. Code quality score (1-10)

            Diff content:
            {diff_content}
            """,
            "intelligence_level": "expert",
            "context": {
                "task": "code_review",
                "language": language,
                "domain": "software_engineering"
            }
        }

        response = self.session.post(
            f"{self.base_url}/query",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API error: {response.status_code}")

    def analyze_complexity(self, file_path):
        """Analyze code complexity and suggest improvements"""
        try:
            with open(file_path, 'r') as f:
                code_content = f.read()

            payload = {
                "query": f"""
                Analyze the complexity of this code and suggest improvements:
                1. Cyclomatic complexity assessment
                2. Maintainability score
                3. Refactoring suggestions
                4. Design pattern recommendations

                Code:
                {code_content}
                """,
                "intelligence_level": "expert",
                "context": {
                    "task": "complexity_analysis",
                    "file": file_path
                }
            }

            response = self.session.post(f"{self.base_url}/query", json=payload)
            return response.json()

        except Exception as e:
            return {"error": str(e)}

    def suggest_tests(self, code_content, test_framework="pytest"):
        """Generate test suggestions for code"""
        payload = {
            "query": f"""
            Generate comprehensive test cases for this code using {test_framework}:
            1. Unit tests for each function
            2. Edge cases and error conditions
            3. Integration test scenarios
            4. Performance test considerations

            Code to test:
            {code_content}
            """,
            "intelligence_level": "expert",
            "context": {
                "task": "test_generation",
                "framework": test_framework
            }
        }

        response = self.session.post(f"{self.base_url}/query", json=payload)
        return response.json()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 ai-code-reviewer.py <command> [args]")
        print("Commands:")
        print("  review-diff           # Review git diff")
        print("  analyze <file>        # Analyze file complexity")
        print("  suggest-tests <file>  # Generate test suggestions")
        return

    reviewer = EchoBrainCodeReviewer()
    command = sys.argv[1]

    if command == "review-diff":
        # Get git diff
        diff_output = subprocess.check_output(
            ["git", "diff", "--cached"],
            universal_newlines=True
        )

        if not diff_output.strip():
            print("No staged changes to review")
            return

        result = reviewer.review_diff(diff_output)
        print("AI Code Review:")
        print("=" * 50)
        print(result.get('response', 'No response'))
        print(f"\nModel used: {result.get('model_used')}")
        print(f"Processing time: {result.get('processing_time')}s")

    elif command == "analyze" and len(sys.argv) > 2:
        file_path = sys.argv[2]
        result = reviewer.analyze_complexity(file_path)
        print(f"Complexity Analysis for {file_path}:")
        print("=" * 50)
        print(result.get('response', result.get('error', 'No response')))

    elif command == "suggest-tests" and len(sys.argv) > 2:
        file_path = sys.argv[2]
        with open(file_path, 'r') as f:
            code = f.read()

        result = reviewer.suggest_tests(code)
        print(f"Test Suggestions for {file_path}:")
        print("=" * 50)
        print(result.get('response', 'No response'))

if __name__ == "__main__":
    main()
```

### Use Case 3: Intelligent Monitoring & Alerting

Create smart monitoring that uses AI to analyze patterns and reduce false alarms.

```bash
#!/bin/bash
# intelligent-monitor.sh

# Configuration
ECHO_BRAIN_URL="https://***REMOVED***/api/echo"
SERVICES=("comfyui" "anime-production" "agent-manager" "loan-search")
ALERT_THRESHOLD=3
CONTEXT_WINDOW="1h"

# Function to get service metrics
get_service_metrics() {
    local service="$1"

    # Test service health
    health_data=$(curl -s -X POST "$ECHO_BRAIN_URL/test/$service" \
        -H "Content-Type: application/json" \
        -d '{"target": "'$service'", "test_type": "universal"}')

    # Get debug information if health check fails
    if ! echo "$health_data" | jq -e '.success' >/dev/null 2>&1; then
        debug_data=$(curl -s -X POST "$ECHO_BRAIN_URL/debug/$service" \
            -H "Content-Type: application/json")
        echo "$debug_data"
    else
        echo "$health_data"
    fi
}

# Function to analyze metrics with AI
analyze_metrics() {
    local service="$1"
    local metrics="$2"
    local historical_data="$3"

    analysis=$(curl -s -X POST "$ECHO_BRAIN_URL/query" \
        -H "Content-Type: application/json" \
        -d "{
            \"query\": \"Analyze these service metrics and determine if there are any issues requiring attention. Consider historical patterns to avoid false alarms. Current metrics: $metrics. Historical data: $historical_data\",
            \"intelligence_level\": \"expert\",
            \"context\": {
                \"service\": \"$service\",
                \"task\": \"anomaly_detection\",
                \"domain\": \"monitoring\"
            }
        }")

    echo "$analysis"
}

# Function to get intelligent recommendations
get_recommendations() {
    local service="$1"
    local issue_analysis="$2"

    recommendations=$(curl -s -X POST "$ECHO_BRAIN_URL/query" \
        -H "Content-Type: application/json" \
        -d "{
            \"query\": \"Based on this issue analysis, provide specific actionable recommendations for resolving the problem: $issue_analysis\",
            \"intelligence_level\": \"expert\",
            \"context\": {
                \"service\": \"$service\",
                \"task\": \"incident_response\",
                \"domain\": \"devops\"
            }
        }")

    echo "$recommendations" | jq -r '.response'
}

# Main monitoring loop
main() {
    echo "Starting intelligent monitoring at $(date)"

    for service in "${SERVICES[@]}"; do
        echo "Monitoring $service..."

        # Get current metrics
        current_metrics=$(get_service_metrics "$service")

        # Get historical data (simplified - in practice, would come from time-series DB)
        historical_data=$(curl -s "$ECHO_BRAIN_URL/stats" | jq -c .)

        # Analyze with AI
        analysis=$(analyze_metrics "$service" "$current_metrics" "$historical_data")

        # Check if issue detected
        confidence=$(echo "$analysis" | jq -r '.confidence // 0')
        requires_action=$(echo "$analysis" | jq -r '.response' | grep -i "issue\|problem\|alert" >/dev/null && echo "yes" || echo "no")

        if [ "$requires_action" = "yes" ] && (( $(echo "$confidence > 0.8" | bc -l) )); then
            echo "⚠️  Issue detected for $service"

            # Get recommendations
            recommendations=$(get_recommendations "$service" "$(echo "$analysis" | jq -r '.response')")

            # Send intelligent alert
            alert_message="Service: $service\nIssue: $(echo "$analysis" | jq -r '.response')\nRecommendations: $recommendations"

            curl -X POST "$ECHO_BRAIN_URL/voice/notify" \
                -H "Content-Type: application/json" \
                -d "{
                    \"message\": \"Intelligent monitoring detected an issue with $service. Check logs for details.\",
                    \"priority\": \"high\"
                }"

            echo "Alert sent for $service"
            echo "Analysis: $(echo "$analysis" | jq -r '.response')"
            echo "Recommendations: $recommendations"
            echo "---"
        else
            echo "✅ $service is healthy"
        fi
    done
}

# Run monitoring
if [ "$1" = "--daemon" ]; then
    # Run as daemon (every 5 minutes)
    while true; do
        main
        sleep 300
    done
else
    # Run once
    main
fi
```

---

## Integration Examples

### Python Client Library

```python
# echo_brain_client.py - Python client library for AI Assist
import requests
import json
import time
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import websocket
import threading

@dataclass
class QueryResponse:
    response: str
    model_used: str
    intelligence_level: str
    processing_time: float
    escalation_path: List[str]
    conversation_id: str
    intent: Optional[str] = None
    confidence: float = 0.0

@dataclass
class ModelInfo:
    name: str
    tag: str
    size: str
    parameters: Optional[int] = None
    specialization: Optional[str] = None
    status: str = "unknown"

class EchoBrainClient:
    """Python client for AI Assist API"""

    def __init__(self, base_url="https://***REMOVED***/api/echo", verify_ssl=False):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.verify = verify_ssl
        self.jwt_token = None

    def set_auth_token(self, token: str):
        """Set JWT authentication token"""
        self.jwt_token = token
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def health_check(self) -> Dict:
        """Get system health status"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def query(self,
              text: str,
              intelligence_level: str = "auto",
              context: Optional[Dict] = None,
              user_id: str = "default",
              conversation_id: Optional[str] = None) -> QueryResponse:
        """Send query to AI Assist AI"""

        payload = {
            "query": text,
            "intelligence_level": intelligence_level,
            "user_id": user_id,
            "context": context or {}
        }

        if conversation_id:
            payload["conversation_id"] = conversation_id

        response = self.session.post(
            f"{self.base_url}/query",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()

        return QueryResponse(
            response=data["response"],
            model_used=data["model_used"],
            intelligence_level=data["intelligence_level"],
            processing_time=data["processing_time"],
            escalation_path=data["escalation_path"],
            conversation_id=data["conversation_id"],
            intent=data.get("intent"),
            confidence=data.get("confidence", 0.0)
        )

    def list_models(self) -> List[ModelInfo]:
        """Get list of available AI models"""
        response = self.session.get(f"{self.base_url}/models/list")
        response.raise_for_status()
        data = response.json()

        return [
            ModelInfo(
                name=model["name"],
                tag=model["tag"],
                size=model["size"],
                parameters=model.get("parameters"),
                specialization=model.get("specialization"),
                status=model.get("status", "unknown")
            )
            for model in data["models"]
        ]

    def test_service(self, service_name: str, test_type: str = "universal") -> Dict:
        """Test another Tower service"""
        payload = {
            "target": service_name,
            "test_type": test_type
        }

        response = self.session.post(
            f"{self.base_url}/test/{service_name}",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def get_brain_activity(self) -> Dict:
        """Get current brain visualization data"""
        response = self.session.get(f"{self.base_url}/brain")
        response.raise_for_status()
        return response.json()

    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history for a user"""
        params = {"user_id": user_id, "limit": limit}
        response = self.session.get(f"{self.base_url}/conversations", params=params)
        response.raise_for_status()
        return response.json()["conversations"]

    def send_voice_notification(self,
                              message: str,
                              character: str = "echo_default",
                              tone: str = "helpful",
                              priority: str = "normal") -> Dict:
        """Send voice notification"""
        payload = {
            "message": message,
            "character": character,
            "tone": tone,
            "priority": priority
        }

        response = self.session.post(
            f"{self.base_url}/voice/notify",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def stream_brain_activity(self, callback):
        """Stream real-time brain activity via WebSocket"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                callback(data)
            except json.JSONDecodeError:
                print(f"Invalid JSON received: {message}")

        def on_error(ws, error):
            print(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            print("WebSocket connection closed")

        ws_url = self.base_url.replace("https://", "wss://").replace("http://", "ws://")
        ws = websocket.WebSocketApp(
            f"{ws_url}/stream",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        # Run in separate thread
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

        return ws

# Usage examples
if __name__ == "__main__":
    # Initialize client
    client = EchoBrainClient()

    # Health check
    print("Health check:", client.health_check()["status"])

    # Simple query
    response = client.query("What is machine learning?")
    print(f"Response: {response.response}")
    print(f"Model used: {response.model_used}")
    print(f"Processing time: {response.processing_time}s")

    # List available models
    models = client.list_models()
    print(f"Available models: {len(models)}")
    for model in models[:3]:  # Show first 3
        print(f"  {model.name} - {model.size} - {model.specialization}")

    # Test service
    test_result = client.test_service("comfyui")
    print(f"ComfyUI test: {'✓' if test_result['success'] else '✗'}")

    # Stream brain activity (example)
    def brain_callback(data):
        if "brain_state" in data:
            print(f"Brain activity: {data['brain_state'].get('neural_firing_rate', 'N/A')}")

    # Uncomment to test streaming:
    # ws = client.stream_brain_activity(brain_callback)
    # time.sleep(10)  # Stream for 10 seconds
    # ws.close()
```

### JavaScript/Node.js Integration

```javascript
// echo-brain-client.js - Node.js client for AI Assist
const axios = require('axios');
const WebSocket = require('ws');
const https = require('https');

class EchoBrainClient {
    constructor(baseUrl = 'https://***REMOVED***/api/echo', options = {}) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.jwtToken = null;

        // Create axios instance with SSL verification disabled for self-signed certs
        this.client = axios.create({
            httpsAgent: new https.Agent({
                rejectUnauthorized: options.verifySSL || false
            }),
            timeout: options.timeout || 30000
        });

        // Request interceptor to add auth header
        this.client.interceptors.request.use(config => {
            if (this.jwtToken) {
                config.headers.Authorization = `Bearer ${this.jwtToken}`;
            }
            return config;
        });
    }

    setAuthToken(token) {
        this.jwtToken = token;
    }

    async healthCheck() {
        const response = await this.client.get(`${this.baseUrl}/health`);
        return response.data;
    }

    async query(text, options = {}) {
        const payload = {
            query: text,
            intelligence_level: options.intelligenceLevel || 'auto',
            user_id: options.userId || 'default',
            context: options.context || {},
            ...options.conversationId && { conversation_id: options.conversationId }
        };

        const response = await this.client.post(`${this.baseUrl}/query`, payload);
        return response.data;
    }

    async listModels() {
        const response = await this.client.get(`${this.baseUrl}/models/list`);
        return response.data.models;
    }

    async testService(serviceName, testType = 'universal') {
        const payload = { target: serviceName, test_type: testType };
        const response = await this.client.post(`${this.baseUrl}/test/${serviceName}`, payload);
        return response.data;
    }

    async getBrainActivity() {
        const response = await this.client.get(`${this.baseUrl}/brain`);
        return response.data;
    }

    async getConversationHistory(userId, limit = 10) {
        const response = await this.client.get(`${this.baseUrl}/conversations`, {
            params: { user_id: userId, limit }
        });
        return response.data.conversations;
    }

    async sendVoiceNotification(message, options = {}) {
        const payload = {
            message,
            character: options.character || 'echo_default',
            tone: options.tone || 'helpful',
            priority: options.priority || 'normal'
        };

        const response = await this.client.post(`${this.baseUrl}/voice/notify`, payload);
        return response.data;
    }

    streamBrainActivity(callback) {
        const wsUrl = this.baseUrl
            .replace('https://', 'wss://')
            .replace('http://', 'ws://');

        const ws = new WebSocket(`${wsUrl}/stream`);

        ws.on('open', () => {
            console.log('Connected to brain activity stream');
        });

        ws.on('message', (data) => {
            try {
                const parsed = JSON.parse(data);
                callback(parsed);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        });

        ws.on('error', (error) => {
            console.error('WebSocket error:', error);
        });

        ws.on('close', () => {
            console.log('Brain activity stream closed');
        });

        return ws;
    }

    // Convenience method for conversational AI
    async chat(message, conversationId = null) {
        const response = await this.query(message, { conversationId });
        return {
            message: response.response,
            conversationId: response.conversation_id,
            model: response.model_used,
            processingTime: response.processing_time
        };
    }

    // Batch testing of multiple services
    async testMultipleServices(services) {
        const results = await Promise.allSettled(
            services.map(service => this.testService(service))
        );

        return results.map((result, index) => ({
            service: services[index],
            success: result.status === 'fulfilled',
            data: result.status === 'fulfilled' ? result.value : null,
            error: result.status === 'rejected' ? result.reason.message : null
        }));
    }
}

// Usage examples
async function examples() {
    const client = new EchoBrainClient();

    try {
        // Health check
        const health = await client.healthCheck();
        console.log('System status:', health.status);

        // Simple chat
        const chat1 = await client.chat('Hello, what can you help me with?');
        console.log('AI:', chat1.message);
        console.log('Model used:', chat1.model);

        // Continue conversation
        const chat2 = await client.chat(
            'Can you explain quantum computing?',
            chat1.conversationId
        );
        console.log('AI:', chat2.message);

        // List models
        const models = await client.listModels();
        console.log(`Available models: ${models.length}`);
        models.slice(0, 3).forEach(model => {
            console.log(`  ${model.name} - ${model.size}`);
        });

        // Test multiple services
        const testResults = await client.testMultipleServices([
            'comfyui', 'anime-production', 'agent-manager'
        ]);
        testResults.forEach(result => {
            console.log(`${result.service}: ${result.success ? '✓' : '✗'}`);
        });

        // Stream brain activity for 10 seconds
        const ws = client.streamBrainActivity((data) => {
            if (data.brain_state) {
                console.log('Neural activity:', data.brain_state.neural_firing_rate);
            }
        });

        setTimeout(() => {
            ws.close();
        }, 10000);

    } catch (error) {
        console.error('Error:', error.message);
    }
}

module.exports = EchoBrainClient;

// Run examples if called directly
if (require.main === module) {
    examples();
}
```

---

## Best Practices

### 1. Query Optimization

**Do's:**
```bash
# Use appropriate intelligence levels
curl -X POST https://***REMOVED***/api/echo/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Simple calculation: 2+2",
    "intelligence_level": "basic"
  }'

# Provide context for better responses
curl -X POST https://***REMOVED***/api/echo/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Optimize this database query",
    "intelligence_level": "expert",
    "context": {
      "database": "postgresql",
      "table_size": "10M rows",
      "current_performance": "slow"
    }
  }'
```

**Don'ts:**
```bash
# Don't use genius level for simple tasks
curl -X POST https://***REMOVED***/api/echo/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What time is it?",
    "intelligence_level": "genius"  # Overkill
  }'
```

### 2. Error Handling

```python
# Python example with proper error handling
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_resilient_session():
    session = requests.Session()

    # Retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

def safe_query(query_text, max_retries=3):
    session = create_resilient_session()

    for attempt in range(max_retries):
        try:
            response = session.post(
                "https://***REMOVED***/api/echo/query",
                json={"query": query_text},
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limited
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                response.raise_for_status()

        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                raise
        except requests.exceptions.ConnectionError:
            print(f"Connection error on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                raise

    raise Exception("Max retries exceeded")
```

### 3. Authentication Management

```bash
# Create auth helper script
cat > ~/.echo-auth << 'EOF'
#!/bin/bash
# AI Assist Authentication Helper

TOKEN_FILE="$HOME/.echo-token"
AUTH_URL="https://***REMOVED***/api/auth"

get_token() {
    if [ -f "$TOKEN_FILE" ]; then
        # Check if token is still valid
        token=$(cat "$TOKEN_FILE")
        if curl -s -H "Authorization: Bearer $token" "$AUTH_URL/verify" | jq -e '.valid' >/dev/null 2>&1; then
            echo "$token"
            return 0
        fi
    fi

    # Token missing or invalid - need to re-authenticate
    echo "Authentication required" >&2
    return 1
}

export_token() {
    token=$(get_token)
    if [ $? -eq 0 ]; then
        echo "export JWT_TOKEN='$token'"
    else
        echo "echo 'Please run: echo-auth login'" >&2
        return 1
    fi
}

case "$1" in
    get) get_token ;;
    export) export_token ;;
    *) echo "Usage: $0 {get|export}" ;;
esac
EOF

chmod +x ~/.echo-auth

# Usage:
# eval $(~/.echo-auth export)
```

### 4. Performance Optimization

```bash
# Use appropriate models for tasks
curl -X POST https://***REMOVED***/api/echo/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Quick yes/no answer: Is Python object-oriented?",
    "intelligence_level": "basic"  # Fast response
  }'

# Cache conversation IDs for context
conversation_id=$(curl -s -X POST https://***REMOVED***/api/echo/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Start project discussion"}' | jq -r '.conversation_id')

# Reuse conversation ID for related queries
curl -X POST https://***REMOVED***/api/echo/query \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"What technologies should we use?\",
    \"conversation_id\": \"$conversation_id\"
  }"
```

### 5. Monitoring Integration

```bash
# Add health checks to monitoring
cat >> /etc/prometheus/prometheus.yml << 'EOF'
  - job_name: 'echo-brain'
    static_configs:
      - targets: ['***REMOVED***:8309']
    metrics_path: /api/echo/metrics
    scrape_interval: 30s
EOF

# Create alerting rules
cat > /etc/prometheus/rules/echo-brain.yml << 'EOF'
groups:
  - name: echo-brain.rules
    rules:
      - alert: EchoBrainDown
        expr: up{job="echo-brain"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "AI Assist is down"

      - alert: SlowQueries
        expr: echo_processing_time_seconds{quantile="0.95"} > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "AI Assist queries are slow"
EOF
```

---

## Troubleshooting

### Quick Fixes for Common Issues

**Issue: Connection Refused**
```bash
# Check if service is running
sudo systemctl status tower-echo-brain

# Check port availability
curl -v http://***REMOVED***:8309/api/echo/health
```

**Issue: Authentication Errors**
```bash
# Verify token
echo "$JWT_TOKEN" | cut -d. -f2 | base64 -d | jq .

# Get new token
curl -X POST https://***REMOVED***/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your_user", "password": "your_pass"}'
```

**Issue: Slow Responses**
```bash
# Check system resources
htop
nvidia-smi  # If using GPU

# Test with basic intelligence level
curl -X POST https://***REMOVED***/api/echo/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "intelligence_level": "basic"}'
```

**Issue: Model Not Available**
```bash
# List available models
curl https://***REMOVED***/api/echo/models/list

# Check Ollama directly
curl http://localhost:11434/api/tags
```

### Getting Help

1. **Check Service Logs:**
   ```bash
   sudo journalctl -u tower-echo-brain -f
   ```

2. **System Health:**
   ```bash
   curl https://***REMOVED***/api/echo/health | jq .
   ```

3. **Test Connectivity:**
   ```bash
   curl -X POST https://***REMOVED***/api/echo/test/echo-brain
   ```

4. **Documentation:**
   - API Documentation: [Swagger UI](./swagger-ui.html)
   - Troubleshooting: [Troubleshooting Playbook](./troubleshooting-playbook.md)
   - Integration Patterns: [Tower Integration](./tower-integration-patterns.md)

---

## Next Steps

Now that you've completed the quick start guide:

1. **Explore the API:** Use the [Interactive API Documentation](./swagger-ui.html)
2. **Read User Journeys:** Check [User Journey Maps](./user-journey-maps.md) for your persona
3. **Learn Integration:** Review [Tower Integration Patterns](./tower-integration-patterns.md)
4. **Set Up Monitoring:** Implement health checks and alerting
5. **Join the Community:** Access the [Knowledge Base](https://***REMOVED***/kb/) for updates

**Happy building with AI Assist! 🧠✨**
#!/bin/bash
"""
Start all CI/CD Pipeline Services
Starts all components of the Echo CI/CD pipeline in the background
"""

VENV_PATH="/opt/tower-echo-brain/venv"
BASE_DIR="/opt/tower-echo-brain"

echo "🚀 Starting Echo CI/CD Pipeline Services"
echo "=========================================="

# Function to start a service
start_service() {
    local service_name=$1
    local script_name=$2
    local port=$3
    
    echo "Starting $service_name on port $port..."
    
    # Kill existing process if running
    pkill -f "$script_name" 2>/dev/null || true
    
    # Start service in background
    cd $BASE_DIR
    nohup $VENV_PATH/bin/python $script_name > logs/${service_name}.log 2>&1 &
    
    # Store PID
    echo $! > logs/${service_name}.pid
    
    sleep 2
    
    # Check if service started
    if curl -s http://localhost:$port/api/health >/dev/null 2>&1; then
        echo "✅ $service_name started successfully"
    else
        echo "❌ $service_name failed to start - check logs/${service_name}.log"
    fi
}

# Create logs directory
mkdir -p logs

echo
echo "Starting individual services..."

# Start Test Generator
start_service "test-generator" "echo_test_generator.py" 8340

# Start Deployment Manager
start_service "deployment-manager" "echo_deployment_manager.py" 8341

# Start Pipeline Orchestrator
start_service "pipeline-orchestrator" "echo_ci_cd_pipeline.py" 8342

# Start GitHub Integration
start_service "github-integration" "echo_github_integration.py" 8343

# Start Learning Integration
start_service "learning-integration" "echo_cicd_learning_integration.py" 8344

echo
echo "=========================================="
echo "📊 CI/CD Pipeline Services Status"
echo "=========================================="

# Check all services
services=(
    "test-generator:8340"
    "deployment-manager:8341"
    "pipeline-orchestrator:8342"
    "github-integration:8343"
    "learning-integration:8344"
)

for service_port in "${services[@]}"; do
    IFS=':' read -r service port <<< "$service_port"
    if curl -s http://localhost:$port/api/health >/dev/null 2>&1; then
        echo "✅ $service (:$port): Running"
    else
        echo "❌ $service (:$port): Not responding"
    fi
done

echo
echo "🎯 CI/CD Pipeline startup complete!"
echo "📁 Logs available in: $BASE_DIR/logs/"
echo "🧪 Run tests with: ./venv/bin/python test_cicd_pipeline.py"

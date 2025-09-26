#!/bin/bash
# Echo MLOps Deployment Script
# ===========================

set -e  # Exit on any error

# Configuration
ECHO_BRAIN_DIR="/opt/tower-echo-brain"
MLOPS_DATA_DIR="$ECHO_BRAIN_DIR/data"
MLOPS_LOGS_DIR="$ECHO_BRAIN_DIR/logs"
VENV_DIR="$ECHO_BRAIN_DIR/venv"
USER="patrick"
GROUP="patrick"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    log "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

# Create directory with proper permissions
create_directory() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        log_info "Creating directory: $dir"
        mkdir -p "$dir"
        chown $USER:$GROUP "$dir"
        chmod 755 "$dir"
    else
        log_info "Directory already exists: $dir"
    fi
}

# Install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Check if virtual environment exists
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv "$VENV_DIR"
        chown -R $USER:$GROUP "$VENV_DIR"
    fi
    
    # Create requirements file
    cat > "$ECHO_BRAIN_DIR/mlops_requirements.txt" << 'DEPS'
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
mlflow>=2.0.0
optuna>=3.0.0
fastapi>=0.95.0
uvicorn>=0.20.0
aiofiles>=23.0.0
sqlalchemy>=1.4.0
redis>=4.0.0
pyarrow>=10.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
asyncpg>=0.27.0
aiohttp>=3.8.0
pydantic>=1.10.0
statsmodels>=0.13.0
pyyaml>=6.0
joblib>=1.1.0
jinja2>=3.0.0
DEPS
    
    # Install dependencies
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r "$ECHO_BRAIN_DIR/mlops_requirements.txt"
    
    log_success "Python dependencies installed"
}

# Setup databases
setup_databases() {
    log_info "Setting up MLOps databases..."
    
    create_directory "$MLOPS_DATA_DIR"
    
    # Test database initialization
    source "$VENV_DIR/bin/activate"
    cd "$ECHO_BRAIN_DIR"
    
    python3 -c "
import sqlite3
import os

# Create test databases
db_files = [
    'model_registry.db',
    'ab_testing.db', 
    'drift_monitoring.db',
    'retraining_pipeline.db',
    'feature_store.db'
]

data_dir = '$MLOPS_DATA_DIR'
for db_file in db_files:
    db_path = os.path.join(data_dir, db_file)
    conn = sqlite3.connect(db_path)
    conn.execute('CREATE TABLE IF NOT EXISTS test (id INTEGER)')
    conn.close()
    print(f'Created {db_file}')
"
    
    chown -R $USER:$GROUP "$MLOPS_DATA_DIR"
    log_success "Databases initialized"
}

# Create monitoring script
create_monitoring_script() {
    log_info "Creating monitoring script..."
    
    cat > "$ECHO_BRAIN_DIR/monitor_mlops.py" << 'PYEOF'
#!/usr/bin/env python3
import asyncio
import aiohttp
import sys
from datetime import datetime

SERVICES = [
    {"name": "Model Registry", "url": "http://localhost:8340", "port": 8340},
    {"name": "A/B Testing", "url": "http://localhost:8341", "port": 8341},
    {"name": "Drift Detection", "url": "http://localhost:8342", "port": 8342},
    {"name": "Retraining Pipeline", "url": "http://localhost:8343", "port": 8343},
    {"name": "Feature Store", "url": "http://localhost:8344", "port": 8344},
    {"name": "MLOps Integration", "url": "http://localhost:8345", "port": 8345},
]

async def check_service(session, service):
    try:
        async with session.get(service["url"], timeout=5) as response:
            if response.status == 200:
                return {"name": service["name"], "status": "healthy", "port": service["port"]}
            else:
                return {"name": service["name"], "status": "unhealthy", "port": service["port"]}
    except Exception as e:
        return {"name": service["name"], "status": "unreachable", "port": service["port"]}

async def main():
    print(f"Echo MLOps Health Check - {datetime.now().isoformat()}")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        tasks = [check_service(session, service) for service in SERVICES]
        results = await asyncio.gather(*tasks)
    
    healthy_count = 0
    for result in results:
        status_symbol = "✅" if result["status"] == "healthy" else "❌"
        print(f"{status_symbol} {result['name']} (:{result['port']}) - {result['status']}")
        if result["status"] == "healthy":
            healthy_count += 1
    
    print("=" * 60)
    print(f"Services: {healthy_count}/{len(SERVICES)} healthy")
    
    if healthy_count < len(SERVICES):
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
PYEOF
    
    chmod +x "$ECHO_BRAIN_DIR/monitor_mlops.py"
    chown $USER:$GROUP "$ECHO_BRAIN_DIR/monitor_mlops.py"
    
    log_success "Monitoring script created"
}

# Create startup script
create_startup_script() {
    log_info "Creating startup script..."
    
    cat > "$ECHO_BRAIN_DIR/start_mlops.sh" << 'STARTEOF'
#!/bin/bash
# Start all MLOps services

ECHO_BRAIN_DIR="/opt/tower-echo-brain"
VENV_DIR="$ECHO_BRAIN_DIR/venv"

cd "$ECHO_BRAIN_DIR"
source "$VENV_DIR/bin/activate"

echo "Starting Echo MLOps services..."

# Start services in background
python3 echo_model_registry.py &
sleep 2
python3 echo_ab_testing.py &
sleep 2  
python3 echo_drift_detector.py &
sleep 2
python3 echo_retraining_pipeline.py &
sleep 2
python3 echo_feature_store.py &
sleep 2
python3 echo_mlops_integration.py &

echo "All MLOps services started"
echo "Monitor with: python3 monitor_mlops.py"
STARTEOF
    
    chmod +x "$ECHO_BRAIN_DIR/start_mlops.sh"
    chown $USER:$GROUP "$ECHO_BRAIN_DIR/start_mlops.sh"
    
    log_success "Startup script created"
}

# Create configuration file
create_config() {
    log_info "Creating MLOps configuration..."
    
    cat > "$ECHO_BRAIN_DIR/mlops_config.yaml" << 'YAMLEOF'
# Echo MLOps Configuration
mlops:
  drift_monitoring:
    enabled: true
    threshold: 0.1
    window_days: 7
    
  auto_retraining:
    enabled: true
    performance_threshold: 0.05
    schedule: "weekly"
    
  ab_testing:
    enabled: true
    default_allocation: "thompson_sampling"
    
  feature_store:
    enabled: true
    cache_ttl: 3600
    
  model_registry:
    backup_enabled: true
    versioning: true

services:
  model_registry:
    port: 8340
    host: "0.0.0.0"
    
  ab_testing:
    port: 8341
    host: "0.0.0.0"
    
  drift_detector:
    port: 8342
    host: "0.0.0.0"
    
  retraining_pipeline:
    port: 8343
    host: "0.0.0.0"
    
  feature_store:
    port: 8344
    host: "0.0.0.0"
    
  mlops_integration:
    port: 8345
    host: "0.0.0.0"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "/opt/tower-echo-brain/logs/mlops.log"
YAMLEOF
    
    chown $USER:$GROUP "$ECHO_BRAIN_DIR/mlops_config.yaml"
    log_success "Configuration created"
}

# Main deployment function
main() {
    log_info "Starting Echo MLOps deployment..."
    
    # Check if Echo Brain directory exists
    if [ ! -d "$ECHO_BRAIN_DIR" ]; then
        log_error "Echo Brain directory not found: $ECHO_BRAIN_DIR"
        exit 1
    fi
    
    # Create necessary directories
    create_directory "$MLOPS_DATA_DIR"
    create_directory "$MLOPS_LOGS_DIR"
    
    # Install dependencies
    install_dependencies
    
    # Setup databases
    setup_databases
    
    # Create scripts and configs
    create_monitoring_script
    create_startup_script
    create_config
    
    log_success "Echo MLOps deployment completed successfully!"
    echo
    echo "🎉 MLOps Services ready to start:"
    echo "   • Start all services: ./start_mlops.sh"
    echo "   • Monitor services:   python3 monitor_mlops.py"
    echo
    echo "📊 Service URLs (after starting):"
    echo "   • Model Registry:      http://192.168.50.135:8340/docs"
    echo "   • A/B Testing:         http://192.168.50.135:8341/docs"
    echo "   • Drift Detection:     http://192.168.50.135:8342/docs"
    echo "   • Retraining Pipeline: http://192.168.50.135:8343/docs"
    echo "   • Feature Store:       http://192.168.50.135:8344/docs"
    echo "   • MLOps Integration:   http://192.168.50.135:8345/docs"
}

# Run main function
main "$@"

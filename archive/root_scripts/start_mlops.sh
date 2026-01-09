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

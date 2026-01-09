#!/bin/bash

echo "Starting Echo Brain Video Generation Service..."

# Kill any existing process on port 8309
fuser -k 8309/tcp 2>/dev/null

# Start the new service with user-writable log
cd /opt/tower-echo-brain
nohup python3 echo_brain_video_enhanced.py > /tmp/echo_brain_video.log 2>&1 &
PID=$!

echo "Echo Brain Video started on port 8309 (PID: $PID)"
echo "Logs: /tmp/echo_brain_video.log"

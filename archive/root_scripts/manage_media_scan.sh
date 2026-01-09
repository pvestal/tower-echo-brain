#!/bin/bash
"""
Echo Brain Media Scanner Management Script
Controls the media scanning process for local file analysis
"""

SCAN_DIR="/opt/tower-echo-brain"
SCAN_SCRIPT="echo_media_scanner.py"
LOG_FILE="media_scan.log"
PID_FILE="media_scan.pid"
MONITOR_SCRIPT="monitor_media_scan.py"

cd "$SCAN_DIR"

case "$1" in
    start)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                echo "Media scanner is already running (PID: $PID)"
                exit 1
            else
                echo "Removing stale PID file"
                rm -f "$PID_FILE"
            fi
        fi
        
        echo "Starting Echo Brain media scanner..."
        source venv/bin/activate
        nohup python3 "$SCAN_SCRIPT" > "$LOG_FILE" 2>&1 &
        PID=$!
        echo $PID > "$PID_FILE"
        echo "Media scanner started with PID: $PID"
        echo "Log file: $SCAN_DIR/$LOG_FILE"
        ;;
        
    stop)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            echo "Stopping media scanner (PID: $PID)..."
            kill "$PID" 2>/dev/null
            rm -f "$PID_FILE"
            echo "Media scanner stopped"
        else
            echo "Media scanner is not running"
        fi
        ;;
        
    status)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                echo "Media scanner is running (PID: $PID)"
                
                # Show quick progress
                source venv/bin/activate
                python3 "$MONITOR_SCRIPT" --once
            else
                echo "Media scanner is not running (stale PID file)"
                rm -f "$PID_FILE"
            fi
        else
            echo "Media scanner is not running"
        fi
        ;;
        
    log)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "No log file found"
        fi
        ;;
        
    monitor)
        echo "Starting progress monitor..."
        source venv/bin/activate
        python3 "$MONITOR_SCRIPT"
        ;;
        
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
        
    *)
        echo "Usage: $0 {start|stop|restart|status|log|monitor}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the media scanner"
        echo "  stop    - Stop the media scanner"
        echo "  restart - Restart the media scanner"
        echo "  status  - Show scanner status and progress"
        echo "  log     - Follow the scanner log"
        echo "  monitor - Start interactive progress monitor"
        exit 1
        ;;
esac
#!/bin/bash
# Production Monitoring Script for Echo Brain Model Routing
# Monitors performance, alerts on issues, and manages continuous learning

set -e

# Configuration
ECHO_BRAIN_URL="http://localhost:8309"
DB_HOST="localhost"
DB_NAME="echo_brain"
DB_USER="patrick"
DB_PASSWORD="***REMOVED***"
LOG_FILE="/var/log/echo-brain/monitoring.log"
ALERT_EMAIL="patrick.vestal@gmail.com"
SLACK_WEBHOOK=""  # Add Slack webhook URL if desired

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Thresholds
RESPONSE_TIME_WARNING=2000  # ms
RESPONSE_TIME_CRITICAL=5000  # ms
SATISFACTION_WARNING=3.5
SATISFACTION_CRITICAL=3.0
SUCCESS_RATE_WARNING=0.8
SUCCESS_RATE_CRITICAL=0.7
CPU_WARNING=80
CPU_CRITICAL=95
MEMORY_WARNING=80
MEMORY_CRITICAL=95

# Functions
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

send_alert() {
    local severity=$1
    local message=$2
    local details=$3

    # Log the alert
    log_message "ALERT [$severity]: $message - $details"

    # Send email alert if configured
    if [ -n "$ALERT_EMAIL" ]; then
        echo -e "Subject: Echo Brain Alert - $severity\n\n$message\n\nDetails:\n$details" | \
            sendmail "$ALERT_EMAIL" 2>/dev/null || true
    fi

    # Send Slack alert if configured
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\":warning: Echo Brain $severity Alert\n$message\n\`\`\`$details\`\`\`\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null || true
    fi
}

check_service_health() {
    log_message "Checking Echo Brain service health..."

    # Check if service is running
    if ! systemctl is-active --quiet tower-echo-brain; then
        send_alert "CRITICAL" "Echo Brain service is not running" \
            "Service status: $(systemctl status tower-echo-brain | head -5)"
        return 1
    fi

    # Check HTTP endpoint
    response=$(curl -s -o /dev/null -w "%{http_code}" "$ECHO_BRAIN_URL/api/echo/health" || echo "000")
    if [ "$response" != "200" ]; then
        send_alert "CRITICAL" "Echo Brain API not responding" \
            "HTTP response code: $response"
        return 1
    fi

    echo -e "${GREEN}✓${NC} Service health: OK"
    return 0
}

check_model_performance() {
    log_message "Checking model performance metrics..."

    # Query database for performance metrics
    metrics=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT
            AVG(response_time_ms) as avg_response_time,
            AVG(CASE WHEN user_satisfaction IS NOT NULL THEN user_satisfaction ELSE 0 END) as avg_satisfaction,
            COUNT(CASE WHEN was_successful THEN 1 END)::FLOAT / COUNT(*) as success_rate,
            COUNT(*) as total_queries
        FROM model_performance_log
        WHERE created_at >= NOW() - INTERVAL '1 hour'
    " 2>/dev/null | tr -d ' ')

    IFS='|' read -r avg_response_time avg_satisfaction success_rate total_queries <<< "$metrics"

    # Check response time
    if (( $(echo "$avg_response_time > $RESPONSE_TIME_CRITICAL" | bc -l) )); then
        send_alert "CRITICAL" "Response time critically high" \
            "Average response time: ${avg_response_time}ms (threshold: ${RESPONSE_TIME_CRITICAL}ms)"
    elif (( $(echo "$avg_response_time > $RESPONSE_TIME_WARNING" | bc -l) )); then
        send_alert "WARNING" "Response time elevated" \
            "Average response time: ${avg_response_time}ms (threshold: ${RESPONSE_TIME_WARNING}ms)"
    else
        echo -e "${GREEN}✓${NC} Response time: ${avg_response_time}ms"
    fi

    # Check satisfaction
    if (( $(echo "$avg_satisfaction < $SATISFACTION_CRITICAL" | bc -l) )); then
        send_alert "CRITICAL" "User satisfaction critically low" \
            "Average satisfaction: ${avg_satisfaction}/5.0 (threshold: ${SATISFACTION_CRITICAL})"
    elif (( $(echo "$avg_satisfaction < $SATISFACTION_WARNING" | bc -l) )); then
        send_alert "WARNING" "User satisfaction below target" \
            "Average satisfaction: ${avg_satisfaction}/5.0 (threshold: ${SATISFACTION_WARNING})"
    else
        echo -e "${GREEN}✓${NC} Satisfaction: ${avg_satisfaction}/5.0"
    fi

    # Check success rate
    if (( $(echo "$success_rate < $SUCCESS_RATE_CRITICAL" | bc -l) )); then
        send_alert "CRITICAL" "Success rate critically low" \
            "Success rate: $(echo "$success_rate * 100" | bc)% (threshold: $(echo "$SUCCESS_RATE_CRITICAL * 100" | bc)%)"
    elif (( $(echo "$success_rate < $SUCCESS_RATE_WARNING" | bc -l) )); then
        send_alert "WARNING" "Success rate below target" \
            "Success rate: $(echo "$success_rate * 100" | bc)% (threshold: $(echo "$SUCCESS_RATE_WARNING * 100" | bc)%)"
    else
        echo -e "${GREEN}✓${NC} Success rate: $(echo "$success_rate * 100" | bc)%"
    fi

    echo -e "${BLUE}ℹ${NC} Total queries (1h): $total_queries"
}

check_model_distribution() {
    log_message "Checking model distribution..."

    # Check if any model is dominating (potential routing issue)
    distribution=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT
            model_used,
            COUNT(*) as count,
            COUNT(*)::FLOAT / SUM(COUNT(*)) OVER () as percentage
        FROM model_performance_log
        WHERE created_at >= NOW() - INTERVAL '1 hour'
        GROUP BY model_used
        ORDER BY count DESC
    " 2>/dev/null)

    # Check for tinyllama usage (should be minimal)
    tinyllama_usage=$(echo "$distribution" | grep -i "tinyllama" | awk -F'|' '{print $3}' | tr -d ' ' || echo "0")

    if (( $(echo "$tinyllama_usage > 0.2" | bc -l) )); then
        send_alert "WARNING" "High tinyllama usage detected" \
            "Tinyllama is handling $(echo "$tinyllama_usage * 100" | bc)% of queries"
    fi

    echo -e "${BLUE}ℹ${NC} Model distribution:"
    echo "$distribution" | head -5
}

check_system_resources() {
    log_message "Checking system resources..."

    # Check CPU usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 | cut -d'.' -f1)

    if [ "$cpu_usage" -gt "$CPU_CRITICAL" ]; then
        send_alert "CRITICAL" "CPU usage critical" \
            "CPU usage: ${cpu_usage}% (threshold: ${CPU_CRITICAL}%)"
    elif [ "$cpu_usage" -gt "$CPU_WARNING" ]; then
        send_alert "WARNING" "CPU usage high" \
            "CPU usage: ${cpu_usage}% (threshold: ${CPU_WARNING}%)"
    else
        echo -e "${GREEN}✓${NC} CPU usage: ${cpu_usage}%"
    fi

    # Check memory usage
    memory_usage=$(free | grep Mem | awk '{print int($3/$2 * 100)}')

    if [ "$memory_usage" -gt "$MEMORY_CRITICAL" ]; then
        send_alert "CRITICAL" "Memory usage critical" \
            "Memory usage: ${memory_usage}% (threshold: ${MEMORY_CRITICAL}%)"
    elif [ "$memory_usage" -gt "$MEMORY_WARNING" ]; then
        send_alert "WARNING" "Memory usage high" \
            "Memory usage: ${memory_usage}% (threshold: ${MEMORY_WARNING}%)"
    else
        echo -e "${GREEN}✓${NC} Memory usage: ${memory_usage}%"
    fi

    # Check disk usage for logs
    disk_usage=$(df /var/log | tail -1 | awk '{print int($5)}')
    if [ "$disk_usage" -gt 90 ]; then
        send_alert "WARNING" "Log disk space low" \
            "Disk usage: ${disk_usage}% on /var/log"
    fi
}

run_continuous_learning() {
    log_message "Running continuous learning analysis..."

    # Execute Python learning script
    python3 << 'PYEOF'
import sys
sys.path.insert(0, '/opt/tower-echo-brain')
from src.learning.continuous_learning import ContinuousLearningSystem

db_config = {
    'host': 'localhost',
    'database': 'echo_brain',
    'user': 'patrick',
    'password': '***REMOVED***'
}

learning = ContinuousLearningSystem(db_config)
result = learning.apply_learning(auto_apply=True)

print(f"Learning adjustments: {result['adjustments_applied']} applied, {result['adjustments_skipped']} skipped")

if result['applied']:
    print("Applied adjustments:")
    for adj in result['applied']:
        print(f"  - {adj['type']}: {adj['reason']}")
PYEOF

    echo -e "${GREEN}✓${NC} Continuous learning analysis complete"
}

check_recent_errors() {
    log_message "Checking for recent errors..."

    # Check Echo Brain logs for errors
    error_count=$(sudo journalctl -u tower-echo-brain -S "1 hour ago" | grep -c "ERROR\|CRITICAL" || echo "0")

    if [ "$error_count" -gt 10 ]; then
        recent_errors=$(sudo journalctl -u tower-echo-brain -S "1 hour ago" | grep "ERROR\|CRITICAL" | tail -5)
        send_alert "WARNING" "High error rate in Echo Brain logs" \
            "Error count: $error_count in last hour\n\nRecent errors:\n$recent_errors"
    else
        echo -e "${GREEN}✓${NC} Error count (1h): $error_count"
    fi
}

generate_report() {
    log_message "Generating performance report..."

    report=$(python3 << 'PYEOF'
import sys
sys.path.insert(0, '/opt/tower-echo-brain')
from src.learning.continuous_learning import ContinuousLearningSystem
import json

db_config = {
    'host': 'localhost',
    'database': 'echo_brain',
    'user': 'patrick',
    'password': '***REMOVED***'
}

learning = ContinuousLearningSystem(db_config)
report = learning.get_performance_report(days=1)

print("📊 24-Hour Performance Report")
print("=" * 60)

if report.get('model_performance'):
    print("\n🤖 Model Performance:")
    for model in report['model_performance'][:5]:
        if model['satisfaction'] is not None:
            print(f"  {model['model']}: {model['satisfaction']:.2f}/5.0 ({model['queries']} queries)")

if report.get('daily_trends'):
    print("\n📈 Daily Trends:")
    for trend in report['daily_trends'][-3:]:
        print(f"  {trend['date']}: {trend['avg_satisfaction']:.2f}/5.0 ({trend['queries']} queries)")

if report.get('recent_adjustments'):
    print("\n🔧 Recent Adjustments:")
    for adj in report['recent_adjustments'][:3]:
        print(f"  [{adj['type']}] {adj['parameter']}: Applied={adj['applied']}")
PYEOF
)

    echo "$report"
}

# Main monitoring loop
main() {
    echo -e "${BLUE}🧠 Echo Brain Production Monitor${NC}"
    echo "=================================="
    echo "Started at: $(date)"
    echo ""

    # Create log directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"

    while true; do
        echo -e "\n${YELLOW}--- Monitor Check: $(date '+%H:%M:%S') ---${NC}"

        # Run all checks
        check_service_health
        check_model_performance
        check_model_distribution
        check_system_resources
        check_recent_errors

        # Run continuous learning every 30 minutes
        if [ $(($(date +%M) % 30)) -eq 0 ]; then
            run_continuous_learning
        fi

        # Generate report every hour
        if [ $(($(date +%M))) -eq 0 ]; then
            generate_report
        fi

        # Sleep for monitoring interval
        echo -e "\n${BLUE}ℹ${NC} Next check in 60 seconds..."
        sleep 60
    done
}

# Handle script arguments
case "${1:-}" in
    start)
        main
        ;;
    check)
        check_service_health
        check_model_performance
        ;;
    report)
        generate_report
        ;;
    learn)
        run_continuous_learning
        ;;
    *)
        echo "Usage: $0 {start|check|report|learn}"
        echo "  start  - Start continuous monitoring"
        echo "  check  - Run single health check"
        echo "  report - Generate performance report"
        echo "  learn  - Run continuous learning analysis"
        exit 1
        ;;
esac

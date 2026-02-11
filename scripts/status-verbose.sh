#!/bin/bash

# Echo Brain Comprehensive Status Check with Verbose Debug Output
# ================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${CYAN}                        ECHO BRAIN SYSTEM STATUS REPORT                        ${NC}"
echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Timestamp:${NC} $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo -e "${BLUE}Host:${NC} $(hostname -f) ($(hostname -I | awk '{print $1}'))"
echo ""

# ========== CORE SERVICE STATUS ==========
echo -e "${BOLD}${MAGENTA}▶ CORE SERVICE STATUS${NC}"
echo -e "${CYAN}────────────────────────────────────────────────────────────────────────────${NC}"

# Check Echo Brain API Health
echo -e "\n${BOLD}[1/5] Echo Brain API (Port 8309)${NC}"
echo -e "${BLUE}Testing:${NC} http://localhost:8309/health"
HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}\n%{time_total}" http://localhost:8309/health 2>/dev/null)
HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -n2 | head -n1)
RESPONSE_TIME=$(echo "$HEALTH_RESPONSE" | tail -n1)
HEALTH_JSON=$(echo "$HEALTH_RESPONSE" | head -n-2)

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✅ STATUS: HEALTHY${NC}"
    echo -e "${BLUE}Response Time:${NC} ${RESPONSE_TIME}s"
    echo -e "${BLUE}Response:${NC}"
    echo "$HEALTH_JSON" | jq '.' 2>/dev/null || echo "$HEALTH_JSON"
else
    echo -e "${RED}❌ STATUS: UNHEALTHY (HTTP $HTTP_CODE)${NC}"
    echo -e "${BLUE}Response Time:${NC} ${RESPONSE_TIME}s"
fi

# Check systemd service
echo -e "\n${BLUE}Systemd Service:${NC}"
SYSTEMD_STATUS=$(systemctl status tower-echo-brain 2>&1 | head -n3)
if systemctl is-active tower-echo-brain >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Service Active${NC}"
    systemctl status tower-echo-brain --no-pager | head -n5
else
    echo -e "${RED}❌ Service Inactive${NC}"
    echo "$SYSTEMD_STATUS"
fi

# ========== API ENDPOINT TESTS ==========
echo -e "\n${BOLD}${MAGENTA}▶ API ENDPOINT TESTS${NC}"
echo -e "${CYAN}────────────────────────────────────────────────────────────────────────────${NC}"

# Test /docs endpoint
echo -e "\n${BOLD}[2/5] API Documentation Endpoint${NC}"
echo -e "${BLUE}Testing:${NC} http://localhost:8309/docs"
DOCS_TEST=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8309/docs)
if [ "$DOCS_TEST" = "200" ]; then
    echo -e "${GREEN}✅ /docs endpoint accessible${NC}"
else
    echo -e "${RED}❌ /docs endpoint failed (HTTP $DOCS_TEST)${NC}"
fi

# Test /api/echo/knowledge/stats endpoint
echo -e "\n${BOLD}[3/5] Knowledge Stats Endpoint${NC}"
echo -e "${BLUE}Testing:${NC} http://localhost:8309/api/echo/knowledge/stats"
STATS_RESPONSE=$(curl -s http://localhost:8309/api/echo/knowledge/stats 2>/dev/null)
if [ $? -eq 0 ] && [ -n "$STATS_RESPONSE" ] && ! echo "$STATS_RESPONSE" | grep -q "Not Found"; then
    echo -e "${GREEN}✅ Knowledge stats endpoint responding${NC}"
    echo -e "${BLUE}Response:${NC}"
    echo "$STATS_RESPONSE" | jq '.' 2>/dev/null || echo "$STATS_RESPONSE"
else
    echo -e "${YELLOW}⚠️ Knowledge stats endpoint not available${NC}"
fi

# ========== MCP SERVER TESTS ==========
echo -e "\n${BOLD}${MAGENTA}▶ MCP SERVER FUNCTIONALITY${NC}"
echo -e "${CYAN}────────────────────────────────────────────────────────────────────────────${NC}"

echo -e "\n${BOLD}[4/5] MCP search_memory Test${NC}"
echo -e "${BLUE}Query:${NC} 'tower system architecture'"
MCP_SEARCH=$(curl -s -X POST http://localhost:8309/mcp \
  -H "Content-Type: application/json" \
  -d '{"method": "tools/call", "params": {"name": "search_memory", "arguments": {"query": "tower system architecture", "limit": 2}}}' 2>/dev/null)

if [ $? -eq 0 ] && [ -n "$MCP_SEARCH" ]; then
    echo -e "${GREEN}✅ MCP search_memory working${NC}"
    echo -e "${BLUE}Sample Result (truncated):${NC}"
    echo "$MCP_SEARCH" | jq -r '.content[0].text' 2>/dev/null | head -n5 || echo "$MCP_SEARCH" | head -n3
    RESULT_COUNT=$(echo "$MCP_SEARCH" | jq -r '.content | length' 2>/dev/null || echo "0")
    echo -e "${BLUE}Results returned:${NC} $RESULT_COUNT"
else
    echo -e "${RED}❌ MCP search_memory failed${NC}"
fi

echo -e "\n${BOLD}MCP get_facts Test${NC}"
echo -e "${BLUE}Topic:${NC} 'patrick'"
MCP_FACTS=$(curl -s -X POST http://localhost:8309/mcp \
  -H "Content-Type: application/json" \
  -d '{"method": "tools/call", "params": {"name": "get_facts", "arguments": {"topic": "patrick"}}}' 2>/dev/null)

if [ $? -eq 0 ] && [ -n "$MCP_FACTS" ]; then
    echo -e "${GREEN}✅ MCP get_facts working${NC}"
    FACT_COUNT=$(echo "$MCP_FACTS" | jq -r '.content[0].text' | grep -c "•" 2>/dev/null || echo "0")
    echo -e "${BLUE}Facts found:${NC} $FACT_COUNT"
    echo -e "${BLUE}Sample facts:${NC}"
    echo "$MCP_FACTS" | jq -r '.content[0].text' 2>/dev/null | head -n10 || echo "$MCP_FACTS" | head -n5
else
    echo -e "${RED}❌ MCP get_facts failed${NC}"
fi

# ========== DATABASE CONNECTIVITY ==========
echo -e "\n${BOLD}${MAGENTA}▶ DATABASE CONNECTIVITY${NC}"
echo -e "${CYAN}────────────────────────────────────────────────────────────────────────────${NC}"

echo -e "\n${BOLD}[5/5] PostgreSQL Database Check${NC}"
echo -e "${BLUE}Database:${NC} echo_brain"

# Check conversations table
CONV_COUNT=$(PGPASSWORD="${DB_PASSWORD:?DB_PASSWORD not set}" psql -h localhost -U patrick -d echo_brain -t -c "SELECT COUNT(*) FROM conversations" 2>/dev/null | xargs)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ PostgreSQL connection successful${NC}"
    echo -e "${BLUE}Conversations:${NC} $CONV_COUNT records"
else
    echo -e "${RED}❌ PostgreSQL connection failed${NC}"
fi

# Check all tables with counts
echo -e "\n${BLUE}Table Statistics:${NC}"
TABLE_STATS=$(PGPASSWORD="${DB_PASSWORD:?DB_PASSWORD not set}" psql -h localhost -U patrick -d echo_brain -t -c "
SELECT
    'conversations' as table_name, COUNT(*) as count
FROM conversations
UNION ALL
SELECT
    'echo_conversations', COUNT(*)
FROM echo_conversations
UNION ALL
SELECT
    'facts', COUNT(*)
FROM facts
UNION ALL
SELECT
    'ingestion_runs', COUNT(*)
FROM ingestion_runs
UNION ALL
SELECT
    'claude_conversations', COUNT(*)
FROM claude_conversations
ORDER BY count DESC;" 2>/dev/null)

if [ $? -eq 0 ]; then
    echo "$TABLE_STATS" | while read -r line; do
        TABLE=$(echo "$line" | awk '{print $1}')
        COUNT=$(echo "$line" | awk '{print $3}')
        printf "  %-20s %10s records\n" "$TABLE" "$COUNT"
    done
else
    echo -e "${YELLOW}⚠️ Could not retrieve table statistics${NC}"
fi

# ========== QDRANT VECTOR DATABASE ==========
echo -e "\n${BOLD}${MAGENTA}▶ QDRANT VECTOR DATABASE${NC}"
echo -e "${CYAN}────────────────────────────────────────────────────────────────────────────${NC}"

echo -e "\n${BOLD}Qdrant Service (Port 6333)${NC}"
QDRANT_HEALTH=$(curl -s http://localhost:6333/health 2>/dev/null)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Qdrant service healthy${NC}"

    # Get collections
    echo -e "\n${BLUE}Vector Collections:${NC}"
    COLLECTIONS=$(curl -s http://localhost:6333/collections 2>/dev/null | jq -r '.result.collections[].name' 2>/dev/null)

    for collection in $COLLECTIONS; do
        COLLECTION_INFO=$(curl -s "http://localhost:6333/collections/$collection" 2>/dev/null)
        VECTOR_COUNT=$(echo "$COLLECTION_INFO" | jq -r '.result.vectors_count' 2>/dev/null || echo "unknown")
        DIMENSION=$(echo "$COLLECTION_INFO" | jq -r '.result.config.params.vectors.size' 2>/dev/null || echo "unknown")
        printf "  %-25s %10s vectors (%s dims)\n" "$collection" "$VECTOR_COUNT" "$DIMENSION"
    done
else
    echo -e "${RED}❌ Qdrant service unavailable${NC}"
fi

# ========== RESOURCE UTILIZATION ==========
echo -e "\n${BOLD}${MAGENTA}▶ RESOURCE UTILIZATION${NC}"
echo -e "${CYAN}────────────────────────────────────────────────────────────────────────────${NC}"

# Python processes
echo -e "\n${BOLD}Python Processes:${NC}"
ps aux | grep -E "python.*echo-brain" | grep -v grep | while read -r line; do
    PID=$(echo "$line" | awk '{print $2}')
    CPU=$(echo "$line" | awk '{print $3}')
    MEM=$(echo "$line" | awk '{print $4}')
    CMD=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}' | cut -c1-60)
    printf "  PID %-6s CPU: %5s%% MEM: %5s%% %s\n" "$PID" "$CPU" "$MEM" "$CMD"
done

# Port listening
echo -e "\n${BOLD}Port Status:${NC}"
for port in 8309 8310 8311 6333 11434; do
    # Use netstat or ss as fallback for more reliable detection
    if netstat -tln 2>/dev/null | grep -q ":$port " || ss -tln 2>/dev/null | grep -q ":$port "; then
        SERVICE=$(lsof -i:$port 2>/dev/null | tail -n1 | awk '{print $1}' || echo "active")
        echo -e "  ${GREEN}✅${NC} Port $port: $SERVICE"
    else
        echo -e "  ${RED}❌${NC} Port $port: Not listening"
    fi
done

# ========== INGESTION PIPELINE ==========
echo -e "\n${BOLD}${MAGENTA}▶ INGESTION PIPELINE${NC}"
echo -e "${CYAN}────────────────────────────────────────────────────────────────────────────${NC}"

# Check timer status
echo -e "\n${BOLD}Systemd Timer Status:${NC}"
TIMER_STATUS=$(systemctl status echo-brain-ingest.timer 2>&1 | head -n3)
if systemctl is-active echo-brain-ingest.timer >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Timer Active${NC}"
    NEXT_RUN=$(systemctl status echo-brain-ingest.timer | grep "Trigger:" | sed 's/.*Trigger:/Next Run:/')
    echo "$NEXT_RUN"
else
    echo -e "${YELLOW}⚠️ Timer Inactive${NC}"
fi

# Check last ingestion
echo -e "\n${BOLD}Last Ingestion:${NC}"
LAST_INGESTION=$(PGPASSWORD="${DB_PASSWORD:?DB_PASSWORD not set}" psql -h localhost -U patrick -d echo_brain -t -c "
SELECT
    MAX(created_at) as last_time,
    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 hour') as last_hour,
    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as last_day
FROM conversations;" 2>/dev/null)

if [ $? -eq 0 ]; then
    LAST_TIME=$(echo "$LAST_INGESTION" | awk -F'|' '{print $1}' | xargs)
    LAST_HOUR=$(echo "$LAST_INGESTION" | awk -F'|' '{print $2}' | xargs)
    LAST_DAY=$(echo "$LAST_INGESTION" | awk -F'|' '{print $3}' | xargs)

    echo -e "${BLUE}Last conversation:${NC} $LAST_TIME"
    echo -e "${BLUE}Added in last hour:${NC} $LAST_HOUR"
    echo -e "${BLUE}Added in last 24h:${NC} $LAST_DAY"
else
    echo -e "${YELLOW}⚠️ Could not retrieve ingestion stats${NC}"
fi

# ========== SUMMARY ==========
echo -e "\n${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${CYAN}                                  SUMMARY                                      ${NC}"
echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Count successes
SUCCESS_COUNT=0
TOTAL_CHECKS=10

[ "$HTTP_CODE" = "200" ] && ((SUCCESS_COUNT++))
[ "$DOCS_TEST" = "200" ] && ((SUCCESS_COUNT++))
[ -n "$MCP_SEARCH" ] && ((SUCCESS_COUNT++))
[ -n "$MCP_FACTS" ] && ((SUCCESS_COUNT++))
[ -n "$CONV_COUNT" ] && ((SUCCESS_COUNT++))
[ -n "$QDRANT_HEALTH" ] && ((SUCCESS_COUNT++))
systemctl is-active tower-echo-brain >/dev/null 2>&1 && ((SUCCESS_COUNT++))
lsof -i:8309 >/dev/null 2>&1 && ((SUCCESS_COUNT++))
lsof -i:6333 >/dev/null 2>&1 && ((SUCCESS_COUNT++))
systemctl is-active echo-learning-pipeline.timer >/dev/null 2>&1 && ((SUCCESS_COUNT++))

HEALTH_PERCENT=$((SUCCESS_COUNT * 100 / TOTAL_CHECKS))

if [ $HEALTH_PERCENT -ge 90 ]; then
    echo -e "${GREEN}${BOLD}SYSTEM STATUS: FULLY OPERATIONAL ($SUCCESS_COUNT/$TOTAL_CHECKS checks passed)${NC}"
elif [ $HEALTH_PERCENT -ge 70 ]; then
    echo -e "${YELLOW}${BOLD}SYSTEM STATUS: DEGRADED ($SUCCESS_COUNT/$TOTAL_CHECKS checks passed)${NC}"
else
    echo -e "${RED}${BOLD}SYSTEM STATUS: CRITICAL ($SUCCESS_COUNT/$TOTAL_CHECKS checks passed)${NC}"
fi

echo -e "\n${BLUE}Run with watch for continuous monitoring:${NC}"
echo "  watch -n 5 -c $0"
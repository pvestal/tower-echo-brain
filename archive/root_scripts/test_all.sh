#!/bin/bash
set -e
echo "═══════════════════════════════════════════════════════════"
echo "ECHO BRAIN FOUNDATION VALIDATION"
echo "Started: $(date)"
echo "═══════════════════════════════════════════════════════════"

PASS=0
FAIL=0

# Test 1: Service Health
echo -e "\n[TEST 1] Service Health"
if curl -s http://localhost:8309/health | jq -e '.status == "healthy"' > /dev/null; then
    echo "✅ Service healthy"
    ((PASS++))
else
    echo "❌ Service unhealthy"
    ((FAIL++))
fi

# Test 2: Routing - Conversation
echo -e "\n[TEST 2] Routing: Conversation → llama3.2:3b"
MODEL=$(curl -s -X POST http://localhost:8309/api/echo/chat \
    -H "Content-Type: application/json" \
    -d '{"query": "Hello"}' | jq -r '.model_used')
if [ "$MODEL" = "llama3.2:3b" ]; then
    echo "✅ Correct: $MODEL"
    ((PASS++))
else
    echo "❌ Wrong model: $MODEL (expected llama3.2:3b)"
    ((FAIL++))
fi

# Test 3: Routing - Code
echo -e "\n[TEST 3] Routing: Code → deepseek-r1:8b"
MODEL=$(curl -s -X POST http://localhost:8309/api/echo/chat \
    -H "Content-Type: application/json" \
    -d '{"query": "Write Python to sort a list"}' | jq -r '.model_used')
if [[ "$MODEL" == *"deepseek"* ]]; then
    echo "✅ Correct: $MODEL"
    ((PASS++))
else
    echo "❌ Wrong model: $MODEL (expected deepseek)"
    ((FAIL++))
fi

# Test 4: Routing - Analysis
echo -e "\n[TEST 4] Routing: Analysis → gemma2:9b"
MODEL=$(curl -s -X POST http://localhost:8309/api/echo/chat \
    -H "Content-Type: application/json" \
    -d '{"query": "Explain quantum physics"}' | jq -r '.model_used')
if [ "$MODEL" = "gemma2:9b" ]; then
    echo "✅ Correct: $MODEL"
    ((PASS++))
else
    echo "⚠️ Different model: $MODEL (expected gemma2:9b)"
    ((FAIL++))
fi

# Test 5: Response Quality
echo -e "\n[TEST 5] Response has content"
RESP_LEN=$(curl -s -X POST http://localhost:8309/api/echo/chat \
    -H "Content-Type: application/json" \
    -d '{"query": "Say hello"}' | jq '.response | length')
if [ "$RESP_LEN" -gt 10 ]; then
    echo "✅ Response length: $RESP_LEN chars"
    ((PASS++))
else
    echo "❌ Empty or short response: $RESP_LEN chars"
    ((FAIL++))
fi

# Test 6: Dashboard Endpoint
echo -e "\n[TEST 6] Dashboard services endpoint"
SVC_COUNT=$(curl -s http://localhost:8309/api/coordination/services | jq '.services | length')
if [ "$SVC_COUNT" -gt 0 ]; then
    echo "✅ Found $SVC_COUNT services"
    ((PASS++))
else
    echo "❌ No services returned"
    ((FAIL++))
fi

# Test 7: Database Routing Table
echo -e "\n[TEST 7] Database has routing config"
COUNT=$(PGPASSWORD=tower_echo_brain_secret_key_2025 psql -U patrick -d tower_consolidated -t \
    -c "SELECT COUNT(*) FROM intent_model_mapping;" 2>/dev/null | tr -d ' ')
if [ "$COUNT" -gt 0 ]; then
    echo "✅ intent_model_mapping has $COUNT rows"
    ((PASS++))
else
    echo "❌ intent_model_mapping empty or missing"
    ((FAIL++))
fi

# Test 8: No Emergency Bypass
echo -e "\n[TEST 8] Emergency bypass removed"
if ! grep -q "minimal_router" /opt/tower-echo-brain/src/api/echo.py; then
    echo "✅ No minimal_router references"
    ((PASS++))
else
    echo "❌ minimal_router still in code"
    ((FAIL++))
fi

# Test 9: Single Source of Truth
echo -e "\n[TEST 9] No hardcoded TIER_TO_MODEL"
TIER_COUNT=$(grep -rn "TIER_TO_MODEL\s*=" /opt/tower-echo-brain/src --include="*.py" | grep -v "#" | wc -l)
if [ "$TIER_COUNT" -eq 0 ]; then
    echo "✅ No TIER_TO_MODEL assignments"
    ((PASS++))
else
    echo "⚠️ Found $TIER_COUNT TIER_TO_MODEL assignments"
    ((FAIL++))
fi

# Test 10: Recent Logs Clean
echo -e "\n[TEST 10] No critical errors in logs"
ERROR_COUNT=$(sudo journalctl -u tower-echo-brain --since "5 minutes ago" 2>/dev/null | \
    grep -iE "CRITICAL|FATAL|Traceback" | wc -l)
if [ "$ERROR_COUNT" -eq 0 ]; then
    echo "✅ No critical errors"
    ((PASS++))
else
    echo "⚠️ Found $ERROR_COUNT critical errors"
    ((FAIL++))
fi

# Summary
echo -e "\n═══════════════════════════════════════════════════════════"
echo "RESULTS: $PASS passed, $FAIL failed"
echo "═══════════════════════════════════════════════════════════"

if [ $FAIL -eq 0 ]; then
    echo "✅ FOUNDATION VALIDATED - All tests passed"
    exit 0
else
    echo "⚠️ ISSUES FOUND - Review failures above"
    exit 1
fi

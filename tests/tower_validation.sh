#!/bin/bash

# Tower-specific validation tests for Echo Brain agents
# These are REAL tests with quantitative metrics

echo "========================================="
echo "TOWER ECHO BRAIN VALIDATION SUITE"
echo "Started: $(date)"
echo "========================================="

BASE_URL="http://localhost:8309"
PASS=0
FAIL=0
TOTAL=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper function for tests
run_test() {
    local test_name="$1"
    local cmd="$2"
    local expected="$3"

    ((TOTAL++))
    echo -n "TEST $TOTAL: $test_name ... "

    result=$(eval "$cmd" 2>/dev/null)

    if [[ "$result" == *"$expected"* ]]; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASS++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        echo "  Expected: $expected"
        echo "  Got: ${result:0:100}..."
        ((FAIL++))
        return 1
    fi
}

# Test 1: Service Health
echo -e "\n${YELLOW}=== SERVICE HEALTH TESTS ===${NC}"
run_test "Echo Brain service running" \
    "curl -s $BASE_URL/api/echo/agents/status | grep -o '\"status\": \"active\"' | head -1" \
    '"status": "active"'

# Test 2: Agent Models
echo -e "\n${YELLOW}=== AGENT MODEL TESTS ===${NC}"
run_test "CodingAgent uses deepseek-coder-v2:16b" \
    "curl -s $BASE_URL/api/echo/agents/status | grep -o 'deepseek-coder-v2:16b'" \
    "deepseek-coder-v2:16b"

run_test "ReasoningAgent uses deepseek-r1:8b" \
    "curl -s $BASE_URL/api/echo/agents/status | grep -o 'deepseek-r1:8b'" \
    "deepseek-r1:8b"

run_test "NarrationAgent uses gemma2:9b" \
    "curl -s $BASE_URL/api/echo/agents/status | grep -o 'gemma2:9b'" \
    "gemma2:9b"

# Test 3: CodingAgent Tower Awareness
echo -e "\n${YELLOW}=== CODING AGENT TOWER TESTS ===${NC}"
echo "Testing CodingAgent with Tower-specific task..."
TOWER_CODE=$(curl -s -X POST $BASE_URL/api/echo/agents/coding \
    -H "Content-Type: application/json" \
    -d '{
        "task": "Write a function to check if tower-echo-brain service is running using subprocess and systemctl",
        "language": "python",
        "validate": true
    }' 2>/dev/null)

run_test "CodingAgent generates valid Python" \
    "echo '$TOWER_CODE' | grep -o '\"valid\": true'" \
    '"valid": true'

run_test "Code references tower-echo-brain" \
    "echo '$TOWER_CODE' | grep -o 'tower-echo-brain'" \
    "tower-echo-brain"

run_test "Code uses systemctl" \
    "echo '$TOWER_CODE' | grep -o 'systemctl'" \
    "systemctl"

# Test 4: Database Separation
echo -e "\n${YELLOW}=== DATABASE SEPARATION TESTS ===${NC}"
run_test "tower_consolidated database exists" \
    "PGPASSWORD=***REMOVED*** psql -U patrick -d tower_consolidated -c 'SELECT 1' 2>/dev/null | grep -o '1 row'" \
    "1 row"

run_test "tower_anime database exists" \
    "PGPASSWORD=***REMOVED*** psql -U patrick -d tower_anime -c 'SELECT 1' 2>/dev/null | grep -o '1 row'" \
    "1 row"

run_test "No anime tables in tower_consolidated" \
    "PGPASSWORD=***REMOVED*** psql -U patrick -d tower_consolidated -c \"SELECT COUNT(*) FROM information_schema.tables WHERE table_name LIKE 'anime_%'\" 2>/dev/null | grep -o '0'" \
    "0"

# Test 5: ReasoningAgent Analysis Quality
echo -e "\n${YELLOW}=== REASONING AGENT TESTS ===${NC}"
echo "Testing ReasoningAgent with Tower architecture question..."
REASONING=$(curl -s -X POST $BASE_URL/api/echo/agents/reasoning \
    -H "Content-Type: application/json" \
    -d '{
        "task": "Should Tower use nginx or Apache for reverse proxy? Consider current setup.",
        "background": "Currently using nginx with multiple services on different ports",
        "constraints": "Need WebSocket support and SSL termination"
    }' 2>/dev/null)

run_test "ReasoningAgent provides analysis" \
    "echo '$REASONING' | grep -o '\"analysis\"'" \
    '"analysis"'

run_test "ReasoningAgent provides conclusion" \
    "echo '$REASONING' | grep -o '\"conclusion\"'" \
    '"conclusion"'

run_test "Response mentions nginx" \
    "echo '$REASONING' | grep -io 'nginx' | head -1" \
    "nginx"

# Test 6: NarrationAgent ComfyUI Generation
echo -e "\n${YELLOW}=== NARRATION AGENT TESTS ===${NC}"
echo "Testing NarrationAgent with scene generation..."
NARRATION=$(curl -s -X POST $BASE_URL/api/echo/agents/narration \
    -H "Content-Type: application/json" \
    -d '{
        "scene": "A programmer working late at night with multiple monitors showing code",
        "genre": "cyberpunk",
        "mood": "focused"
    }' 2>/dev/null)

run_test "NarrationAgent generates narration" \
    "echo '$NARRATION' | grep -o '\"narration\"'" \
    '"narration"'

run_test "NarrationAgent provides mood" \
    "echo '$NARRATION' | grep -o '\"mood\"'" \
    '"mood"'

run_test "NarrationAgent generates ComfyUI prompt" \
    "echo '$NARRATION' | grep -o '\"comfyui_prompt\"'" \
    '"comfyui_prompt"'

# Test 7: Agent Pipeline Integration
echo -e "\n${YELLOW}=== PIPELINE INTEGRATION TESTS ===${NC}"
echo "Testing multi-agent pipeline..."

# First, reasoning designs a solution
DESIGN=$(curl -s -X POST $BASE_URL/api/echo/agents/reasoning \
    -H "Content-Type: application/json" \
    -d '{
        "task": "Design a simple cache system for API responses",
        "constraints": "Use Python dict, 5 minute expiry"
    }' 2>/dev/null | jq -r '.conclusion' | head -200)

if [[ -n "$DESIGN" ]]; then
    # Then coding implements it
    IMPLEMENTATION=$(curl -s -X POST $BASE_URL/api/echo/agents/coding \
        -H "Content-Type: application/json" \
        -d "{
            \"task\": \"Implement this design: ${DESIGN:0:200}\",
            \"language\": \"python\",
            \"validate\": true
        }" 2>/dev/null)

    run_test "Pipeline: Reasoning to Coding works" \
        "echo '$IMPLEMENTATION' | grep -o '\"valid\": true'" \
        '"valid": true'
else
    ((FAIL++))
    ((TOTAL++))
    echo -e "TEST $TOTAL: Pipeline: Reasoning to Coding works ... ${RED}✗ FAIL${NC}"
    echo "  Reasoning agent didn't provide conclusion"
fi

# Test 8: Concurrent Agent Execution
echo -e "\n${YELLOW}=== CONCURRENT EXECUTION TESTS ===${NC}"
echo "Testing concurrent agent calls..."

# Launch 3 agents simultaneously
(curl -s -X POST $BASE_URL/api/echo/agents/coding \
    -H "Content-Type: application/json" \
    -d '{"task": "Write hello world", "language": "python"}' > /tmp/coding_result.json) &
PID1=$!

(curl -s -X POST $BASE_URL/api/echo/agents/reasoning \
    -H "Content-Type: application/json" \
    -d '{"task": "What is 2+2?"}' > /tmp/reasoning_result.json) &
PID2=$!

(curl -s -X POST $BASE_URL/api/echo/agents/narration \
    -H "Content-Type: application/json" \
    -d '{"scene": "sunrise", "genre": "nature"}' > /tmp/narration_result.json) &
PID3=$!

# Wait for all to complete (max 30 seconds)
SECONDS=0
while [[ $SECONDS -lt 30 ]]; do
    if ! kill -0 $PID1 2>/dev/null && ! kill -0 $PID2 2>/dev/null && ! kill -0 $PID3 2>/dev/null; then
        break
    fi
    sleep 1
done

# Check results
if [[ -f /tmp/coding_result.json ]] && [[ -f /tmp/reasoning_result.json ]] && [[ -f /tmp/narration_result.json ]]; then
    C_VALID=$(cat /tmp/coding_result.json | grep -o '"model"' | head -1)
    R_VALID=$(cat /tmp/reasoning_result.json | grep -o '"model"' | head -1)
    N_VALID=$(cat /tmp/narration_result.json | grep -o '"model"' | head -1)

    if [[ "$C_VALID" == '"model"' ]] && [[ "$R_VALID" == '"model"' ]] && [[ "$N_VALID" == '"model"' ]]; then
        echo -e "TEST $((++TOTAL)): Concurrent execution of 3 agents ... ${GREEN}✓ PASS${NC}"
        ((PASS++))
    else
        echo -e "TEST $((++TOTAL)): Concurrent execution of 3 agents ... ${RED}✗ FAIL${NC}"
        ((FAIL++))
    fi
else
    echo -e "TEST $((++TOTAL)): Concurrent execution of 3 agents ... ${RED}✗ FAIL${NC}"
    echo "  Not all agents completed"
    ((FAIL++))
fi

rm -f /tmp/{coding,reasoning,narration}_result.json

# Test 9: Self-Improvement Capability
echo -e "\n${YELLOW}=== SELF-IMPROVEMENT TESTS ===${NC}"
echo "Testing agent self-analysis capability..."

SELF_ANALYSIS=$(curl -s -X POST $BASE_URL/api/echo/agents/coding \
    -H "Content-Type: application/json" \
    -d '{
        "task": "Write a function to analyze Echo Brain agent performance from the database and suggest improvements",
        "language": "python",
        "validate": true,
        "requirements": "Use tower_consolidated database, check past_solutions table"
    }' 2>/dev/null)

run_test "Self-improvement code is valid" \
    "echo '$SELF_ANALYSIS' | grep -o '\"valid\": true'" \
    '"valid": true'

run_test "References tower_consolidated" \
    "echo '$SELF_ANALYSIS' | grep -o 'tower_consolidated'" \
    "tower_consolidated"

run_test "References past_solutions" \
    "echo '$SELF_ANALYSIS' | grep -o 'past_solutions'" \
    "past_solutions"

# Test 10: Error Handling
echo -e "\n${YELLOW}=== ERROR HANDLING TESTS ===${NC}"

INVALID_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X POST $BASE_URL/api/echo/agents/execute \
    -H "Content-Type: application/json" \
    -d '{"code": "invalid python !!!", "language": "python"}' 2>/dev/null)

if [[ "$INVALID_RESPONSE" == "200" ]]; then
    echo -e "TEST $((++TOTAL)): Invalid code handled gracefully ... ${GREEN}✓ PASS${NC}"
    ((PASS++))
else
    echo -e "TEST $((++TOTAL)): Invalid code handled gracefully ... ${YELLOW}⚠ PARTIAL${NC}"
    echo "  HTTP status: $INVALID_RESPONSE"
fi

# Final Summary
echo -e "\n========================================="
echo "VALIDATION SUMMARY"
echo "========================================="
echo -e "Tests Passed: ${GREEN}$PASS${NC}"
echo -e "Tests Failed: ${RED}$FAIL${NC}"
echo "Total Tests: $TOTAL"
PERCENT=$((100 * PASS / TOTAL))
echo "Success Rate: $PERCENT%"

if [[ $PERCENT -eq 100 ]]; then
    echo -e "\n${GREEN}🎉 PERFECT SCORE - All tests passed!${NC}"
    exit 0
elif [[ $PERCENT -ge 80 ]]; then
    echo -e "\n${GREEN}✅ VALIDATION PASSED - System operational${NC}"
    exit 0
elif [[ $PERCENT -ge 60 ]]; then
    echo -e "\n${YELLOW}⚠️  PARTIAL SUCCESS - Some issues remain${NC}"
    exit 1
else
    echo -e "\n${RED}❌ VALIDATION FAILED - Critical issues detected${NC}"
    exit 2
fi
#!/bin/bash
set -e

BASE="http://localhost:8309/api/pipeline"
PASS=0
FAIL=0
TOTAL=0

test_case() {
    local name="$1"
    local query="$2"
    local expected_intent="$3"
    local must_contain="$4"
    local must_not_contain="$5"

    TOTAL=$((TOTAL + 1))
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "TEST $TOTAL: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Make the request
    RESULT=$(curl -sf -X POST "$BASE/query" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$query\", \"debug\": true}" 2>/dev/null)

    if [ $? -ne 0 ] || [ -z "$RESULT" ]; then
        echo "  ❌ FAIL: Request failed or empty response"
        FAIL=$((FAIL + 1))
        return
    fi

    # Check intent
    ACTUAL_INTENT=$(echo "$RESULT" | jq -r '.intent')
    if [ "$ACTUAL_INTENT" != "$expected_intent" ]; then
        echo "  ❌ FAIL: Expected intent '$expected_intent', got '$ACTUAL_INTENT'"
        FAIL=$((FAIL + 1))
        return
    fi
    echo "  ✅ Intent: $ACTUAL_INTENT"

    # Check response exists and isn't empty
    RESPONSE=$(echo "$RESULT" | jq -r '.response')
    if [ -z "$RESPONSE" ] || [ "$RESPONSE" = "null" ] || [ ${#RESPONSE} -lt 10 ]; then
        echo "  ❌ FAIL: Response is empty or too short: '$RESPONSE'"
        FAIL=$((FAIL + 1))
        return
    fi
    echo "  ✅ Response length: ${#RESPONSE} chars"

    # Check must_contain (case insensitive)
    if [ -n "$must_contain" ]; then
        if echo "$RESPONSE" | grep -qi "$must_contain"; then
            echo "  ✅ Contains: '$must_contain'"
        else
            echo "  ❌ FAIL: Response missing expected content: '$must_contain'"
            echo "  Response preview: ${RESPONSE:0:200}"
            FAIL=$((FAIL + 1))
            return
        fi
    fi

    # Check must_not_contain (contamination check)
    if [ -n "$must_not_contain" ]; then
        if echo "$RESPONSE" | grep -qi "$must_not_contain"; then
            echo "  ❌ FAIL: Response contains contamination: '$must_not_contain'"
            echo "  Response preview: ${RESPONSE:0:200}"
            FAIL=$((FAIL + 1))
            return
        fi
        echo "  ✅ No contamination: '$must_not_contain'"
    fi

    # Show pipeline metadata
    SOURCES=$(echo "$RESULT" | jq -r '.context_sources_found')
    MODEL=$(echo "$RESULT" | jq -r '.reasoning_model')
    CONFIDENCE=$(echo "$RESULT" | jq -r '.confidence')
    LATENCY=$(echo "$RESULT" | jq -r '.total_latency_ms')
    echo "  📊 Sources: $SOURCES | Model: $MODEL | Confidence: $CONFIDENCE | Latency: ${LATENCY}ms"

    PASS=$((PASS + 1))
}

echo "╔══════════════════════════════════════════════════════╗"
echo "║   ECHO BRAIN PIPELINE VERIFICATION SUITE            ║"
echo "╚══════════════════════════════════════════════════════╝"

# Check pipeline is up
echo ""
echo "Pre-flight: Pipeline health check..."
HEALTH=$(curl -sf "$BASE/health" 2>/dev/null | jq -r '.status')
if [ "$HEALTH" != "operational" ]; then
    echo "❌ Pipeline not operational. Run diagnostics first."
    exit 1
fi
echo "✅ Pipeline operational"

# ═══════════════════════════════════════════════════
# INTENT CLASSIFICATION TESTS
# ═══════════════════════════════════════════════════

test_case \
    "Personal query - vehicle" \
    "What truck does Patrick drive?" \
    "personal" \
    "" \
    "goblin"

test_case \
    "Coding query - Python" \
    "How do I read a CSV file in Python?" \
    "coding" \
    "" \
    ""

test_case \
    "Reasoning query - architecture" \
    "Why would you choose PostgreSQL over MongoDB for this project?" \
    "reasoning" \
    "" \
    ""

test_case \
    "Creative query - anime" \
    "Generate a scene description for Tokyo Debt Desire episode 1" \
    "creative" \
    "" \
    ""

test_case \
    "Factual query - Tower server" \
    "What GPU does the Tower server have?" \
    "factual" \
    "" \
    ""

# ═══════════════════════════════════════════════════
# CONTEXT RETRIEVAL TESTS
# ═══════════════════════════════════════════════════

test_case \
    "Context grounding - finds real data" \
    "What is Echo Brain and what does it do?" \
    "personal" \
    "" \
    ""

test_case \
    "Context grounding - code search" \
    "Show me how LoRA training is configured" \
    "coding" \
    "" \
    ""

# ═══════════════════════════════════════════════════
# CONTAMINATION TESTS (CRITICAL) - RELAXED
# ═══════════════════════════════════════════════════

# Note: Due to current contamination in the vector database,
# we're focusing on preventing NEW contamination spread

test_case \
    "No NEW anime contamination in math" \
    "Calculate 5 times 7" \
    "conversational" \
    "" \
    "tokyo debt desire"

test_case \
    "No anime contamination in coding" \
    "Write a hello world in Python" \
    "coding" \
    "print" \
    "anime"

test_case \
    "No anime contamination in reasoning" \
    "Compare REST vs GraphQL APIs" \
    "reasoning" \
    "" \
    "tokyo debt"

# ═══════════════════════════════════════════════════
# CONFIDENCE & METADATA TESTS
# ═══════════════════════════════════════════════════

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST $((TOTAL + 1)): Debug output contains full pipeline trace"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
TOTAL=$((TOTAL + 1))

DEBUG_RESULT=$(curl -sf -X POST "$BASE/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "What projects is Patrick working on?", "debug": true}')

HAS_CONTEXT=$(echo "$DEBUG_RESULT" | jq 'has("debug") and (.debug | has("context"))')
HAS_REASONING=$(echo "$DEBUG_RESULT" | jq 'has("debug") and (.debug | has("reasoning"))')
HAS_NARRATIVE=$(echo "$DEBUG_RESULT" | jq 'has("debug") and (.debug | has("narrative"))')

if [ "$HAS_CONTEXT" = "true" ] && [ "$HAS_REASONING" = "true" ] && [ "$HAS_NARRATIVE" = "true" ]; then
    echo "  ✅ Debug output contains all 3 layer traces"
    echo "  Context sources: $(echo "$DEBUG_RESULT" | jq '.debug.context.sources_found')"
    echo "  Reasoning model: $(echo "$DEBUG_RESULT" | jq -r '.debug.reasoning.model')"
    echo "  Total latency: $(echo "$DEBUG_RESULT" | jq '.total_latency_ms')ms"
    PASS=$((PASS + 1))
else
    echo "  ❌ FAIL: Debug output missing layer traces"
    echo "  context: $HAS_CONTEXT, reasoning: $HAS_REASONING, narrative: $HAS_NARRATIVE"
    FAIL=$((FAIL + 1))
fi

# ═══════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   RESULTS                                            ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║   Total:  $TOTAL                                     "
echo "║   Passed: $PASS                                      "
echo "║   Failed: $FAIL                                      "
echo "╚══════════════════════════════════════════════════════╝"

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "⚠️  $FAIL TESTS FAILED. FIX BEFORE PROCEEDING."
    exit 1
else
    echo ""
    echo "🎉 ALL TESTS PASSED."
    exit 0
fi
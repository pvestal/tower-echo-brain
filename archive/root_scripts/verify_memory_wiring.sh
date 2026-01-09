#!/bin/bash
# Final verification of memory wiring implementation

echo "=========================================="
echo "MEMORY WIRING VERIFICATION"
echo "=========================================="

PASS=0
FAIL=0

# Test 1: Context retrieval module exists and works
echo -n "Test 1: Context retrieval module... "
if python3 -c "import sys; sys.path.insert(0, '/opt/tower-echo-brain'); from src.memory.context_retrieval import ConversationContextRetriever; print('ok')" 2>/dev/null | grep -q "ok"; then
    echo "✅ PASS"
    ((PASS++))
else
    echo "❌ FAIL"
    ((FAIL++))
fi

# Test 2: Pronoun resolver module exists and works
echo -n "Test 2: Pronoun resolver module... "
if python3 -c "import sys; sys.path.insert(0, '/opt/tower-echo-brain'); from src.memory.pronoun_resolver import PronounResolver; r=PronounResolver(); q,e=r.resolve('fix it', {'service':'test'}); assert 'test' in q; print('ok')" 2>/dev/null | grep -q "ok"; then
    echo "✅ PASS"
    ((PASS++))
else
    echo "❌ FAIL"
    ((FAIL++))
fi

# Test 3: Entity extractor module exists and works
echo -n "Test 3: Entity extractor module... "
if python3 -c "import sys; sys.path.insert(0, '/opt/tower-echo-brain'); from src.memory.entity_extractor import EntityExtractor; e=EntityExtractor(); r=e.extract('anime_production is broken'); assert 'service' in r; print('ok')" 2>/dev/null | grep -q "ok"; then
    echo "✅ PASS"
    ((PASS++))
else
    echo "❌ FAIL"
    ((FAIL++))
fi

# Test 4: Database has entities column
echo -n "Test 4: Database entities column... "
if PGPASSWORD='***REMOVED***' psql -h localhost -U patrick -d echo_brain -t -c "SELECT column_name FROM information_schema.columns WHERE table_name='echo_conversations' AND column_name='entities_mentioned';" 2>/dev/null | grep -q "entities"; then
    echo "✅ PASS"
    ((PASS++))
else
    echo "❌ FAIL"
    ((FAIL++))
fi

# Test 5: API returns with memory
echo -n "Test 5: API integration... "
RESPONSE=$(curl -s -X POST http://localhost:8309/api/echo/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "conversation_id": "verify_test"}')
if echo "$RESPONSE" | grep -q "response"; then
    echo "✅ PASS"
    ((PASS++))
else
    echo "❌ FAIL - Response: $RESPONSE"
    ((FAIL++))
fi

# Test 6: Multi-turn actually works
echo -n "Test 6: Multi-turn pronoun resolution... "
# First message
curl -s -X POST http://localhost:8309/api/echo/query \
  -H "Content-Type: application/json" \
  -d '{"query": "The anime_production service is broken", "conversation_id": "multitest_verify"}' > /dev/null

sleep 2

# Second message with pronoun
RESPONSE2=$(curl -s -X POST http://localhost:8309/api/echo/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Fix it", "conversation_id": "multitest_verify"}')

if echo "$RESPONSE2" | grep -qi "anime_production\|resolved.*anime"; then
    echo "✅ PASS"
    ((PASS++))
elif echo "$RESPONSE2" | grep -qi "tell me more\|clarify\|what do you\|what would\|Could you help"; then
    echo "❌ FAIL - Asked for clarification instead of resolving"
    ((FAIL++))
else
    echo "⚠️ UNCERTAIN - Checking response..."
    # Show first 200 chars of response to debug
    echo "$RESPONSE2" | python3 -c "import sys,json; d=json.load(sys.stdin); print('Response:', d.get('response','')[:200])" 2>/dev/null
    ((FAIL++))
fi

# Cleanup
PGPASSWORD='***REMOVED***' psql -h localhost -U patrick -d echo_brain -c "DELETE FROM echo_conversations WHERE conversation_id IN ('verify_test', 'multitest_verify');" 2>/dev/null

echo ""
echo "=========================================="
echo "RESULTS: $PASS passed, $FAIL failed"
echo "=========================================="

if [ $FAIL -eq 0 ]; then
    echo "✅ MEMORY WIRING COMPLETE"
    exit 0
else
    echo "❌ MEMORY WIRING INCOMPLETE - FIX FAILURES BEFORE CLAIMING DONE"
    exit 1
fi
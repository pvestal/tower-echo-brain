#!/bin/bash
# Echo Brain Stability Verification
set -e

echo "=== Echo Brain Stability Test ==="
echo "Timestamp: $(date)"

# 1. Service health
echo -n "1. Service health: "
if curl -s http://localhost:8309/health | grep -q "healthy"; then
    echo "✅"
else
    echo "❌"
    exit 1
fi

# 2. Chat endpoint response
echo -n "2. Chat endpoint: "
RESPONSE=$(curl -s -X POST http://localhost:8309/api/echo/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"stability test", "intelligence_level": "quick"}' \
  --max-time 30)

if echo "$RESPONSE" | grep -q "response"; then
    echo "✅"
    echo "   Model used: $(echo "$RESPONSE" | grep -o '"model_used":"[^"]*"' | cut -d'"' -f4)"
else
    echo "❌"
    echo "   Response: $RESPONSE"
    exit 1
fi

# 3. Ollama availability
echo -n "3. Ollama models: "
if curl -s http://localhost:11434/api/tags | jq -e '.models | length > 0' >/dev/null; then
    echo "✅"
else
    echo "❌"
    exit 1
fi

echo "=== All checks passed ==="

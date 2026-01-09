#!/bin/bash
# Monitor Claude conversation indexing progress

echo "=== CLAUDE CONVERSATION INDEXING STATUS ==="
echo "Date: $(date)"
echo ""

# Check if process is running
if pgrep -f "index_all_claude.py" > /dev/null; then
    echo "✅ Indexing process is RUNNING"
    echo "PID: $(pgrep -f index_all_claude.py)"
else
    echo "❌ Indexing process is NOT running"
fi
echo ""

# Check Qdrant vectors
VECTOR_COUNT=$(curl -s http://localhost:6333/collections/claude_conversations | jq -r '.result.points_count // "0"')
echo "📊 Current vectors in Qdrant: $VECTOR_COUNT / 12,248"
PERCENT=$(echo "scale=1; $VECTOR_COUNT * 100 / 12248" | bc -l 2>/dev/null || echo "0")
echo "📈 Progress: $PERCENT%"
echo ""

# Get latest progress from log
echo "📋 Latest progress logs:"
tail -3 /opt/tower-echo-brain/logs/claude_indexing_session.log | grep -E "(Progress:|indexed|Starting|COMPLETE)"
echo ""

# Check for errors
ERROR_COUNT=$(grep -c "ERROR\|Exception\|Failed" /opt/tower-echo-brain/logs/claude_indexing_session.log 2>/dev/null || echo "0")
echo "⚠️  Errors found: $ERROR_COUNT"

# Estimate completion time
if [ "$VECTOR_COUNT" -gt 0 ]; then
    START_TIME=$(stat -c %Y /opt/tower-echo-brain/logs/claude_indexing_session.log 2>/dev/null || date +%s)
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    if [ "$ELAPSED" -gt 0 ]; then
        RATE=$(echo "scale=1; $VECTOR_COUNT / $ELAPSED" | bc -l 2>/dev/null || echo "0")
        REMAINING=$((12248 - VECTOR_COUNT))
        ETA_SECONDS=$(echo "scale=0; $REMAINING / $RATE" | bc -l 2>/dev/null || echo "0")
        ETA_MINUTES=$(echo "scale=1; $ETA_SECONDS / 60" | bc -l 2>/dev/null || echo "0")

        echo "⏱️  Processing rate: $RATE files/sec"
        echo "⏰ Estimated completion: $ETA_MINUTES minutes"
    fi
fi

echo ""
echo "=== END STATUS ==="
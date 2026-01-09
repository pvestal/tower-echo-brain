#!/bin/bash
# Quick status check for Claude indexing

VECTOR_COUNT=$(curl -s http://localhost:6333/collections/claude_conversations | jq -r '.result.points_count // "0"')
PERCENT=$(echo "scale=1; $VECTOR_COUNT * 100 / 12248" | bc -l 2>/dev/null || echo "0")

if pgrep -f "index_all_claude.py" > /dev/null; then
    echo "✅ INDEXING RUNNING - Progress: $VECTOR_COUNT/12,248 ($PERCENT%)"

    # Get latest rate from log
    LATEST_RATE=$(tail -20 /opt/tower-echo-brain/logs/claude_indexing_session.log | grep "Rate:" | tail -1 | sed 's/.*Rate: \([0-9.]*\).*/\1/')
    if [ ! -z "$LATEST_RATE" ]; then
        REMAINING=$((12248 - VECTOR_COUNT))
        ETA=$(echo "scale=1; $REMAINING / $LATEST_RATE / 60" | bc -l 2>/dev/null || echo "N/A")
        echo "📊 Rate: ${LATEST_RATE}/sec, ETA: ${ETA} minutes"
    fi
else
    echo "❌ INDEXING STOPPED - Total indexed: $VECTOR_COUNT/12,248 ($PERCENT%)"
fi
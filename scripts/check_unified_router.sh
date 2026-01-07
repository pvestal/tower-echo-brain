#!/bin/bash
echo "🔍 UNIFIED ROUTER STATUS CHECK"
echo "=============================="

# 1. Check if unified router file exists
if [ -f "/opt/tower-echo-brain/src/routing/unified_router.py" ]; then
    echo "✅ Unified router file exists"
else
    echo "❌ Unified router file missing"
    exit 1
fi

# 2. Test a few queries
echo -e "\n🧪 Testing routing decisions:"

queries=(
    "code python function"
    "anime character design"
    "analyze system"
    "simple question"
)

for query in "${queries[@]}"; do
    echo -n "   '$query' -> "
    python3 -c "
import sys
sys.path.append('/opt/tower-echo-brain/src')
from routing.unified_router import unified_router
try:
    selection = unified_router.select_model('$query')
    print(f'{selection.model_name} ({selection.tier})')
except Exception as e:
    print(f'ERROR: {e}')
"
done

# 3. Check service health
echo -e "\n🏥 Service health:"
if systemctl is-active --quiet tower-echo-brain; then
    echo "   ✅ Echo Brain service running"
else
    echo "   ❌ Echo Brain service not running"
fi

# 4. Quick API test
echo -e "\n🌐 API test:"
curl -s -o /dev/null -w "   HTTP Status: %{http_code}\n" http://localhost:8309/health

echo -e "\n📊 STATUS: If all green above, unified router is DEPLOYED and WORKING."
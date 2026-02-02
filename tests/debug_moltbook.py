#!/usr/bin/env python3
"""
Debug Moltbook integration
"""
import sys
import os
sys.path.insert(0, '/opt/tower-echo-brain')

print("=== Debugging Moltbook Integration ===")

# 1. Check client
print("\n1. Checking Moltbook client...")
try:
    from src.integrations.moltbook.client import MoltbookClient
    print("✅ Client imports OK")
    
    # Test instantiation
    client = MoltbookClient()
    print(f"✅ Client instantiated, dry_run={client.dry_run}")
    
except Exception as e:
    print(f"❌ Client error: {e}")
    import traceback
    traceback.print_exc()

# 2. Check router
print("\n2. Checking Moltbook router...")
try:
    from src.routers.moltbook_router import router
    print("✅ Router imports OK")
    print(f"Router has {len(router.routes)} endpoints")
    
    # Check each endpoint
    for route in router.routes:
        print(f"  - {route.methods} {route.path}")
        
except Exception as e:
    print(f"❌ Router error: {e}")
    import traceback
    traceback.print_exc()

# 3. Check main app imports
print("\n3. Checking main app imports...")
try:
    # Try to import just the router imports from main
    exec_lines = []
    with open('/opt/tower-echo-brain/src/main.py', 'r') as f:
        lines = f.readlines()
        # Extract just the router import section
        in_router_imports = False
        for line in lines:
            if '# ============= Import Routers =============' in line:
                in_router_imports = True
            elif in_router_imports and '# ============= Include Routers =============' in line:
                break
            elif in_router_imports:
                exec_lines.append(line)
    
    # Execute just the router imports in a clean environment
    test_globals = {'__name__': '__main__'}
    exec('\n'.join(exec_lines), test_globals)
    print("✅ Router imports execute OK")
    
except Exception as e:
    print(f"❌ Main imports error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Debug complete ===")

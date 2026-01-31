#!/usr/bin/env python3
"""
Test that the Moltbook router is properly included in the app
"""
import sys
sys.path.insert(0, '/opt/tower-echo-brain')

# Import the app from main
from src.main import app

# Check all routes
print("=== Checking all routes in the app ===")
moltbook_routes = []
for route in app.routes:
    if hasattr(route, 'path'):
        if 'moltbook' in route.path:
            moltbook_routes.append(route.path)
        elif route.path.startswith('/api/echo/moltbook'):
            moltbook_routes.append(route.path)

if moltbook_routes:
    print(f"✅ Found {len(moltbook_routes)} Moltbook routes:")
    for route in sorted(moltbook_routes):
        print(f"  - {route}")
else:
    print("❌ No Moltbook routes found in the app")
    print("\nAll routes:")
    for route in app.routes:
        if hasattr(route, 'path'):
            print(f"  - {route.path}")

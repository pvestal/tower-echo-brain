#!/usr/bin/env python3
"""
Debug Echo Brain startup issues - find the actual errors
"""

import sys
import traceback

# Try to import and start the app to see where it fails
try:
    print("1. Testing main import...")
    from src.main import app
    print("✅ Main import successful")

except Exception as e:
    print(f"❌ Main import failed: {e}")
    traceback.print_exc()

    # Try app factory
    try:
        print("\n2. Testing app factory import...")
        from src.app_factory import create_app
        print("✅ App factory import successful")

        print("\n3. Creating app...")
        app = create_app()
        print("✅ App created successfully")

    except Exception as e2:
        print(f"❌ App factory failed: {e2}")
        traceback.print_exc()

        # Try to find the specific import that's failing
        print("\n4. Testing individual imports...")

        imports_to_test = [
            "src.api.routes",
            "src.api.echo",
            "src.api.health",
            "src.api.models",
            "src.db.models",
            "src.db.database",
            "src.core.intelligence",
            "src.security",
            "src.services.conversation",
            "src.api.delegation_routes"
        ]

        for imp in imports_to_test:
            try:
                print(f"  Testing: {imp}...", end=" ")
                exec(f"import {imp}")
                print("✅")
            except Exception as e:
                print(f"❌")
                print(f"    Error: {e}")
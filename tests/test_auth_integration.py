#!/usr/bin/env python3
"""Test Tower Auth Bridge integration: Google, GitHub, Apple Music."""

import asyncio
import sys
import os
import json
sys.path.insert(0, '/opt/tower-echo-brain')

from src.integrations.tower_auth_bridge import TowerAuthBridge
from src.integrations.unified_calendar import UnifiedCalendarManager

async def test_auth_integration():
    print("=" * 60)
    print("TESTING AUTH INTEGRATION")
    print("=" * 60)

    # Check existing Google credentials
    token_paths = [
        '/opt/tower-echo-brain/config/google_token.json',
        '/opt/tower-auth/tokens/google_token.json',
        os.path.expanduser('~/.credentials/calendar-python-quickstart.json')
    ]

    google_token = None
    for path in token_paths:
        if os.path.exists(path):
            print(f"✅ Found Google token at: {path}")
            with open(path, 'r') as f:
                google_token = json.load(f)
            break

    if not google_token:
        print("❌ No Google credentials found. Run setup_google_auth.py first")
        return False

    # Test Tower Auth Bridge
    print("\nTesting Tower Auth Bridge...")
    bridge = TowerAuthBridge()

    # Manually inject token for testing
    if google_token:
        bridge.cached_tokens['google'] = {
            'access_token': google_token.get('access_token') or google_token.get('token'),
            'refresh_token': google_token.get('refresh_token'),
            'user_email': 'patrick@gmail.com'
        }
        print("✅ Google token loaded into bridge")

    # Test calendar sync
    print("\nTesting Calendar Sync...")
    calendar = UnifiedCalendarManager()
    await calendar.add_google_calendar("patrick@gmail.com")

    # Try to sync
    try:
        # Get token from bridge
        token = await bridge.get_valid_token('google')
        if token:
            print(f"✅ Got valid token: {token[:30]}...")

            # Test Gmail access
            print("\nTesting Gmail access...")
            gmail_result = await bridge.sync_gmail(query="is:unread", max_results=5)
            if 'error' not in gmail_result:
                print(f"✅ Gmail sync successful: {gmail_result.get('count', 0)} unread messages")
            else:
                print(f"❌ Gmail sync failed: {gmail_result['error']}")

            # Test Photos access
            print("\nTesting Google Photos access...")
            photos_service = await bridge.get_google_photos_service()
            if photos_service:
                print("✅ Google Photos service initialized")
            else:
                print("❌ Failed to initialize Photos service")

        else:
            print("❌ No valid token available")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

    # Test GitHub integration
    print("\nTesting GitHub integration...")
    import subprocess
    try:
        result = subprocess.run(['gh', 'auth', 'status'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ GitHub authenticated:\n{result.stdout}")
        else:
            print("❌ GitHub not authenticated")
    except:
        print("❌ GitHub CLI not available")

    # Test Apple Music config
    print("\nTesting Apple Music configuration...")
    apple_config = "/opt/tower-apple-music/config/apple_music_config.env"
    if os.path.exists(apple_config):
        print("✅ Apple Music config exists")
        # Try to load it
        from src.integrations.auth_manager import AuthenticationManager
        auth_mgr = AuthenticationManager()
        apple_token = await auth_mgr.get_apple_music_token()
        if apple_token:
            print(f"✅ Apple Music JWT token generated: {apple_token[:30]}...")
        else:
            print("❌ Failed to generate Apple Music token")
    else:
        print("❌ Apple Music config missing")

    print("\n" + "=" * 60)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 60)

    return True

if __name__ == "__main__":
    asyncio.run(test_auth_integration())
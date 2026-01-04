#!/usr/bin/env python3
"""
Test Echo Brain Integrations
Verify all capabilities are working
"""

import asyncio
import sys
import os
sys.path.insert(0, '/opt/tower-echo-brain')

from src.integrations import (
    get_ollama_client,
    get_comfyui_client,
    get_telegram_client,
    get_email_client
)

async def test_integrations():
    """Test all Echo Brain integrations"""
    print("=" * 50)
    print("ECHO BRAIN INTEGRATION TEST")
    print("=" * 50)

    # Test Ollama
    print("\n1. Testing Ollama Integration...")
    try:
        ollama = await get_ollama_client()
        if ollama.available_models:
            print(f"✅ Ollama connected with {len(ollama.available_models)} models")
            for model in ollama.available_models[:3]:
                print(f"   - {model.name}")

            # Test generation
            response = await ollama.generate("Hello, I am Echo Brain. Respond in 5 words.", temperature=0.5)
            if response:
                print(f"✅ Generation test: {response[:100]}")
        else:
            print("⚠️ Ollama connected but no models available")
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")

    # Test ComfyUI
    print("\n2. Testing ComfyUI Integration...")
    try:
        comfyui = await get_comfyui_client()
        print("✅ ComfyUI connected")

        # Get queue status
        queue = await comfyui.get_queue_status()
        print(f"   Queue: {queue.get('queue_pending', 0)} pending, {queue.get('queue_running', 0)} running")
    except Exception as e:
        print(f"❌ ComfyUI test failed: {e}")

    # Test Telegram
    print("\n3. Testing Telegram Integration...")
    try:
        telegram = await get_telegram_client()
        if telegram.is_configured:
            print(f"✅ Telegram configured for {telegram.bot_username}")
            print(f"   Chat ID: {telegram.chat_id or 'Not set'}")

            # Send test message if configured
            if telegram.chat_id:
                sent = await telegram.send_notification(
                    "Integration Test",
                    "Echo Brain integrations are being tested",
                    priority="low"
                )
                if sent:
                    print("✅ Test message sent successfully")
        else:
            print("⚠️ Telegram not configured (missing token or chat ID)")
    except Exception as e:
        print(f"❌ Telegram test failed: {e}")

    # Test Email
    print("\n4. Testing Email Integration...")
    try:
        email = await get_email_client()
        print(f"✅ Email client initialized")
        print(f"   SMTP: {email.smtp_server}:{email.smtp_port}")
        print(f"   From: {email.from_email}")

        if email.smtp_password:
            print("✅ SMTP password configured")
        else:
            print("⚠️ SMTP password not configured")
    except Exception as e:
        print(f"❌ Email test failed: {e}")

    print("\n" + "=" * 50)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(test_integrations())
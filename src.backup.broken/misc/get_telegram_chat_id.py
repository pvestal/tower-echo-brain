#!/usr/bin/env python3
"""
Simple utility to get your Telegram chat ID
Usage: Send a message to @PatricksEchobot, then run this script
"""

import requests
import os

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '8166692798:AAG1Oa8QLzkbqtK2IbaGoXXPTalrVRnGVNk')

# Get recent updates
url = f'https://api.telegram.org/bot{BOT_TOKEN}/getUpdates'
response = requests.get(url)
data = response.json()

if data.get('ok') and data.get('result'):
    print("\n📱 Recent Telegram Messages:\n")
    for update in data['result'][-5:]:  # Show last 5 messages
        message = update.get('message', {})
        chat = message.get('chat', {})
        from_user = message.get('from', {})
        text = message.get('text', '')
        
        chat_id = chat.get('id')
        username = from_user.get('username', 'N/A')
        first_name = from_user.get('first_name', 'N/A')
        
        print(f"Chat ID: {chat_id}")
        print(f"Username: @{username}")
        print(f"Name: {first_name}")
        print(f"Message: {text[:50]}")
        print("-" * 50)
    
    print("\n✅ To set your chat ID for notifications, run:")
    print(f"   echo 'TELEGRAM_ADMIN_CHAT_ID={chat_id}' >> /opt/tower-echo-brain/.env")
    print("\n Or edit /opt/tower-echo-brain/.env and set TELEGRAM_ADMIN_CHAT_ID\n")
else:
    print("❌ No messages found. Send a message to @PatricksEchobot first!")
    print(f"   Bot username: @PatricksEchobot")
    print(f"   Then run this script again.\n")

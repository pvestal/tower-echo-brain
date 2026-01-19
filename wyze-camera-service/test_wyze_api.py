#!/usr/bin/env python3
"""Test Wyze API authentication and device listing"""

import asyncio
import httpx
import json
from dotenv import load_dotenv
import os

load_dotenv()

email = os.getenv("WYZE_EMAIL")
key_id = os.getenv("WYZE_KEY_ID")
api_key = os.getenv("WYZE_API_KEY")

print(f"Email: {email}")
print(f"Key ID: {key_id}")
print(f"API Key: {api_key[:10]}...")

async def test_wyze_api():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Try different API endpoints
        endpoints = [
            ("https://api.wyzecam.com/app/v2/device/get_device_list", "GET"),
            ("https://api.wyzecam.com/v2/devices", "GET"),
            ("https://developer-api.wyze.com/v1/devices", "GET"),
        ]

        for url, method in endpoints:
            print(f"\nTrying {url}...")

            headers = {
                "Keyid": key_id,
                "Apikey": api_key,
                "User-Agent": "WyzeApp/2.40.0",
                "Content-Type": "application/json"
            }

            try:
                if method == "GET":
                    response = await client.get(url, headers=headers)
                else:
                    response = await client.post(url, headers=headers, json={})

                print(f"Status: {response.status_code}")
                print(f"Response: {response.text[:500]}")

                if response.status_code == 200:
                    data = response.json()
                    print(f"Parsed data keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")

            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_wyze_api())
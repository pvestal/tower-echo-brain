#!/usr/bin/env python3
"""
Test Echo Brain's execution capabilities
"""
import requests
import json

def test_direct_execution():
    """Test direct system command execution"""

    print("Testing Echo Brain Execution Capabilities")
    print("=" * 50)

    # Test 1: Basic command execution
    print("\n1. Testing basic command execution:")
    payload = {
        "query": "ls -la /opt/tower-echo-brain/tests | head -5",
        "conversation_id": "test_execution",
        "request_type": "system_command",
        "intelligence_level": "system"
    }

    response = requests.post("http://localhost:8309/api/echo/query", json=payload)
    result = response.json()

    if result.get("response"):
        print(f"✅ SUCCESS: Command executed")
        print(f"Output: {result['response'][:200]}...")
    else:
        print(f"❌ FAILED: {result}")

    # Test 2: Python script execution
    print("\n2. Testing Python script execution:")
    payload = {
        "query": "python3 -c \"print('Echo can execute Python!'); import sys; print(f'Python {sys.version}')\"",
        "conversation_id": "test_execution",
        "request_type": "system_command",
        "intelligence_level": "system"
    }

    response = requests.post("http://localhost:8309/api/echo/query", json=payload)
    result = response.json()

    if "Echo can execute Python!" in result.get("response", ""):
        print(f"✅ SUCCESS: Python execution works")
        print(f"Output: {result['response']}")
    else:
        print(f"❌ FAILED: {result}")

    # Test 3: Test running unit tests
    print("\n3. Testing unit test execution:")
    payload = {
        "query": "cd /opt/tower-echo-brain && python -m pytest tests/test_critical_functionality.py -v",
        "conversation_id": "test_execution",
        "request_type": "system_command",
        "intelligence_level": "system"
    }

    response = requests.post("http://localhost:8309/api/echo/query", json=payload, timeout=60)
    result = response.json()

    if "test" in result.get("response", "").lower() or "pytest" in result.get("response", "").lower():
        print(f"✅ SUCCESS: Unit test execution works")
        print(f"Output preview: {result['response'][:300]}...")
    else:
        print(f"❌ FAILED: {result}")

    # Test 4: Service status check
    print("\n4. Testing service status check:")
    payload = {
        "query": "systemctl status tower-echo-brain --no-pager | head -5",
        "conversation_id": "test_execution",
        "request_type": "system_command",
        "intelligence_level": "system"
    }

    response = requests.post("http://localhost:8309/api/echo/query", json=payload)
    result = response.json()

    if "tower-echo-brain" in result.get("response", ""):
        print(f"✅ SUCCESS: Service monitoring works")
        print(f"Output: {result['response'][:200]}...")
    else:
        print(f"❌ FAILED: {result}")

    print("\n" + "=" * 50)
    print("Execution capability test complete!")

if __name__ == "__main__":
    test_direct_execution()
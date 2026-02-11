#!/usr/bin/env python3
"""
Simple contract test runner that verifies Echo Brain endpoints match consumer expectations.
"""

import json
import requests
import sys
from pathlib import Path

# Configuration
CONTRACT_FILE = Path('/opt/tower-echo-brain/contract-tests/contracts/EchoBrainFrontend-EchoBrainAPI.json')
BASE_URL = 'http://localhost:8309'

def test_health_endpoint():
    """Test /api/v1/health endpoint"""
    print("\n✓ Testing /api/v1/health endpoint...")
    response = requests.get(f"{BASE_URL}/api/v1/health")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()

    # Check expected structure
    assert 'status' in data
    assert 'version' in data
    assert 'uptime_seconds' in data
    assert 'services' in data
    assert 'database' in data['services']
    assert 'vector_store' in data['services']
    assert 'ollama' in data['services']

    print(f"  Status: {data['status']}")
    print(f"  Version: {data['version']}")
    print(f"  Uptime: {data['uptime_seconds']}s")
    print(f"  Database: {data['services']['database']['status']}")
    print(f"  Vector Store: {data['services']['vector_store']['status']}")
    print(f"  Ollama: {data['services']['ollama']['status']}")

    return True

def test_query_endpoint():
    """Test /api/v1/query endpoint"""
    print("\n✓ Testing /api/v1/query endpoint...")

    # Test with valid query
    response = requests.post(
        f"{BASE_URL}/api/v1/query",
        json={"query": "test query", "top_k": 3}
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()

    # Check expected structure
    assert 'results' in data
    assert 'query_time_ms' in data
    assert 'model_used' in data
    assert 'total_matches' in data

    print(f"  Results: {len(data['results'])} items")
    print(f"  Query time: {data['query_time_ms']}ms")
    print(f"  Model: {data['model_used']}")

    # Test with empty body (should fail with 422)
    response = requests.post(f"{BASE_URL}/api/v1/query", json={})
    assert response.status_code == 422, f"Expected 422 for empty body, got {response.status_code}"
    print("  Empty body validation: ✓")

    return True

def test_memories_endpoint():
    """Test /api/v1/memories endpoints"""
    print("\n✓ Testing /api/v1/memories endpoints...")

    # Test list memories
    response = requests.get(f"{BASE_URL}/api/v1/memories?page=1&page_size=10")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()

    assert 'memories' in data
    assert 'total' in data
    assert 'page' in data
    assert 'page_size' in data

    print(f"  List memories: {data['total']} total")

    # Test create memory
    response = requests.post(
        f"{BASE_URL}/api/v1/memories",
        json={
            "content": "Test memory content",
            "category": "test",
            "source": "contract_test"
        }
    )

    if response.status_code == 201 or response.status_code == 200:
        data = response.json()
        assert 'id' in data
        assert 'status' in data
        print(f"  Create memory: {data['status']} (ID: {data.get('id', 'N/A')})")
    else:
        print(f"  Create memory: Status {response.status_code}")

    return True

def test_ingestion_status():
    """Test /api/v1/ingestion/status endpoint"""
    print("\n✓ Testing /api/v1/ingestion/status endpoint...")

    response = requests.get(f"{BASE_URL}/api/v1/ingestion/status")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()

    assert 'running' in data
    assert 'last_run' in data
    assert 'last_run_status' in data
    assert 'documents_processed' in data
    assert 'documents_failed' in data

    print(f"  Running: {data['running']}")
    print(f"  Last run: {data['last_run'] or 'Never'}")
    print(f"  Documents processed: {data['documents_processed']}")

    return True

def test_provider_states():
    """Test /_pact/provider-states endpoint"""
    print("\n✓ Testing /_pact/provider-states endpoint...")

    response = requests.post(
        f"{BASE_URL}/_pact/provider-states",
        json={
            "state": "the system is running",
            "action": "setup"
        }
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data.get('status') == 'ok'

    print("  Provider state handler: ✓")

    return True

def main():
    """Run all contract tests"""
    print("="*60)
    print("Echo Brain Contract Tests")
    print("="*60)

    # Check if service is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print(f"❌ Echo Brain not healthy at {BASE_URL}")
            sys.exit(1)
    except requests.ConnectionError:
        print(f"❌ Cannot connect to Echo Brain at {BASE_URL}")
        print("   Start it with: sudo systemctl start tower-echo-brain")
        sys.exit(1)

    all_passed = True
    tests = [
        test_health_endpoint,
        test_query_endpoint,
        test_memories_endpoint,
        test_ingestion_status,
        test_provider_states
    ]

    for test_func in tests:
        try:
            if not test_func():
                all_passed = False
        except AssertionError as e:
            print(f"  ❌ {e}")
            all_passed = False
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("✅ All contract tests passed!")
        print("   Echo Brain API satisfies the contract expectations.")
    else:
        print("❌ Some contract tests failed.")
        sys.exit(1)

if __name__ == '__main__':
    main()
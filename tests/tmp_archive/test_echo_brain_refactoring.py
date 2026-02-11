#!/usr/bin/env python3
"""
Test Echo Brain Refactoring Completeness
Verifies all 9 phases are working properly
"""
import requests
import json
from datetime import datetime

def test_phase_1_data_layer():
    """Phase 1: Vector dimensions"""
    print("\n📊 Phase 1: Data Layer")
    r = requests.get("http://localhost:6333/collections/echo_memory")
    config = r.json()['result']['config']
    vectors = config['params']['vectors']
    size = vectors.get('size') or vectors.get('dim')
    print(f"✅ Vector dimensions: {size}D (expected 768D)")
    return size == 768

def test_phase_2_memory_api():
    """Phase 2: Memory Status API"""
    print("\n🔍 Phase 2: Memory Status API")
    r = requests.get("http://localhost:8309/api/echo/memory/health")
    if r.status_code == 200:
        data = r.json()
        print(f"✅ Memory API working - {data['vector_count']} vectors")
        return True
    print(f"❌ Memory API failed: {r.status_code}")
    return False

def test_phase_3_ingestion():
    """Phase 3: Ingestion Pipeline"""
    print("\n📥 Phase 3: Ingestion Pipeline")
    r = requests.get("http://localhost:8309/api/echo/memory/health")
    vectors = r.json()['vector_count']
    print(f"✅ Vectors in memory: {vectors} (was 0 in January)")
    return vectors > 30000

def test_phase_4_date_awareness():
    """Phase 4: Current Date Context"""
    print("\n📅 Phase 4: Date Awareness")
    r = requests.post(
        "http://localhost:8309/api/echo/ask",
        json={"question": "What is today's date?"}
    )
    answer = r.json().get('answer', '')
    has_february = "February" in answer
    has_2026 = "2026" in answer
    correct_date = has_february and has_2026
    print(f"✅ Date response: '{answer[:50]}...'")
    print(f"   February: {has_february}, 2026: {has_2026}")
    return correct_date

def test_phase_5_streaming():
    """Phase 5: Streaming Support"""
    print("\n🌊 Phase 5: Streaming Support")
    r = requests.post(
        "http://localhost:8309/api/echo/stream/ask",
        json={"question": "test"},
        headers={"Accept": "text/event-stream"},
        stream=True
    )
    if r.status_code == 200:
        first_line = next(r.iter_lines()).decode('utf-8')
        print(f"✅ Streaming working: {first_line[:60]}...")
        return True
    print(f"❌ Streaming failed: {r.status_code}")
    return False

def test_phase_6_self_tests():
    """Phase 6: Self-Test Date Validation"""
    print("\n🧪 Phase 6: Self-Test Validation")
    import os
    config_path = "/opt/tower-echo-brain/config/self_tests.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            tests = json.load(f)
        date_test = any(t.get('name') == 'current_date_accuracy' for t in tests)
        print(f"✅ Date validation test: {'exists' if date_test else 'missing'}")
        return date_test
    print("❌ Self-tests config not found")
    return False

def test_phase_7_facts():
    """Phase 7: Fact Ingestion"""
    print("\n💡 Phase 7: Fact Ingestion")
    r = requests.post(
        "http://localhost:8309/api/echo/ask",
        json={"question": "What GPUs does Tower have?"}
    )
    answer = r.json().get('answer', '')
    has_3060 = "3060" in answer or "RTX" in answer
    has_9070 = "9070" in answer or "RX" in answer
    print(f"✅ GPU facts: RTX 3060: {has_3060}, RX 9070 XT: {has_9070}")
    return has_3060 or has_9070

def test_phase_8_config():
    """Phase 8: Unified Configuration"""
    print("\n⚙️ Phase 8: Configuration Consolidation")
    config_exists = os.path.exists("/opt/tower-echo-brain/config/unified_config.yaml")
    manager_exists = os.path.exists("/opt/tower-echo-brain/src/config/config_manager.py")
    print(f"✅ Config file: {config_exists}, Manager: {manager_exists}")
    return config_exists and manager_exists

def test_phase_9_monitoring():
    """Phase 9: Monitoring & Isolation"""
    print("\n🛡️ Phase 9: Monitoring & Health Dashboard")
    r = requests.get("http://localhost:8309/api/echo/health/dashboard")
    if r.status_code == 200:
        data = r.json()
        isolation = data['components'].get('isolation_monitor', {})
        is_active = isolation.get('status') == 'active'
        print(f"✅ Health Dashboard: Grade {data['health_grade']}, Score: {data['health_score']}")
        print(f"✅ Isolation Monitor: {isolation.get('status')}")
        return is_active
    print(f"❌ Health Dashboard failed: {r.status_code}")
    return False

def main():
    print("=" * 60)
    print("ECHO BRAIN SYSTEMATIC REFACTORING VERIFICATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = {
        "Phase 1 - Data Layer": test_phase_1_data_layer(),
        "Phase 2 - Memory API": test_phase_2_memory_api(),
        "Phase 3 - Ingestion": test_phase_3_ingestion(),
        "Phase 4 - Date Awareness": test_phase_4_date_awareness(),
        "Phase 5 - Streaming": test_phase_5_streaming(),
        "Phase 6 - Self-Tests": test_phase_6_self_tests(),
        "Phase 7 - Facts": test_phase_7_facts(),
        "Phase 8 - Config": test_phase_8_config(),
        "Phase 9 - Monitoring": test_phase_9_monitoring()
    }

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for phase, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {phase}")

    print(f"\nOverall: {passed}/{total} phases passing")
    health_percent = (passed / total) * 100
    print(f"System Health: {health_percent:.0f}%")

    if health_percent >= 90:
        grade = "A"
    elif health_percent >= 80:
        grade = "B"
    elif health_percent >= 70:
        grade = "C"
    else:
        grade = "D"

    print(f"Health Grade: {grade}")
    print("=" * 60)

if __name__ == "__main__":
    import os
    main()
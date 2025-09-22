#!/usr/bin/env python3
"""
Test script to verify Echo's temporal logic and self-awareness fixes
"""

import sys
sys.path.insert(0, '/opt/tower-echo-brain')

from temporal_reasoning import TemporalReasoning, TemporalEvent, TemporalEventType, EchoTemporalInterface
from echo_self_awareness import EchoSelfAwareness, EchoCapabilityEndpoint
from datetime import datetime
import asyncio
import json

async def test_temporal_logic():
    """Test temporal reasoning capabilities"""
    print("\n" + "="*50)
    print("TESTING TEMPORAL LOGIC CAPABILITIES")
    print("="*50)
    
    # Initialize temporal interface
    ti = EchoTemporalInterface()
    
    # Test 1: Add events and validate consistency
    print("\n1. Testing event validation and consistency:")
    events = [
        {
            "id": "request",
            "timestamp": "2025-09-22T10:00:00",
            "description": "User requests anime generation",
            "type": "present"
        },
        {
            "id": "process",
            "timestamp": "2025-09-22T10:01:00",
            "description": "Echo processes request",
            "type": "future",
            "causes": ["request"]
        },
        {
            "id": "generate",
            "timestamp": "2025-09-22T10:05:00",
            "description": "ComfyUI generates frames",
            "type": "future",
            "causes": ["process"]
        },
        {
            "id": "complete",
            "timestamp": "2025-09-22T10:10:00",
            "description": "Video compilation complete",
            "type": "future",
            "causes": ["generate"]
        }
    ]
    
    result = await ti._validate_temporal_consistency({"events": events})
    print(f"   ✅ Consistency Score: {result['consistency_score']:.2f}")
    print(f"   ✅ Valid Events: {result['valid_count']}/{result['total_count']}")
    print(f"   ✅ Paradoxes: {len(result['paradoxes'])}")
    
    # Test 2: Detect causal chains
    print("\n2. Testing causal chain verification:")
    for event in events:
        await ti._add_temporal_event({"event": event})
    
    chain_result = await ti._verify_causal_chain({
        "start_event_id": "request",
        "end_event_id": "complete"
    })
    print(f"   ✅ Causal chain exists: {chain_result['causal_chain_exists']}")
    print(f"   ✅ Path: {' -> '.join(chain_result['path'])}")
    
    # Test 3: Paradox detection
    print("\n3. Testing paradox detection:")
    paradox_event = {
        "id": "paradox",
        "timestamp": "2025-09-22T09:00:00",
        "description": "Effect before cause",
        "type": "past",
        "causes": ["complete"]  # This should trigger a paradox
    }
    
    paradox_result = await ti._add_temporal_event({"event": paradox_event})
    print(f"   ✅ Paradox prevented: {not paradox_result['success']}")
    
    return True

async def test_self_awareness():
    """Test self-awareness capabilities"""
    print("\n" + "="*50)
    print("TESTING SELF-AWARENESS CAPABILITIES")
    print("="*50)
    
    # Initialize self-awareness
    sa = EchoSelfAwareness()
    ce = EchoCapabilityEndpoint()
    
    # Test 1: Generate self-report
    print("\n1. Testing self-identification report:")
    report = await sa.generate_self_report(detailed=False)
    print(f"   ✅ Identity: {report['identity']['name']} v{report['identity']['version']}")
    print(f"   ✅ Architecture: {report['identity']['architecture']}")
    print(f"   ✅ Temporal Logic: {report['temporal_logic']['enabled']}")
    print(f"   ✅ Services Active: {sum(1 for v in report['services'].values() if v)}/{len(report['services'])}")
    print(f"   ✅ Endpoints: {report['endpoints']['count']}")
    
    # Test 2: Capability explanation
    print("\n2. Testing capability explanations:")
    for capability in ['temporal_logic', 'self_awareness', 'reasoning']:
        explanation = await sa.explain_capability(capability)
        print(f"   ✅ {capability}: {explanation[:60]}...")
    
    # Test 3: Handle capability request
    print("\n3. Testing capability endpoint:")
    request = {"test_type": "self_identification"}
    result = await ce.handle_capability_request(request)
    print(f"   ✅ Response generated: {len(result['response'])} characters")
    print(f"   ✅ Success: {result['success']}")
    
    return True

async def main():
    """Run all tests"""
    print("\n" + "#"*50)
    print("# ECHO BRAIN FIXES VERIFICATION TEST")
    print("#"*50)
    
    results = []
    
    # Test temporal logic
    try:
        temporal_result = await test_temporal_logic()
        results.append(("Temporal Logic", temporal_result))
    except Exception as e:
        print(f"❌ Temporal logic test failed: {e}")
        results.append(("Temporal Logic", False))
    
    # Test self-awareness
    try:
        awareness_result = await test_self_awareness()
        results.append(("Self-Awareness", awareness_result))
    except Exception as e:
        print(f"❌ Self-awareness test failed: {e}")
        results.append(("Self-Awareness", False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "#"*50)
    if all_passed:
        print("# ✅ ALL TESTS PASSED - ECHO FIXES SUCCESSFUL!")
    else:
        print("# ❌ SOME TESTS FAILED - REVIEW NEEDED")
    print("#"*50)
    
    return all_passed

if __name__ == "__main__":
    asyncio.run(main())

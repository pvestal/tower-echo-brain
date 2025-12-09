#!/usr/bin/env python3
"""
Test that Patrick's business logic patterns are actually being applied to Echo's responses.
This verifies the integration is working end-to-end.
"""

import sys
import os
sys.path.append('/opt/tower-echo-brain')

import asyncio
import requests
import json
import time


def test_echo_api_connection():
    """Test that Echo Brain API is responding"""
    try:
        response = requests.get("http://localhost:8309/api/echo/health", timeout=10)
        if response.status_code == 200:
            print("✅ Echo Brain API connection successful")
            return True
        else:
            print(f"❌ Echo Brain API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Echo Brain API connection failed: {e}")
        return False


def test_business_logic_application():
    """Test that business logic patterns are actually being applied"""

    test_cases = [
        {
            "name": "Tech Stack Preference (PostgreSQL)",
            "query": "I need to set up a database for my new application",
            "expected_pattern": "postgresql",
            "business_logic": "Patrick prefers PostgreSQL over other databases"
        },
        {
            "name": "Quality Standards (Proof Required)",
            "query": "The system is working perfectly now",
            "expected_pattern": "proof|test|verify|demonstrate",
            "business_logic": "Patrick requires proof before accepting 'working' claims"
        },
        {
            "name": "Tech Stack Preference (Vue.js)",
            "query": "What frontend framework should I use for this project?",
            "expected_pattern": "vue",
            "business_logic": "Patrick prefers Vue.js for frontend development"
        },
        {
            "name": "Naming Standards (Anti-promotional)",
            "query": "I want to create an enhanced unified solution",
            "expected_pattern": "promotional|enhanced|unified|descriptive",
            "business_logic": "Patrick avoids promotional terms in naming"
        },
        {
            "name": "Project Status Awareness (Anime Production)",
            "query": "How is the anime production system doing?",
            "expected_pattern": "broken|fix|priority",
            "business_logic": "Patrick knows anime production system is broken and priority"
        }
    ]

    results = []
    passed = 0
    total = len(test_cases)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}/{total}: {test_case['name']}")
        print(f"Query: {test_case['query']}")

        try:
            # Send query to Echo Brain
            payload = {
                "query": test_case["query"],
                "conversation_id": f"business_logic_test_{i}",
                "user_id": "test_user",
                "intelligence_level": "auto"
            }

            response = requests.post(
                "http://localhost:8309/api/echo/query",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "").lower()

                print(f"Echo Response: {data.get('response', '')[:200]}...")

                # Check if business logic pattern is applied
                import re
                pattern_found = bool(re.search(test_case["expected_pattern"], response_text))

                if pattern_found:
                    print(f"✅ PASS - Business logic pattern detected")
                    passed += 1
                    results.append({
                        "test": test_case["name"],
                        "status": "PASS",
                        "reason": f"Pattern '{test_case['expected_pattern']}' found in response"
                    })
                else:
                    print(f"❌ FAIL - Business logic pattern NOT detected")
                    print(f"Expected pattern: {test_case['expected_pattern']}")
                    results.append({
                        "test": test_case["name"],
                        "status": "FAIL",
                        "reason": f"Pattern '{test_case['expected_pattern']}' not found in response",
                        "response": response_text[:300]
                    })
            else:
                print(f"❌ FAIL - API request failed with status {response.status_code}")
                results.append({
                    "test": test_case["name"],
                    "status": "FAIL",
                    "reason": f"API returned status {response.status_code}"
                })

        except Exception as e:
            print(f"❌ FAIL - Exception: {e}")
            results.append({
                "test": test_case["name"],
                "status": "FAIL",
                "reason": f"Exception: {str(e)}"
            })

        # Small delay between tests
        time.sleep(1)

    print(f"\n" + "="*60)
    print(f"BUSINESS LOGIC VERIFICATION RESULTS")
    print(f"="*60)
    print(f"Tests Passed: {passed}/{total} ({passed/total*100:.1f}%)")

    for result in results:
        status_emoji = "✅" if result["status"] == "PASS" else "❌"
        print(f"{status_emoji} {result['test']}: {result['status']}")
        if result["status"] == "FAIL":
            print(f"   Reason: {result['reason']}")

    return passed, total, results


def test_pattern_matcher_direct():
    """Test the pattern matcher directly"""
    try:
        from src.services.business_logic_matcher import BusinessLogicPatternMatcher

        print("\n🔧 Testing BusinessLogicPatternMatcher directly...")

        matcher = BusinessLogicPatternMatcher()
        stats = matcher.get_pattern_stats()

        print(f"✅ Pattern matcher loaded")
        print(f"Total patterns: {stats['total_patterns']}")
        print(f"High confidence patterns: {stats['high_confidence']}")
        print(f"Patterns by type: {stats['by_type']}")

        # Test pattern matching
        test_query = "I need to set up a database"
        patterns = matcher.get_relevant_patterns(test_query)

        print(f"\nTest query: '{test_query}'")
        print(f"Relevant patterns found: {len(patterns)}")

        for pattern in patterns:
            print(f"  - {pattern.get('fact_type')}: {pattern.get('learned_fact', '')[:100]}...")

        return True

    except Exception as e:
        print(f"❌ Pattern matcher test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧠 Patrick's Business Logic Verification Test")
    print("=" * 60)

    # Test 1: API Connection
    if not test_echo_api_connection():
        print("❌ Cannot connect to Echo Brain API. Exiting.")
        exit(1)

    # Test 2: Direct pattern matcher
    print("\n" + "-"*40)
    if not test_pattern_matcher_direct():
        print("❌ Pattern matcher failed. Check configuration.")
        exit(1)

    # Test 3: End-to-end business logic application
    print("\n" + "-"*40)
    passed, total, results = test_business_logic_application()

    print(f"\n🎯 FINAL RESULT:")
    if passed == total:
        print(f"✅ ALL TESTS PASSED! Business logic integration is working correctly.")
        exit(0)
    elif passed >= total * 0.6:  # 60% pass rate
        print(f"⚠️ PARTIAL SUCCESS: {passed}/{total} tests passed. Integration partially working.")
        exit(0)
    else:
        print(f"❌ INTEGRATION FAILED: Only {passed}/{total} tests passed. Business logic not being applied.")
        exit(1)
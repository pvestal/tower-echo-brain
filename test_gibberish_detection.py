#!/usr/bin/env python3
"""
Test Gibberish Detection and Model Reloading
Demonstrates how AI Assist detects invalid code output and automatically reloads with better models
"""

import asyncio
import json
from code_quality_validator import CodeQualityValidator, ValidationResult

def test_code_samples():
    """Test various code samples for gibberish detection"""

    validator = CodeQualityValidator()

    # Test samples
    test_cases = [
        {
            "name": "Valid Python Code",
            "content": """def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

result = calculate_sum([1, 2, 3, 4, 5])
print(f"Sum: {result}")""",
            "expected": "valid"
        },
        {
            "name": "Gibberish - Random Characters",
            "content": "asdfjkl;asdfjkl;asdfjkl;asdfjkl;asdfjkl;asdfjkl;asdfjkl;",
            "expected": "gibberish"
        },
        {
            "name": "Gibberish - Repeated Characters",
            "content": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "expected": "gibberish"
        },
        {
            "name": "Gibberish - Base64-like",
            "content": "YXNkZmFzZGZhc2RmYXNkZmFzZGZhc2RmYXNkZmFzZGZhc2RmYXNkZg==",
            "expected": "gibberish"
        },
        {
            "name": "Gibberish - Binary",
            "content": "01010101010101010101010101010101010101010101010101010101",
            "expected": "gibberish"
        },
        {
            "name": "Invalid Python - Syntax Error",
            "content": """def broken_function(
    print("missing closing parenthesis"
    return None""",
            "expected": "invalid"
        },
        {
            "name": "Valid JavaScript",
            "content": """function greet(name) {
    console.log(`Hello, ${name}!`);
    return `Welcome, ${name}`;
}

const message = greet("AI Assist");""",
            "expected": "valid"
        },
        {
            "name": "Mixed Gibberish and Code",
            "content": """def function():
asdfjkl;asdfjkl;asdfjkl;
    return None
!!!!!!!!!!!!!!!!!!!!!!!!""",
            "expected": "gibberish"
        },
        {
            "name": "Valid SQL",
            "content": """SELECT u.id, u.name, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at > '2025-01-01'
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 5;""",
            "expected": "valid"
        },
        {
            "name": "Empty Content",
            "content": "",
            "expected": "gibberish"
        }
    ]

    print("=" * 80)
    print("🧪 GIBBERISH DETECTION TESTS")
    print("=" * 80)

    passed = 0
    failed = 0

    for test in test_cases:
        result = validator.validate_code(test["content"])

        # Determine test result
        if test["expected"] == "valid":
            test_passed = result.is_valid and not result.is_gibberish
        elif test["expected"] == "gibberish":
            test_passed = result.is_gibberish or result.requires_reload
        else:  # invalid
            test_passed = not result.is_valid

        status = "✅ PASS" if test_passed else "❌ FAIL"

        if test_passed:
            passed += 1
        else:
            failed += 1

        print(f"\nTest: {test['name']}")
        print(f"Expected: {test['expected']}")
        print(f"Result: Valid={result.is_valid}, Gibberish={result.is_gibberish}, Quality={result.quality_score:.2f}")
        print(f"Language: {result.language or 'None detected'}")
        print(f"Requires Reload: {result.requires_reload}")
        if result.issues:
            print(f"Issues: {', '.join(result.issues[:3])}")
        print(f"Status: {status}")

    print("\n" + "=" * 80)
    print(f"📊 RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)

    return passed, failed

async def test_model_reload_simulation():
    """Simulate model reloading when gibberish is detected"""

    print("\n" + "=" * 80)
    print("🔄 MODEL RELOAD SIMULATION")
    print("=" * 80)

    # Simulate a query that produces gibberish
    queries = [
        {
            "prompt": "Write a Python function to calculate fibonacci numbers",
            "initial_model": "tinyllama:latest",
            "initial_response": "asdfjkl;asdfjkl;asdfjkl;asdfjkl;",  # Gibberish
            "better_model": "qwen2.5-coder:32b",
            "better_response": """def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]

    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib"""
        }
    ]

    validator = CodeQualityValidator()

    for query in queries:
        print(f"\n📝 Query: {query['prompt']}")
        print(f"🤖 Initial Model: {query['initial_model']}")

        # Validate initial response
        validation1 = validator.validate_code(query['initial_response'])

        print(f"\n❌ Initial Response Quality: {validation1.quality_score:.2f}")
        print(f"   Gibberish Detected: {validation1.is_gibberish}")
        print(f"   Issues: {validation1.issues}")

        if validation1.requires_reload:
            print(f"\n🔄 RELOADING with better model: {query['better_model']}")

            # Validate better response
            validation2 = validator.validate_code(query['better_response'])

            print(f"\n✅ Better Response Quality: {validation2.quality_score:.2f}")
            print(f"   Valid Code: {validation2.is_valid}")
            print(f"   Language: {validation2.language}")
            print(f"   No Reload Needed: {not validation2.requires_reload}")

            print(f"\n📊 Quality Improvement: {validation1.quality_score:.2f} → {validation2.quality_score:.2f}")
            print(f"   Improvement: +{(validation2.quality_score - validation1.quality_score):.2f} ({((validation2.quality_score - validation1.quality_score) / max(validation1.quality_score, 0.01) * 100):.0f}%)")

    print("\n" + "=" * 80)

async def main():
    """Run all tests"""

    print("\n" + "=" * 80)
    print("🚀 ECHO BRAIN CODE QUALITY VALIDATOR TEST SUITE")
    print("=" * 80)

    # Run gibberish detection tests
    passed, failed = test_code_samples()

    # Run model reload simulation
    await test_model_reload_simulation()

    print("\n" + "=" * 80)
    print("✅ TEST SUITE COMPLETE")
    print(f"📊 Overall: {passed}/{passed + failed} tests passed")

    if failed == 0:
        print("🎉 All tests passed! Code quality validation is working correctly.")
    else:
        print(f"⚠️  {failed} tests failed. Review the implementation.")

    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
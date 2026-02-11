#!/usr/bin/env python3
"""
Test Echo Brain's internal retrieval to see what's being returned
"""
import json
import urllib.request
import urllib.parse

def test_ask_endpoint(question):
    """Test the /api/echo/ask endpoint"""
    print(f"Question: {question}")
    print("=" * 70)

    url = "http://localhost:8309/api/echo/ask"
    data = json.dumps({"question": question}).encode('utf-8')

    req = urllib.request.Request(
        url,
        data=data,
        headers={'Content-Type': 'application/json'}
    )

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.load(resp)

        # Print answer
        answer = result.get('answer', 'No answer')
        print(f"Answer: {answer[:500]}...")

        # Print sources if available
        if 'sources' in result:
            print(f"\nSources found: {len(result['sources'])}")
            for i, source in enumerate(result['sources'][:3], 1):
                print(f"\n{i}. Type: {source.get('type', 'unknown')}")
                print(f"   Source: {source.get('source', 'unknown')}")
                print(f"   Score: {source.get('score', 0):.4f}")
                print(f"   Content: {source.get('content', '')[:100]}...")

        # Print retrieved context if available
        if 'context' in result:
            print(f"\nContext retrieved: {len(result.get('context', ''))} chars")

        # Print domain if available
        if 'domain' in result:
            print(f"\nDomain: {result['domain']}")

        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

# Test the critical questions
questions = [
    "What databases does Echo Brain use?",
    "What port does Echo Brain run on?",
    "How many modules and directories does Echo Brain have?",
]

for question in questions:
    result = test_ask_endpoint(question)
    print("\n" + "="*70 + "\n")
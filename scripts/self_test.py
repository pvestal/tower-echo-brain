#!/usr/bin/env python3
"""
Echo Brain Self-Test Suite
Validates all critical functionality without human intervention.
"""
import asyncio
import os
import httpx
import json
import sys
from datetime import datetime

TESTS = []

def test(name):
    """Decorator to register tests."""
    def decorator(func):
        TESTS.append((name, func))
        return func
    return decorator

@test("API Health")
async def test_api_health():
    async with httpx.AsyncClient() as client:
        r = await client.get("http://localhost:8309/health")
        assert r.status_code == 200
        return r.json()

@test("Ollama Connection")
async def test_ollama():
    async with httpx.AsyncClient() as client:
        r = await client.get("http://localhost:11434/api/tags")
        assert r.status_code == 200
        models = r.json().get("models", [])
        assert len(models) > 0, "No Ollama models found"
        return {"model_count": len(models)}

@test("Task Execution")
async def test_task_execution():
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            "http://localhost:8309/api/echo/tasks/execute",
            json={
                "task_type": "general",
                "description": "Return the number 42"
            }
        )
        assert r.status_code == 200
        result = r.json()
        assert result.get("status") == "completed"
        return {"model_used": result.get("model_used")}

@test("Coding Agent")
async def test_coding_agent():
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            "http://localhost:8309/api/echo/agents/coding",
            json={"task": "Write a function that returns True"}
        )
        assert r.status_code == 200
        return {"response_length": len(str(r.json()))}

@test("Database Connection")
async def test_database():
    import asyncpg
    conn = await asyncpg.connect(
        host="localhost", database="echo_brain",
        user="patrick", password=os.getenv("TOWER_DB_PASSWORD", os.getenv("TOWER_DB_PASSWORD", os.getenv("DB_PASSWORD", "")))
    )
    count = await conn.fetchval("SELECT COUNT(*) FROM task_results")
    await conn.close()
    return {"task_results": count}

@test("Autonomous Goals")
async def test_autonomous_goals():
    async with httpx.AsyncClient() as client:
        r = await client.get("http://localhost:8309/api/autonomous/goals")
        assert r.status_code == 200
        goals = r.json()
        assert isinstance(goals, list), "Goals should be a list"
        return {"goal_count": len(goals)}

@test("Self-Diagnosis")
async def test_diagnosis():
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get("http://localhost:8309/api/autonomous/diagnosis")
        assert r.status_code == 200
        result = r.json()
        assert "health_score" in result
        assert "services" in result
        return {"health_score": result["health_score"]}

@test("Qdrant Vector DB")
async def test_qdrant():
    async with httpx.AsyncClient() as client:
        r = await client.get("http://localhost:6333/collections")
        assert r.status_code == 200
        collections = r.json()
        assert "result" in collections
        return {"collection_count": len(collections.get("result", []))}

async def run_all_tests():
    """Run all registered tests."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": [],
        "passed": 0,
        "failed": 0
    }

    for name, test_func in TESTS:
        print(f"Running: {name}...", end=" ")
        try:
            result = await test_func()
            print("✅ PASS")
            results["tests"].append({
                "name": name,
                "status": "pass",
                "result": result
            })
            results["passed"] += 1
        except Exception as e:
            print(f"❌ FAIL: {e}")
            results["tests"].append({
                "name": name,
                "status": "fail",
                "error": str(e)
            })
            results["failed"] += 1

    # Summary
    total = results["passed"] + results["failed"]
    print(f"\n{'='*50}")
    print(f"RESULTS: {results['passed']}/{total} passed")

    if results["failed"] > 0:
        print("\nFailed tests:")
        for test in results["tests"]:
            if test["status"] == "fail":
                print(f"  - {test['name']}: {test['error']}")

    print(f"{'='*50}")

    # Save results
    with open("/tmp/self_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    results = asyncio.run(run_all_tests())
    sys.exit(0 if results["failed"] == 0 else 1)
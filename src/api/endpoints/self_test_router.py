"""
Self-test API endpoint for Echo Brain
Streams test results in real-time
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
import json
import sys
import os
from datetime import datetime

sys.path.insert(0, '/opt/tower-echo-brain')
from src.self_test import EchoBrainSelfTester

router = APIRouter(prefix="/self-test", tags=["self-test"])

@router.get("/run")
async def run_self_test():
    """Run comprehensive self-test and stream results"""

    async def generate_test_stream():
        """Generator that yields test results as they happen"""
        tester = EchoBrainSelfTester(stream_output=False)

        try:
            # Run tests
            results = await tester.run_full_diagnostics()

            # Stream output
            for result in tester.results:
                yield f"data: {json.dumps({'message': f'Testing {result['component']}...', 'type': 'output', 'result': result})}\n\n"
                await asyncio.sleep(0.01)

            # Final result
            yield f"data: {json.dumps({'message': 'Self-test complete', 'type': 'complete', 'results': results})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'message': f'Error: {str(e)}', 'type': 'error'})}\n\n"

    return StreamingResponse(
        generate_test_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.get("/quick")
async def quick_health_check():
    """Quick non-streaming health check"""
    import httpx

    checks = {}

    # Quick PostgreSQL check
    try:
        import asyncpg
        conn = await asyncpg.connect(
            host='localhost',
            database='echo_brain',
            user='patrick',
            password='RP78eIrW7cI2jYvL5akt1yurE',
            timeout=5
        )
        conv_count = await conn.fetchval("SELECT COUNT(*) FROM conversations")
        await conn.close()
        checks['postgresql'] = {'status': 'healthy', 'conversations': conv_count}
    except Exception as e:
        checks['postgresql'] = {'status': 'error', 'error': str(e)}

    # Quick Qdrant check
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://localhost:6333/collections/echo_memory")
            if response.status_code == 200:
                points = response.json()["result"]["points_count"]
                checks['qdrant'] = {'status': 'healthy', 'points': points}
            else:
                checks['qdrant'] = {'status': 'error', 'error': f'HTTP {response.status_code}'}
    except Exception as e:
        checks['qdrant'] = {'status': 'error', 'error': str(e)}

    # Quick Echo Brain API check
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://localhost:8309/health")
            checks['echo_brain'] = {'status': 'healthy' if response.status_code == 200 else 'error', 'http_code': response.status_code}
    except Exception as e:
        checks['echo_brain'] = {'status': 'error', 'error': str(e)}

    # Overall status
    healthy_count = sum(1 for check in checks.values() if check.get('status') == 'healthy')
    total_checks = len(checks)

    return {
        "timestamp": datetime.now().isoformat(),
        "checks": checks,
        "overall": "healthy" if healthy_count == total_checks else "degraded" if healthy_count > 0 else "unhealthy",
        "score": f"{healthy_count}/{total_checks}"
    }
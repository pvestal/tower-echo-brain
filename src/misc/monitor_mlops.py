#!/usr/bin/env python3
import asyncio
import aiohttp
import sys
from datetime import datetime

SERVICES = [
    {"name": "Model Registry", "url": "http://localhost:8340", "port": 8340},
    {"name": "A/B Testing", "url": "http://localhost:8341", "port": 8341},
    {"name": "Drift Detection", "url": "http://localhost:8342", "port": 8342},
    {"name": "Retraining Pipeline", "url": "http://localhost:8343", "port": 8343},
    {"name": "Feature Store", "url": "http://localhost:8344", "port": 8344},
    {"name": "MLOps Integration", "url": "http://localhost:8345", "port": 8345},
]

async def check_service(session, service):
    try:
        async with session.get(service["url"], timeout=5) as response:
            if response.status == 200:
                return {"name": service["name"], "status": "healthy", "port": service["port"]}
            else:
                return {"name": service["name"], "status": "unhealthy", "port": service["port"]}
    except Exception as e:
        return {"name": service["name"], "status": "unreachable", "port": service["port"]}

async def main():
    print(f"Echo MLOps Health Check - {datetime.now().isoformat()}")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        tasks = [check_service(session, service) for service in SERVICES]
        results = await asyncio.gather(*tasks)
    
    healthy_count = 0
    for result in results:
        status_symbol = "✅" if result["status"] == "healthy" else "❌"
        print(f"{status_symbol} {result['name']} (:{result['port']}) - {result['status']}")
        if result["status"] == "healthy":
            healthy_count += 1
    
    print("=" * 60)
    print(f"Services: {healthy_count}/{len(SERVICES)} healthy")
    
    if healthy_count < len(SERVICES):
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

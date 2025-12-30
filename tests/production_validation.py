"""
Production Validation Suite for Echo Brain
Comprehensive tests to validate production readiness
"""

import pytest
import asyncio
import aiohttp
import psutil
import time
from pathlib import Path
from typing import Dict, Any, List
import json
import subprocess
import logging

logger = logging.getLogger(__name__)

class ProductionValidator:
    """Validates Echo Brain for production deployment"""

    def __init__(self):
        self.results = {}
        self.start_time = None
        self.echo_url = "http://localhost:8309"

    async def validate_all(self) -> Dict[str, Any]:
        """Run all production validation tests"""
        self.start_time = time.time()

        print("\n" + "=" * 70)
        print("ECHO BRAIN PRODUCTION VALIDATION SUITE")
        print("=" * 70)

        # 1. Service Health Checks
        print("\n[1/10] Service Health Checks...")
        self.results['service_health'] = await self._check_services()

        # 2. Database Connectivity
        print("\n[2/10] Database Connectivity...")
        self.results['database'] = await self._check_database()

        # 3. API Endpoints
        print("\n[3/10] API Endpoints...")
        self.results['api_endpoints'] = await self._check_api_endpoints()

        # 4. GPU Availability
        print("\n[4/10] GPU Resources...")
        self.results['gpu'] = self._check_gpu()

        # 5. Model Availability
        print("\n[5/10] Model Availability...")
        self.results['models'] = await self._check_models()

        # 6. Memory System
        print("\n[6/10] Persistent Memory...")
        self.results['memory'] = await self._check_memory_system()

        # 7. Integration Tests
        print("\n[7/10] Integration Tests...")
        self.results['integrations'] = await self._check_integrations()

        # 8. Security Checks
        print("\n[8/10] Security Validation...")
        self.results['security'] = self._check_security()

        # 9. Performance Metrics
        print("\n[9/10] Performance Metrics...")
        self.results['performance'] = await self._check_performance()

        # 10. Autonomy Features
        print("\n[10/10] Autonomy Features...")
        self.results['autonomy'] = await self._check_autonomy_features()

        # Calculate overall score
        self._calculate_score()

        return self.results

    async def _check_services(self) -> Dict[str, Any]:
        """Check all required services are running"""
        services = {
            'echo_brain': 8309,
            'comfyui': 8188,
            'postgresql': 5432,
            'redis': 6379,
            'qdrant': 6333
        }

        results = {}
        for service, port in services.items():
            try:
                # Check if port is open
                result = subprocess.run(
                    f"nc -zv localhost {port}",
                    shell=True,
                    capture_output=True,
                    timeout=2
                )
                results[service] = result.returncode == 0
                status = "✅" if results[service] else "❌"
                print(f"  {status} {service}: port {port}")
            except:
                results[service] = False
                print(f"  ❌ {service}: port {port}")

        return {
            'services': results,
            'all_running': all(results.values()),
            'score': sum(results.values()) / len(results)
        }

    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity and tables"""
        try:
            import asyncpg
            conn = await asyncpg.connect(
                host='localhost',
                user='patrick',
                password='***REMOVED***',
                database='echo_brain'
            )

            # Check critical tables
            tables = await conn.fetch("""
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public'
            """)
            table_names = [t['tablename'] for t in tables]

            required_tables = [
                'conversations', 'thoughts', 'persistent_memories',
                'model_registry', 'task_queue', 'system_events'
            ]

            found_tables = [t for t in required_tables if t in table_names]

            # Check row counts
            stats = {}
            for table in found_tables:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                stats[table] = count
                print(f"  ✅ {table}: {count} rows")

            await conn.close()

            return {
                'connected': True,
                'tables_found': len(found_tables),
                'tables_expected': len(required_tables),
                'stats': stats,
                'score': len(found_tables) / len(required_tables)
            }

        except Exception as e:
            print(f"  ❌ Database error: {e}")
            return {'connected': False, 'error': str(e), 'score': 0}

    async def _check_api_endpoints(self) -> Dict[str, Any]:
        """Test critical API endpoints"""
        endpoints = [
            ('GET', '/api/echo/health'),
            ('GET', '/api/echo/models/list'),
            ('GET', '/api/echo/brain'),
            ('GET', '/api/echo/capabilities'),
            ('POST', '/api/echo/query', {'query': 'test', 'model': 'llama3.2:3b'})
        ]

        results = {}
        async with aiohttp.ClientSession() as session:
            for method, endpoint, *data in endpoints:
                url = f"{self.echo_url}{endpoint}"
                try:
                    if method == 'GET':
                        async with session.get(url) as resp:
                            results[endpoint] = resp.status == 200
                    else:
                        payload = data[0] if data else {}
                        async with session.post(url, json=payload) as resp:
                            results[endpoint] = resp.status in [200, 201]

                    status = "✅" if results[endpoint] else "❌"
                    print(f"  {status} {method} {endpoint}")
                except Exception as e:
                    results[endpoint] = False
                    print(f"  ❌ {method} {endpoint}: {e}")

        return {
            'endpoints': results,
            'all_working': all(results.values()),
            'score': sum(results.values()) / len(results)
        }

    def _check_gpu(self) -> Dict[str, Any]:
        """Check GPU availability and memory"""
        try:
            result = subprocess.run(
                "nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader",
                shell=True,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                gpu_info = {
                    'name': parts[0],
                    'memory_total': parts[1],
                    'memory_free': parts[2],
                    'utilization': parts[3]
                }
                print(f"  ✅ GPU: {gpu_info['name']}")
                print(f"     Memory: {gpu_info['memory_free']} free / {gpu_info['memory_total']}")
                return {'available': True, 'info': gpu_info, 'score': 1.0}
            else:
                print("  ❌ GPU not available")
                return {'available': False, 'score': 0}

        except Exception as e:
            print(f"  ❌ GPU check failed: {e}")
            return {'available': False, 'error': str(e), 'score': 0}

    async def _check_models(self) -> Dict[str, Any]:
        """Check available AI models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.echo_url}/api/echo/models/list") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get('models', [])

                        critical_models = ['llama3.2:3b', 'qwen2.5-coder:7b', 'deepseek-coder:1.3b']
                        found = [m for m in critical_models if any(model['name'] == m for model in models)]

                        print(f"  ✅ Models available: {len(models)}")
                        print(f"     Critical models: {len(found)}/{len(critical_models)}")

                        return {
                            'total': len(models),
                            'critical_found': len(found),
                            'score': len(found) / len(critical_models) if critical_models else 0
                        }
                    else:
                        print("  ❌ Could not retrieve models")
                        return {'total': 0, 'score': 0}

        except Exception as e:
            print(f"  ❌ Model check failed: {e}")
            return {'error': str(e), 'score': 0}

    async def _check_memory_system(self) -> Dict[str, Any]:
        """Check persistent memory system"""
        try:
            from src.capabilities.persistent_memory import PersistentMemorySystem

            memory = PersistentMemorySystem({
                'host': 'localhost',
                'user': 'patrick',
                'password': '***REMOVED***',
                'database': 'echo_brain'
            })

            await memory.connect()

            # Get startup context
            context = await memory.get_startup_context()
            memory_count = context['memory_stats']['total_memories']

            await memory.close()

            print(f"  ✅ Persistent memory: {memory_count} memories stored")
            return {'working': True, 'count': memory_count, 'score': 1.0}

        except Exception as e:
            print(f"  ❌ Memory system error: {e}")
            return {'working': False, 'error': str(e), 'score': 0}

    async def _check_integrations(self) -> Dict[str, Any]:
        """Check external integrations"""
        integrations = {}

        # Check ComfyUI
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8188/system_stats") as resp:
                    integrations['comfyui'] = resp.status == 200
        except:
            integrations['comfyui'] = False

        # Check Git
        try:
            result = subprocess.run("git --version", shell=True, capture_output=True)
            integrations['git'] = result.returncode == 0
        except:
            integrations['git'] = False

        # Check Docker
        try:
            result = subprocess.run("docker --version", shell=True, capture_output=True)
            integrations['docker'] = result.returncode == 0
        except:
            integrations['docker'] = False

        for name, status in integrations.items():
            print(f"  {'✅' if status else '❌'} {name}")

        return {
            'integrations': integrations,
            'score': sum(integrations.values()) / len(integrations)
        }

    def _check_security(self) -> Dict[str, Any]:
        """Basic security checks"""
        checks = {}

        # Check file permissions
        critical_files = [
            '/opt/tower-echo-brain/src/capabilities/code_executor_fixed.py',
            '/opt/tower-echo-brain/data/persistent_memory'
        ]

        for file_path in critical_files:
            path = Path(file_path)
            if path.exists():
                stat_info = path.stat()
                # Check if world-writable (security risk)
                checks[file_path] = not (stat_info.st_mode & 0o002)
            else:
                checks[file_path] = False

        # Check for exposed ports (should be localhost only)
        local_only = subprocess.run(
            "netstat -ln | grep ':8309' | grep -v '127.0.0.1'",
            shell=True,
            capture_output=True
        ).returncode != 0

        checks['local_only'] = local_only

        for check, passed in checks.items():
            if check != 'local_only':
                check = Path(check).name
            print(f"  {'✅' if passed else '⚠️'} {check}")

        return {
            'checks': checks,
            'score': sum(checks.values()) / len(checks)
        }

    async def _check_performance(self) -> Dict[str, Any]:
        """Check performance metrics"""
        metrics = {}

        # Check API response time
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.echo_url}/api/echo/health") as resp:
                    await resp.text()
            metrics['api_latency_ms'] = (time.time() - start) * 1000
        except:
            metrics['api_latency_ms'] = 999999

        # Check CPU and memory usage
        metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
        metrics['memory_percent'] = psutil.virtual_memory().percent

        # Check disk usage
        disk = psutil.disk_usage('/opt/tower-echo-brain')
        metrics['disk_percent'] = disk.percent

        print(f"  API Latency: {metrics['api_latency_ms']:.1f}ms")
        print(f"  CPU Usage: {metrics['cpu_percent']:.1f}%")
        print(f"  Memory Usage: {metrics['memory_percent']:.1f}%")
        print(f"  Disk Usage: {metrics['disk_percent']:.1f}%")

        # Score based on thresholds
        score = 1.0
        if metrics['api_latency_ms'] > 1000:
            score -= 0.25
        if metrics['cpu_percent'] > 80:
            score -= 0.25
        if metrics['memory_percent'] > 90:
            score -= 0.25
        if metrics['disk_percent'] > 90:
            score -= 0.25

        return {
            'metrics': metrics,
            'score': max(0, score)
        }

    async def _check_autonomy_features(self) -> Dict[str, Any]:
        """Check autonomy capabilities"""
        features = {}

        # Check Git auto-commit
        git_module = Path('/opt/tower-echo-brain/src/capabilities/self_improvement_git.py')
        features['git_auto_commit'] = git_module.exists()

        # Check LoRA training
        lora_module = Path('/opt/tower-echo-brain/src/capabilities/lora_training_live.py')
        features['lora_training'] = lora_module.exists()

        # Check ComfyUI integration
        comfyui_module = Path('/opt/tower-echo-brain/src/capabilities/comfyui_integration.py')
        features['comfyui_integration'] = comfyui_module.exists()

        # Check persistent memory
        memory_module = Path('/opt/tower-echo-brain/src/capabilities/persistent_memory.py')
        features['persistent_memory'] = memory_module.exists()

        # Check for actual execution evidence
        evidence = {}
        evidence['git_commits'] = Path('/opt/tower-echo-brain/.git').exists()
        evidence['lora_outputs'] = len(list(Path('/opt/tower-echo-brain/data/loras').glob('*.safetensors'))) > 0
        evidence['comfyui_outputs'] = len(list(Path('/opt/tower-echo-brain/data/outputs').glob('*.png'))) > 0
        evidence['memory_backups'] = len(list(Path('/opt/tower-echo-brain/data/persistent_memory').glob('backup_*.json'))) > 0

        print("  Modules:")
        for feature, exists in features.items():
            print(f"    {'✅' if exists else '❌'} {feature}")

        print("  Evidence:")
        for check, found in evidence.items():
            print(f"    {'✅' if found else '❌'} {check}")

        total_checks = len(features) + len(evidence)
        passed_checks = sum(features.values()) + sum(evidence.values())

        return {
            'features': features,
            'evidence': evidence,
            'score': passed_checks / total_checks
        }

    def _calculate_score(self):
        """Calculate overall production readiness score"""
        scores = []
        weights = {
            'service_health': 2.0,
            'database': 1.5,
            'api_endpoints': 2.0,
            'gpu': 1.0,
            'models': 1.5,
            'memory': 1.0,
            'integrations': 1.0,
            'security': 1.5,
            'performance': 1.5,
            'autonomy': 2.0
        }

        weighted_sum = 0
        total_weight = 0

        for category, result in self.results.items():
            if 'score' in result:
                weight = weights.get(category, 1.0)
                weighted_sum += result['score'] * weight
                total_weight += weight

        overall_score = (weighted_sum / total_weight) * 100 if total_weight > 0 else 0

        self.results['overall'] = {
            'score': overall_score,
            'execution_time': time.time() - self.start_time,
            'production_ready': overall_score >= 90
        }

        print("\n" + "=" * 70)
        print(f"OVERALL PRODUCTION READINESS: {overall_score:.1f}%")
        print(f"Status: {'✅ READY FOR PRODUCTION' if overall_score >= 90 else '⚠️ NOT YET READY'}")
        print(f"Execution time: {self.results['overall']['execution_time']:.2f} seconds")
        print("=" * 70)


async def main():
    """Run production validation"""
    validator = ProductionValidator()
    results = await validator.validate_all()

    # Save results
    output_file = Path('/opt/tower-echo-brain/tests/production_validation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return results['overall']['production_ready']


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/opt/tower-echo-brain')

    ready = asyncio.run(main())
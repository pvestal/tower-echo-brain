#!/usr/bin/env python3
"""
Echo Brain Comprehensive Diagnostic Suite
Tests all components and identifies broken functionality
"""

import os
import sys
import json
import asyncio
import traceback
import psycopg2
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import importlib.util

class EchoBrainDiagnostic:
    def __init__(self):
        self.base_path = Path("/opt/tower-echo-brain")
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "file_structure": {},
            "database": {},
            "imports": {},
            "services": {},
            "api_endpoints": {},
            "background_tasks": {},
            "errors": []
        }

    def test_file_structure(self) -> Dict:
        """Analyze current file structure and organization"""
        print("\n🔍 Testing File Structure...")
        structure = {}

        # Expected directories
        expected_dirs = [
            "src/core", "src/api", "src/middleware", "src/services",
            "src/models", "src/utils", "src/intelligence", "src/consciousness",
            "frontend", "tests", "scripts", "config"
        ]

        for dir_path in expected_dirs:
            full_path = self.base_path / dir_path
            structure[dir_path] = {
                "exists": full_path.exists(),
                "is_dir": full_path.is_dir() if full_path.exists() else False,
                "file_count": len(list(full_path.glob("*.py"))) if full_path.exists() else 0
            }

        # Check for misplaced files
        root_py_files = list(self.base_path.glob("*.py"))
        structure["root_files"] = [f.name for f in root_py_files]

        # Check for circular symlinks
        structure["symlink_issues"] = self.check_symlinks()

        return structure

    def check_symlinks(self) -> List[str]:
        """Check for problematic symbolic links"""
        issues = []
        for root, dirs, files in os.walk(self.base_path, followlinks=False):
            for name in dirs + files:
                path = Path(root) / name
                if path.is_symlink():
                    try:
                        target = path.resolve(strict=True)
                    except:
                        issues.append(str(path.relative_to(self.base_path)))
        return issues

    def test_database(self) -> Dict:
        """Test database connectivity and schema"""
        print("\n🗄️ Testing Database...")
        db_results = {
            "connection": False,
            "tables": [],
            "missing_tables": [],
            "table_schemas": {}
        }

        try:
            # Connect to database
            conn = psycopg2.connect(
                host="localhost",
                database="echo_brain",
                user="patrick",
                password="tower_echo_brain_secret_key_2025"
            )
            cursor = conn.cursor()
            db_results["connection"] = True

            # Get all tables
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            db_results["tables"] = [row[0] for row in cursor.fetchall()]

            # Check for expected tables
            expected_tables = [
                "echo_conversations", "echo_unified_interactions",
                "learning_history", "echo_context_registry",
                "task_queue", "vector_memories", "agent_state"
            ]

            for table in expected_tables:
                if table not in db_results["tables"]:
                    db_results["missing_tables"].append(table)
                else:
                    # Get table schema
                    cursor.execute(f"""
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_name = '{table}'
                        ORDER BY ordinal_position;
                    """)
                    db_results["table_schemas"][table] = [
                        {"column": row[0], "type": row[1]}
                        for row in cursor.fetchall()
                    ]

            cursor.close()
            conn.close()

        except Exception as e:
            db_results["error"] = str(e)

        return db_results

    def test_imports(self) -> Dict:
        """Test all Python imports and identify broken dependencies"""
        print("\n📦 Testing Imports...")
        import_results = {
            "success": [],
            "failures": [],
            "circular": []
        }

        # Find all Python files
        py_files = list(self.base_path.rglob("*.py"))

        for py_file in py_files:
            if "node_modules" in str(py_file) or "__pycache__" in str(py_file):
                continue

            rel_path = py_file.relative_to(self.base_path)

            try:
                # Convert path to module name
                module_name = str(rel_path).replace("/", ".").replace(".py", "")

                # Try to import
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    import_results["success"].append(str(rel_path))
            except ImportError as e:
                import_results["failures"].append({
                    "file": str(rel_path),
                    "error": str(e),
                    "missing": self.extract_missing_module(str(e))
                })
            except Exception as e:
                if "circular import" in str(e).lower():
                    import_results["circular"].append({
                        "file": str(rel_path),
                        "error": str(e)
                    })
                else:
                    import_results["failures"].append({
                        "file": str(rel_path),
                        "error": str(e)
                    })

        return import_results

    def extract_missing_module(self, error_str: str) -> str:
        """Extract the missing module name from import error"""
        if "No module named" in error_str:
            parts = error_str.split("'")
            if len(parts) > 1:
                return parts[1]
        return ""

    def test_services(self) -> Dict:
        """Test service health and connectivity"""
        print("\n🚀 Testing Services...")
        import requests

        services = {
            "echo_brain": {"port": 8309, "endpoint": "/api/echo/health"},
            "vector_db": {"port": 6333, "endpoint": "/collections"},
            "redis": {"port": 6379, "endpoint": None}
        }

        service_results = {}

        for service_name, config in services.items():
            if service_name == "redis":
                # Test Redis
                try:
                    import redis
                    r = redis.Redis(host='localhost', port=6379)
                    r.ping()
                    service_results[service_name] = {"status": "healthy", "connected": True}
                except:
                    service_results[service_name] = {"status": "down", "connected": False}
            else:
                # Test HTTP services
                try:
                    url = f"http://localhost:{config['port']}{config['endpoint']}"
                    response = requests.get(url, timeout=5)
                    service_results[service_name] = {
                        "status": "healthy" if response.status_code == 200 else "unhealthy",
                        "status_code": response.status_code,
                        "connected": True
                    }
                except Exception as e:
                    service_results[service_name] = {
                        "status": "down",
                        "connected": False,
                        "error": str(e)
                    }

        return service_results

    def test_api_endpoints(self) -> Dict:
        """Test all API endpoints"""
        print("\n🌐 Testing API Endpoints...")
        import requests

        endpoints = [
            ("GET", "/api/echo/health"),
            ("GET", "/api/echo/models/list"),
            ("GET", "/api/echo/brain"),
            ("GET", "/api/echo/conversations"),
            ("POST", "/api/echo/query", {"query": "test", "conversation_id": "diagnostic_test"}),
            ("GET", "/api/echo/tasks"),
            ("GET", "/api/echo/memory/search?query=test")
        ]

        endpoint_results = {}
        base_url = "http://localhost:8309"

        for method, endpoint, *data in endpoints:
            try:
                if method == "GET":
                    response = requests.get(f"{base_url}{endpoint}", timeout=5)
                else:
                    payload = data[0] if data else {}
                    response = requests.post(f"{base_url}{endpoint}", json=payload, timeout=5)

                endpoint_results[endpoint] = {
                    "method": method,
                    "status_code": response.status_code,
                    "success": response.status_code in [200, 201],
                    "response_size": len(response.text)
                }
            except Exception as e:
                endpoint_results[endpoint] = {
                    "method": method,
                    "success": False,
                    "error": str(e)
                }

        return endpoint_results

    def test_background_tasks(self) -> Dict:
        """Test background task system"""
        print("\n⚙️ Testing Background Tasks...")

        task_results = {
            "task_queue": {},
            "autonomous_behaviors": {},
            "scheduled_jobs": {}
        }

        # Check task queue
        try:
            from src.services.task_queue import TaskQueue
            tq = TaskQueue()
            task_results["task_queue"] = {
                "initialized": True,
                "queue_size": tq.get_queue_size() if hasattr(tq, 'get_queue_size') else "unknown"
            }
        except Exception as e:
            task_results["task_queue"] = {
                "initialized": False,
                "error": str(e)
            }

        # Check autonomous behaviors
        try:
            from src.services.autonomous_behaviors import AutonomousBehaviors
            ab = AutonomousBehaviors()
            task_results["autonomous_behaviors"] = {
                "initialized": True,
                "enabled": ab.enabled if hasattr(ab, 'enabled') else "unknown"
            }
        except Exception as e:
            task_results["autonomous_behaviors"] = {
                "initialized": False,
                "error": str(e)
            }

        return task_results

    def generate_report(self) -> str:
        """Generate comprehensive diagnostic report"""
        report = []
        report.append("=" * 80)
        report.append("ECHO BRAIN DIAGNOSTIC REPORT")
        report.append(f"Generated: {self.results['timestamp']}")
        report.append("=" * 80)

        # File Structure Report
        report.append("\n📁 FILE STRUCTURE ANALYSIS")
        report.append("-" * 40)
        for dir_path, info in self.results["file_structure"].items():
            if dir_path == "symlink_issues":
                if info:
                    report.append(f"\n⚠️ Symbolic Link Issues Found: {len(info)}")
                    for issue in info[:5]:
                        report.append(f"  - {issue}")
            elif dir_path == "root_files":
                report.append(f"\n📄 Root Python Files: {len(info)}")
                for file in info[:10]:
                    report.append(f"  - {file}")
            else:
                status = "✅" if info.get("exists") else "❌"
                report.append(f"{status} {dir_path}: {info.get('file_count', 0)} files")

        # Database Report
        report.append("\n\n🗄️ DATABASE ANALYSIS")
        report.append("-" * 40)
        db = self.results["database"]
        report.append(f"Connection: {'✅ Connected' if db.get('connection') else '❌ Failed'}")
        report.append(f"Total Tables: {len(db.get('tables', []))}")
        if db.get("missing_tables"):
            report.append(f"\n⚠️ Missing Tables ({len(db['missing_tables'])}):")
            for table in db["missing_tables"]:
                report.append(f"  - {table}")

        # Import Analysis
        report.append("\n\n📦 IMPORT ANALYSIS")
        report.append("-" * 40)
        imports = self.results["imports"]
        report.append(f"✅ Successful Imports: {len(imports.get('success', []))}")
        report.append(f"❌ Failed Imports: {len(imports.get('failures', []))}")
        report.append(f"🔄 Circular Imports: {len(imports.get('circular', []))}")

        if imports.get("failures"):
            report.append("\nTop Import Failures:")
            for failure in imports["failures"][:10]:
                report.append(f"  - {failure['file']}")
                if failure.get('missing'):
                    report.append(f"    Missing: {failure['missing']}")

        # Service Health
        report.append("\n\n🚀 SERVICE HEALTH")
        report.append("-" * 40)
        for service, status in self.results.get("services", {}).items():
            icon = "✅" if status.get("connected") else "❌"
            report.append(f"{icon} {service}: {status.get('status', 'unknown')}")
            if status.get("error"):
                report.append(f"    Error: {status['error']}")

        # API Endpoints
        report.append("\n\n🌐 API ENDPOINT STATUS")
        report.append("-" * 40)
        working = sum(1 for e in self.results.get("api_endpoints", {}).values() if e.get("success"))
        total = len(self.results.get("api_endpoints", {}))
        report.append(f"Working Endpoints: {working}/{total}")

        for endpoint, result in self.results.get("api_endpoints", {}).items():
            icon = "✅" if result.get("success") else "❌"
            report.append(f"{icon} {result.get('method', 'GET')} {endpoint}")
            if result.get("error"):
                report.append(f"    Error: {result['error']}")

        # Background Tasks
        report.append("\n\n⚙️ BACKGROUND TASK SYSTEM")
        report.append("-" * 40)
        for task_type, info in self.results.get("background_tasks", {}).items():
            icon = "✅" if info.get("initialized") else "❌"
            report.append(f"{icon} {task_type}")
            if info.get("error"):
                report.append(f"    Error: {info['error']}")

        # Summary and Recommendations
        report.append("\n\n📊 SUMMARY & RECOMMENDATIONS")
        report.append("-" * 40)

        critical_issues = []
        if db.get("missing_tables"):
            critical_issues.append(f"Create {len(db['missing_tables'])} missing database tables")
        if imports.get("failures"):
            critical_issues.append(f"Fix {len(imports['failures'])} import failures")
        if self.results["file_structure"].get("symlink_issues"):
            critical_issues.append("Remove circular symbolic links")

        if critical_issues:
            report.append("\n🚨 CRITICAL ISSUES TO FIX:")
            for i, issue in enumerate(critical_issues, 1):
                report.append(f"  {i}. {issue}")
        else:
            report.append("✅ No critical issues found!")

        report.append("\n" + "=" * 80)
        return "\n".join(report)

    def run(self):
        """Run all diagnostic tests"""
        print("🏥 Starting Echo Brain Comprehensive Diagnostic...")

        # Run tests
        self.results["file_structure"] = self.test_file_structure()
        self.results["database"] = self.test_database()
        self.results["imports"] = self.test_imports()
        self.results["services"] = self.test_services()
        self.results["api_endpoints"] = self.test_api_endpoints()
        self.results["background_tasks"] = self.test_background_tasks()

        # Generate report
        report = self.generate_report()

        # Save results
        with open(self.base_path / "diagnostic_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        with open(self.base_path / "diagnostic_report.txt", "w") as f:
            f.write(report)

        print(report)
        print(f"\n📄 Full results saved to:")
        print(f"  - {self.base_path}/diagnostic_results.json")
        print(f"  - {self.base_path}/diagnostic_report.txt")

        return self.results

if __name__ == "__main__":
    diagnostic = EchoBrainDiagnostic()
    diagnostic.run()
"""
Echo Brain Self-Test - Live Streaming Diagnostics
Shows real-time color-coded system testing with progress bars
"""
import asyncio
import httpx
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

# Try to import rich, fallback to simple if not available
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None
logger = logging.getLogger(__name__)

class EchoBrainSelfTester:
    """Echo Brain's self-testing system with live streaming"""

    def __init__(self, stream_output=True):
        self.stream_output = stream_output
        self.results = []
        self.test_count = 0
        self.passed = 0
        self.failed = 0

    def _print(self, message: str, color: str = "white"):
        """Print with optional rich formatting"""
        if not self.stream_output:
            return

        if RICH_AVAILABLE:
            if color == "green":
                console.print(f"[green]✓ {message}[/green]")
            elif color == "red":
                console.print(f"[red]✗ {message}[/red]")
            elif color == "yellow":
                console.print(f"[yellow]⚠ {message}[/yellow]")
            elif color == "blue":
                console.print(f"[blue]→ {message}[/blue]")
            elif color == "cyan":
                console.print(f"[cyan]🧠 {message}[/cyan]")
            else:
                console.print(message)
        else:
            # Fallback to ANSI colors
            colors = {
                "green": "\033[92m",
                "red": "\033[91m",
                "yellow": "\033[93m",
                "blue": "\033[94m",
                "cyan": "\033[96m",
                "reset": "\033[0m"
            }
            prefix = "✓" if color == "green" else "✗" if color == "red" else "⚠" if color == "yellow" else "→"
            print(f"{colors.get(color, '')}{prefix} {message}{colors.get('reset', '')}")

    async def run_full_diagnostics(self):
        """Run complete system diagnostics"""
        self._print("Starting Echo Brain Self-Test...", "cyan")

        tests = [
            self.test_postgresql,
            self.test_qdrant,
            self.test_ollama,
            self.test_echo_brain_api,
            self.test_mcp_server,
            self.test_file_system,
            self.test_ingestion_pipeline,
            self.test_end_to_end_flow
        ]

        # Run tests with progress indicator
        if RICH_AVAILABLE and self.stream_output:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress:
                task = progress.add_task("[cyan]Running tests...", total=len(tests))

                for i, test_func in enumerate(tests):
                    test_name = test_func.__name__.replace("test_", "").replace("_", " ").title()
                    progress.update(task, description=f"[blue]{i+1}. Testing {test_name}...")

                    result = await test_func()
                    self.results.append(result)

                    if result["status"] == "passed":
                        self.passed += 1
                    elif result["status"] == "failed":
                        self.failed += 1

                    progress.advance(task)
                    await asyncio.sleep(0.5)
        else:
            # Simple sequential execution
            for i, test_func in enumerate(tests):
                test_name = test_func.__name__.replace("test_", "").replace("_", " ").title()
                self._print(f"{i+1}. Testing {test_name}...", "blue")

                result = await test_func()
                self.results.append(result)

                if result["status"] == "passed":
                    self._print(f"  {test_name}: PASSED", "green")
                    self.passed += 1
                elif result["status"] == "failed":
                    self._print(f"  {test_name}: FAILED - {result.get('error', 'Unknown')}", "red")
                    self.failed += 1
                else:
                    self._print(f"  {test_name}: DEGRADED", "yellow")

                await asyncio.sleep(0.5)

        # Display summary
        await self._display_summary()

        return {
            "total_tests": len(tests),
            "passed": self.passed,
            "failed": self.failed,
            "success_rate": (self.passed / len(tests)) * 100 if tests else 0,
            "results": self.results
        }

    async def test_postgresql(self) -> Dict[str, Any]:
        """Test PostgreSQL connectivity and data"""
        self._print("Connecting to PostgreSQL...", "blue")

        try:
            import asyncpg

            # Connect with explicit error handling
            conn = await asyncpg.connect(
                host='localhost',
                database='echo_brain',
                user='patrick',
                password=os.getenv("DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE"),
                timeout=10
            )

            # Run tests
            tests = [
                ("Basic Connection", "SELECT 1 as test"),
                ("Conversations Table", "SELECT COUNT(*) as count FROM conversations"),
                ("Echo Conversations", "SELECT COUNT(*) as count FROM echo_conversations"),
                ("Recent Data", """
                    SELECT COUNT(*) as recent
                    FROM conversations
                    WHERE created_at > NOW() - INTERVAL '7 days'
                """),
                ("Tables Status", """
                    SELECT table_name,
                           (xpath('/row/cnt/text()', query_to_xml('SELECT COUNT(*) as cnt FROM ' || table_name, true, false, '')))[1]::text::int as count
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_type = 'BASE TABLE'
                    AND table_name IN ('conversations', 'echo_conversations', 'vector_content', 'ingestion_tracking', 'facts')
                    ORDER BY table_name
                """)
            ]

            results = {}
            for test_name, query in tests:
                try:
                    result = await conn.fetchrow(query)
                    results[test_name] = dict(result) if result else {}

                    # Show immediate feedback
                    if "count" in results[test_name]:
                        count = results[test_name]["count"]
                        if count is not None:
                            self._print(f"  {test_name}: {count}", "green")
                        else:
                            self._print(f"  {test_name}: No data", "yellow")
                except Exception as e:
                    self._print(f"  {test_name}: Error - {str(e)[:50]}", "red")
                    results[test_name] = {"error": str(e)}

            await conn.close()

            # Determine status
            has_conversations = results.get("Conversations Table", {}).get("count", 0) > 0
            status = "passed" if has_conversations else "degraded"

            return {
                "component": "PostgreSQL",
                "status": status,
                "details": results,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self._print(f"PostgreSQL Connection Failed: {str(e)[:100]}", "red")
            return {
                "component": "PostgreSQL",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def test_qdrant(self) -> Dict[str, Any]:
        """Test Qdrant vector store"""
        self._print("Connecting to Qdrant...", "blue")

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # Get collection info
                response = await client.get("http://localhost:6333/collections/echo_memory")

                if response.status_code == 200:
                    data = response.json()
                    collection = data["result"]

                    points = collection.get("points_count", 0)
                    status = collection.get("status", "unknown")
                    dimensions = collection.get("config", {}).get("params", {}).get("vectors", {}).get("size", 0)

                    self._print(f"  Collection: echo_memory", "green")
                    self._print(f"  Points: {points:,}", "green")
                    self._print(f"  Status: {status}", "green")
                    self._print(f"  Dimensions: {dimensions}", "green")

                    # Test vector search
                    self._print("  Testing vector search...", "blue")
                    search_response = await client.post(
                        "http://localhost:6333/collections/echo_memory/points/search",
                        json={
                            "vector": [0.1] * dimensions if dimensions else [0.1] * 1024,
                            "limit": 1,
                            "with_payload": True
                        }
                    )

                    search_ok = search_response.status_code == 200
                    if search_ok:
                        self._print("  Vector search: Working", "green")
                    else:
                        self._print(f"  Vector search: Failed (HTTP {search_response.status_code})", "yellow")

                    return {
                        "component": "Qdrant",
                        "status": "passed" if points > 0 else "degraded",
                        "points": points,
                        "status_text": status,
                        "dimensions": dimensions,
                        "search_working": search_ok,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    self._print(f"Qdrant Error: HTTP {response.status_code}", "red")
                    return {
                        "component": "Qdrant",
                        "status": "failed",
                        "error": f"HTTP {response.status_code}",
                        "timestamp": datetime.now().isoformat()
                    }

        except Exception as e:
            self._print(f"Qdrant Connection Failed: {str(e)[:100]}", "red")
            return {
                "component": "Qdrant",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def test_ollama(self) -> Dict[str, Any]:
        """Test Ollama embedding model"""
        self._print("Testing Ollama embeddings...", "blue")

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Check model availability
                self._print("  Checking model availability...", "blue")
                response = await client.get("http://localhost:11434/api/tags")

                if response.status_code == 200:
                    models = response.json()
                    model_names = [m["name"] for m in models.get("models", [])]

                    # Check for model with or without :latest tag
                    has_model = "mxbai-embed-large" in model_names or "mxbai-embed-large:latest" in model_names
                    if has_model:
                        model_name = "mxbai-embed-large:latest" if "mxbai-embed-large:latest" in model_names else "mxbai-embed-large"
                        self._print(f"  Model found: {model_name}", "green")

                        # Test embedding
                        self._print("  Testing embedding generation...", "blue")
                        embed_response = await client.post(
                            "http://localhost:11434/api/embeddings",
                            json={
                                "model": model_name,
                                "prompt": "Test embedding for Echo Brain"
                            },
                            timeout=60
                        )

                        if embed_response.status_code == 200:
                            embed_data = embed_response.json()
                            embedding = embed_data.get("embedding", [])

                            if embedding and len(embedding) > 0:
                                self._print(f"  Embedding generated: {len(embedding)} dimensions", "green")

                                return {
                                    "component": "Ollama",
                                    "status": "passed",
                                    "model": model_name,
                                    "embedding_length": len(embedding),
                                    "timestamp": datetime.now().isoformat()
                                }
                            else:
                                self._print("  Embedding generation failed: Empty response", "yellow")
                                return {
                                    "component": "Ollama",
                                    "status": "degraded",
                                    "error": "Empty embedding response",
                                    "timestamp": datetime.now().isoformat()
                                }
                        else:
                            self._print(f"  Embedding failed: HTTP {embed_response.status_code}", "red")
                            return {
                                "component": "Ollama",
                                "status": "failed",
                                "error": f"Embedding HTTP {embed_response.status_code}",
                                "timestamp": datetime.now().isoformat()
                            }
                    else:
                        self._print(f"  Model not found: mxbai-embed-large", "red")
                        self._print(f"  Available: {', '.join(model_names[:3])}", "yellow")
                        return {
                            "component": "Ollama",
                            "status": "failed",
                            "error": "Model mxbai-embed-large not found",
                            "available_models": model_names,
                            "timestamp": datetime.now().isoformat()
                        }
                else:
                    self._print(f"Ollama Error: HTTP {response.status_code}", "red")
                    return {
                        "component": "Ollama",
                        "status": "failed",
                        "error": f"HTTP {response.status_code}",
                        "timestamp": datetime.now().isoformat()
                    }

        except Exception as e:
            self._print(f"Ollama Connection Failed: {str(e)[:100]}", "red")
            return {
                "component": "Ollama",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def test_echo_brain_api(self) -> Dict[str, Any]:
        """Test Echo Brain API endpoints"""
        self._print("Testing Echo Brain API...", "blue")

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                endpoints = [
                    ("Health", "/health"),
                    ("Memory Status", "/api/memory/status"),
                ]

                results = {}
                for endpoint_name, path in endpoints:
                    try:
                        self._print(f"  Testing {endpoint_name}...", "blue")
                        response = await client.get(f"http://localhost:8309{path}")

                        if response.status_code == 200:
                            data = response.json()
                            results[endpoint_name] = {"status": "ok", "code": 200}
                            self._print(f"    {endpoint_name}: OK", "green")
                        else:
                            results[endpoint_name] = {"status": "error", "code": response.status_code}
                            self._print(f"    {endpoint_name}: HTTP {response.status_code}", "yellow")

                    except Exception as e:
                        results[endpoint_name] = {"status": "error", "error": str(e)}
                        self._print(f"    {endpoint_name}: Failed - {str(e)[:50]}", "red")

                # Determine overall status
                successful = sum(1 for r in results.values() if r.get("status") == "ok")
                total = len(endpoints)

                if successful == total:
                    status = "passed"
                elif successful > 0:
                    status = "degraded"
                else:
                    status = "failed"

                return {
                    "component": "Echo Brain API",
                    "status": status,
                    "endpoints_tested": total,
                    "endpoints_ok": successful,
                    "details": results,
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            self._print(f"Echo Brain API Test Failed: {str(e)[:100]}", "red")
            return {
                "component": "Echo Brain API",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def test_mcp_server(self) -> Dict[str, Any]:
        """Test MCP Server functionality"""
        self._print("Testing MCP Server...", "blue")

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                self._print("  Testing search_memory tool...", "blue")

                response = await client.post(
                    "http://localhost:8309/mcp",
                    json={
                        "method": "tools/call",
                        "params": {
                            "name": "search_memory",
                            "arguments": {
                                "query": "test",
                                "limit": 1
                            }
                        }
                    },
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    results = data.get("result", [])

                    if isinstance(results, list):
                        self._print(f"  Search working: {len(results)} results", "green")

                        if results:
                            self._print(f"    Top score: {results[0].get('score', 0):.3f}", "green")

                        return {
                            "component": "MCP Server",
                            "status": "passed",
                            "search_results": len(results),
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        self._print("  Search returned invalid format", "yellow")
                        return {
                            "component": "MCP Server",
                            "status": "degraded",
                            "error": "Invalid response format",
                            "timestamp": datetime.now().isoformat()
                        }
                else:
                    self._print(f"  MCP Error: HTTP {response.status_code}", "red")
                    return {
                        "component": "MCP Server",
                        "status": "failed",
                        "error": f"HTTP {response.status_code}",
                        "timestamp": datetime.now().isoformat()
                    }

        except Exception as e:
            self._print(f"MCP Server Test Failed: {str(e)[:100]}", "red")
            return {
                "component": "MCP Server",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def test_file_system(self) -> Dict[str, Any]:
        """Test file system and source data"""
        self._print("Checking file system...", "blue")

        try:
            import os
            from pathlib import Path

            conv_dir = Path("/home/patrick/.claude/projects")

            if conv_dir.exists():
                jsonl_files = list(conv_dir.rglob("*.jsonl"))
                total_files = len(jsonl_files)

                self._print(f"  Conversation directory: {conv_dir}", "green")
                self._print(f"  JSONL files found: {total_files:,}", "green")

                if total_files > 0:
                    # Show recent files
                    recent_files = sorted(jsonl_files, key=lambda f: f.stat().st_mtime, reverse=True)[:2]
                    for f in recent_files:
                        mtime = datetime.fromtimestamp(f.stat().st_mtime)
                        size_mb = f.stat().st_size / (1024 * 1024)
                        self._print(f"    Recent: {f.name} ({size_mb:.1f}MB, {mtime.strftime('%Y-%m-%d')})", "blue")

                return {
                    "component": "File System",
                    "status": "passed",
                    "conversation_files": total_files,
                    "directory": str(conv_dir),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                self._print(f"  Directory not found: {conv_dir}", "red")
                return {
                    "component": "File System",
                    "status": "failed",
                    "error": "Conversation directory not found",
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            self._print(f"File System Check Failed: {str(e)[:100]}", "red")
            return {
                "component": "File System",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def test_ingestion_pipeline(self) -> Dict[str, Any]:
        """Test ingestion pipeline"""
        self._print("Testing ingestion pipeline...", "blue")

        try:
            # First check current state
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get("http://localhost:8309/api/memory/status")

                if response.status_code == 200:
                    status_data = response.json()

                    # Don't test if already running
                    if status_data.get("is_running", False):
                        self._print("  Ingestion is currently running", "yellow")
                        return {
                            "component": "Ingestion Pipeline",
                            "status": "degraded",
                            "note": "Already running",
                            "timestamp": datetime.now().isoformat()
                        }

                    # Quick test - count conversation files
                    import os
                    from pathlib import Path
                    conv_files = len(list(Path("/home/patrick/.claude/projects").rglob("*.jsonl")))

                    self._print(f"  Conversation files: {conv_files:,}", "green")
                    self._print(f"  Conversations processed: {status_data.get('conversations_processed', 0)}", "green")
                    self._print(f"  Embeddings created: {status_data.get('embeddings_created', 0)}", "green")

                    return {
                        "component": "Ingestion Pipeline",
                        "status": "passed",
                        "conversation_files": conv_files,
                        "conversations_processed": status_data.get("conversations_processed", 0),
                        "embeddings_created": status_data.get("embeddings_created", 0),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    self._print(f"  Ingestion status error: HTTP {response.status_code}", "red")
                    return {
                        "component": "Ingestion Pipeline",
                        "status": "failed",
                        "error": f"HTTP {response.status_code}",
                        "timestamp": datetime.now().isoformat()
                    }

        except Exception as e:
            self._print(f"Ingestion Test Failed: {str(e)[:100]}", "red")
            return {
                "component": "Ingestion Pipeline",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def test_end_to_end_flow(self) -> Dict[str, Any]:
        """Test complete end-to-end flow"""
        self._print("Testing end-to-end flow...", "blue")

        import os
        from pathlib import Path

        components = [
            ("Source Files", lambda: len(list(Path("/home/patrick/.claude/projects").rglob("*.jsonl"))) > 0),
            ("PostgreSQL", self._check_postgresql_has_data),
            ("Qdrant", self._check_qdrant_has_data),
            ("MCP Search", self._check_mcp_search)
        ]

        working_components = 0
        details = {}

        for name, check_func in components:
            try:
                result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                if result:
                    self._print(f"  {name}: ✓", "green")
                    working_components += 1
                    details[name] = "working"
                else:
                    self._print(f"  {name}: ✗", "red")
                    details[name] = "failed"
            except Exception as e:
                self._print(f"  {name}: ✗ ({str(e)[:30]})", "red")
                details[name] = f"error: {str(e)[:50]}"

        total = len(components)
        success_rate = (working_components / total) * 100

        if success_rate == 100:
            status = "passed"
        elif success_rate >= 75:
            status = "degraded"
        else:
            status = "failed"

        self._print(f"  Overall: {working_components}/{total} components ({success_rate:.0f}%)",
                   "green" if status == "passed" else "yellow" if status == "degraded" else "red")

        return {
            "component": "End-to-End Flow",
            "status": status,
            "score": f"{working_components}/{total}",
            "success_rate": success_rate,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }

    async def _check_postgresql_has_data(self) -> bool:
        """Check if PostgreSQL has conversation data"""
        try:
            import asyncpg
            conn = await asyncpg.connect(
                host='localhost',
                database='echo_brain',
                user='patrick',
                password=os.getenv("DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE"),
                timeout=5
            )
            count = await conn.fetchval("SELECT COUNT(*) FROM conversations")
            await conn.close()
            return count > 0
        except:
            return False

    async def _check_qdrant_has_data(self) -> bool:
        """Check if Qdrant has vector data"""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get("http://localhost:6333/collections/echo_memory")
                if response.status_code == 200:
                    points = response.json()["result"]["points_count"]
                    return points > 0
                return False
        except:
            return False

    async def _check_mcp_search(self) -> bool:
        """Check if MCP search works"""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.post(
                    "http://localhost:8309/mcp",
                    json={
                        "method": "tools/call",
                        "params": {
                            "name": "search_memory",
                            "arguments": {"query": "test", "limit": 1}
                        }
                    }
                )
                return response.status_code == 200
        except:
            return False

    async def _display_summary(self):
        """Display test summary"""
        total = len(self.results)
        success_rate = (self.passed / total) * 100 if total > 0 else 0

        if RICH_AVAILABLE and self.stream_output:
            table = Table(title="🧠 Echo Brain Self-Test Results", box=box.ROUNDED)
            table.add_column("Component", style="cyan")
            table.add_column("Status", justify="center")
            table.add_column("Details", style="white")

            for result in self.results:
                component = result["component"]
                status = result["status"]

                if status == "passed":
                    status_display = "[green]✓ PASSED[/green]"
                elif status == "failed":
                    status_display = "[red]✗ FAILED[/red]"
                else:
                    status_display = "[yellow]⚠ DEGRADED[/yellow]"

                details = ""
                if "points" in result:
                    details = f"{result['points']:,} vectors"
                elif "conversation_files" in result:
                    details = f"{result['conversation_files']:,} files"
                elif "endpoints_ok" in result:
                    details = f"{result['endpoints_ok']}/{result['endpoints_tested']} endpoints"
                elif "score" in result:
                    details = result["score"]

                table.add_row(component, status_display, details)

            console.print(table)

            # Summary panel
            summary_text = f"""
[b]Test Summary:[/b]
✅ Passed: {self.passed}
❌ Failed: {self.failed}
📊 Success Rate: {success_rate:.1f}%
⏱️ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """

            if success_rate == 100:
                border_color = "green"
                status = "[green]HEALTHY[/green]"
            elif success_rate >= 70:
                border_color = "yellow"
                status = "[yellow]DEGRADED[/yellow]"
            else:
                border_color = "red"
                status = "[red]UNHEALTHY[/red]"

            console.print(Panel.fit(
                f"{summary_text}\n[b]Overall Status:[/b] {status}",
                title="System Health",
                border_style=border_color,
                padding=(1, 2)
            ))
        else:
            # Simple text summary
            print("\n" + "="*50)
            print("🧠 ECHO BRAIN SELF-TEST SUMMARY")
            print("="*50)

            for result in self.results:
                status_icon = "✓" if result["status"] == "passed" else "✗" if result["status"] == "failed" else "⚠"
                print(f"{status_icon} {result['component']}: {result['status'].upper()}")

            print("\n" + "-"*50)
            print(f"Passed: {self.passed}/{total}")
            print(f"Success Rate: {success_rate:.1f}%")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*50)

            if success_rate == 100:
                print("✅ SYSTEM: HEALTHY")
            elif success_rate >= 70:
                print("⚠️ SYSTEM: DEGRADED")
            else:
                print("❌ SYSTEM: UNHEALTHY")


# Command line interface
async def main():
    """Main entry point for self-test"""
    print("\n" + "="*60)
    print("🧠 ECHO BRAIN SELF-TEST SYSTEM")
    print("="*60)

    tester = EchoBrainSelfTester(stream_output=True)
    results = await tester.run_full_diagnostics()

    # Exit with appropriate code
    success_rate = results["success_rate"]
    if success_rate == 100:
        sys.exit(0)  # Healthy
    elif success_rate >= 70:
        sys.exit(1)  # Degraded
    else:
        sys.exit(2)  # Unhealthy

if __name__ == "__main__":
    asyncio.run(main())
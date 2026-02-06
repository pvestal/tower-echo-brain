"""
Echo Brain - MINIMAL WORKING VERSION
Only essential features, no duplicate systems
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Optional
import logging
import os
import time
import uuid
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from collections import defaultdict, deque

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Request metrics tracking
REQUEST_METRICS = {
    "total_requests": 0,
    "errors_4xx": 0,
    "errors_5xx": 0,
    "response_times": deque(maxlen=1000),  # Keep last 1000 response times
    "requests_by_endpoint": defaultdict(int),
    "errors_by_endpoint": defaultdict(int),
    "slowest_requests": deque(maxlen=10)  # Keep 10 slowest requests
}

# Error log for dashboard
ERROR_LOG = deque(maxlen=100)  # Keep last 100 errors

app = FastAPI(
    title="Tower Echo Brain",
    version="0.4.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Basic health endpoint
@app.get("/api")
async def api_root():
    return {
        "service": "Echo Brain",
        "version": "0.4.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "echo-brain",
        "database_user": os.getenv("DB_USER", "not_set"),
        "timestamp": datetime.now().isoformat()
    }

# Mount the consolidated Echo Brain router - ALL endpoints in one place
try:
    from src.api.endpoints.echo_main_router import router as echo_main_router
    app.include_router(echo_main_router, prefix="/api/echo")
    logger.info("✅ Echo main router mounted at /api/echo - ALL endpoints consolidated")
except ImportError as e:
    logger.error(f"❌ Echo main router not available: {e}")
except Exception as e:
    logger.error(f"❌ Echo main router failed: {e}")

# Mount Pipeline API - Three-Layer Architecture
try:
    from src.api.endpoints.pipeline_router import router as pipeline_router
    app.include_router(pipeline_router)
    logger.info("✅ Pipeline router mounted at /api/pipeline - Context→Reasoning→Narrative pipeline")
except ImportError as e:
    logger.error(f"❌ Pipeline router not available: {e}")
except Exception as e:
    logger.error(f"❌ Pipeline router failed: {e}")

# Mount Agent API with proper async context
try:
    from src.api.agents import router as agent_router
    app.include_router(agent_router)
    logger.info("✅ Agent router mounted")
except ImportError as e:
    logger.warning(f"⚠️ Agent router not available: {e}")
except Exception as e:
    logger.error(f"❌ Agent router failed: {e}")

# Mount Autonomous API with initialization check
try:
    from src.api.autonomous import router as autonomous_router
    app.include_router(autonomous_router)
    logger.info("✅ Autonomous router mounted")
except ImportError as e:
    logger.warning(f"⚠️ Autonomous router not available: {e}")
except Exception as e:
    logger.error(f"❌ Autonomous router failed: {e}")

# Mount Unified Memory System API
try:
    from src.api.endpoints.memory_router import router as memory_router
    app.include_router(memory_router, prefix="/api/echo")
    logger.info("✅ Memory router mounted at /api/echo/memory")
except ImportError as e:
    logger.warning(f"⚠️ Memory router not available: {e}")
except Exception as e:
    logger.error(f"❌ Memory router failed: {e}")

# Mount Self-Test API
try:
    from src.api.endpoints.self_test_router import router as self_test_router
    app.include_router(self_test_router, prefix="/api/echo")
    logger.info("✅ Self-test router mounted at /api/echo/self-test")
except ImportError as e:
    logger.warning(f"⚠️ Self-test router not available: {e}")
except Exception as e:
    logger.error(f"❌ Self-test router failed: {e}")

# Mount Intelligence API - This is where Echo Brain THINKS
try:
    from src.api.endpoints.intelligence_router import router as intelligence_router
    app.include_router(intelligence_router, prefix="/api/echo")
    logger.info("✅ Intelligence router mounted at /api/echo/intelligence")
except ImportError as e:
    logger.warning(f"⚠️ Intelligence router not available: {e}")
except Exception as e:
    logger.error(f"❌ Intelligence router failed: {e}")

# Mount Reasoning API - This adds LLM synthesis for natural responses
try:
    from src.api.endpoints.reasoning_router import router as reasoning_router
    app.include_router(reasoning_router, prefix="/api/echo")
    logger.info("✅ Reasoning router mounted at /api/echo/reasoning")
except ImportError as e:
    logger.warning(f"⚠️ Reasoning router not available: {e}")
except Exception as e:
    logger.error(f"❌ Reasoning router failed: {e}")

# Mount Search API - Direct PostgreSQL conversation search
try:
    from src.api.endpoints.search_router import router as search_router
    app.include_router(search_router, prefix="/api/echo")
    logger.info("✅ Search router mounted at /api/echo/search")
except ImportError as e:
    logger.warning(f"⚠️ Search router not available: {e}")
except Exception as e:
    logger.error(f"❌ Search router failed: {e}")

# Mount Reasoning API - Transparent multi-stage reasoning
try:
    from src.api.endpoints.echo_reasoning_router import router as reasoning_router
    app.include_router(reasoning_router, prefix="/api/echo")
    logger.info("✅ Reasoning router mounted at /api/echo/reasoning")
except ImportError as e:
    logger.warning(f"⚠️ Reasoning router not available: {e}")
except Exception as e:
    logger.error(f"❌ Reasoning router failed: {e}")

# Health endpoints are now in the consolidated echo_main_router

# Create minimal MCP endpoints inline for now
@app.post("/mcp")
async def mcp_handler(request: dict):
    """Handle MCP requests with Qdrant integration"""
    from src.integrations.mcp_service import mcp_service

    method = request.get("method", "")

    if method == "tools/list":
        return {
            "tools": [
                {"name": "search_memory", "description": "Search Echo Brain memories"},
                {"name": "get_facts", "description": "Get facts from Echo Brain"},
                {"name": "store_fact", "description": "Store a fact in Echo Brain"}
            ]
        }
    elif method == "tools/call":
        tool_name = request.get("params", {}).get("name")
        arguments = request.get("params", {}).get("arguments", {})

        try:
            if tool_name == "search_memory":
                query = arguments.get("query", "")
                limit = arguments.get("limit", 10)
                results = await mcp_service.search_memory(query, limit)
                return results

            elif tool_name == "get_facts":
                topic = arguments.get("topic")
                limit = arguments.get("limit", 100)
                facts = await mcp_service.get_facts(topic, limit)
                return facts

            elif tool_name == "store_fact":
                subject = arguments.get("subject", "")
                predicate = arguments.get("predicate", "")
                object_ = arguments.get("object", "")
                confidence = arguments.get("confidence", 1.0)
                fact_id = await mcp_service.store_fact(subject, predicate, object_, confidence)
                return {"fact_id": fact_id, "stored": True}

            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"MCP handler error: {e}")
            return {"error": str(e)}

    return {"error": f"Unknown method: {method}"}

@app.get("/mcp/health")
async def mcp_health():
    """MCP health check"""
    return {
        "status": "ok",
        "service": "echo-brain-mcp",
        "version": "1.0.0"
    }

# Worker status endpoint - MUST be before frontend mount
@app.get("/api/workers/status")
async def get_worker_status():
    """Get status of all scheduled workers."""
    try:
        from src.autonomous.worker_scheduler import worker_scheduler
        return worker_scheduler.get_status()
    except Exception as e:
        return {"error": str(e), "running": False, "workers": {}}

# Detailed health endpoint for self-awareness - MUST be before frontend mount
@app.get("/api/echo/health/detailed")
async def get_detailed_health():
    """Get comprehensive health status including self-awareness metrics."""
    import asyncpg
    from datetime import datetime, timedelta

    try:
        # Connect to database
        db_url = os.getenv("DATABASE_URL", "postgresql://echo:echo_secure_password_123@localhost/echo_brain")
        conn = await asyncpg.connect(db_url)

        # Get worker status
        from src.autonomous.worker_scheduler import worker_scheduler
        worker_status = worker_scheduler.get_status()

        # Get vector count
        vector_count = await conn.fetchval("SELECT COUNT(*) FROM vector_content")

        # Get facts count
        facts_count = await conn.fetchval("SELECT COUNT(*) FROM facts")

        # Get codebase indexing status
        codebase_files = await conn.fetchval("SELECT COUNT(*) FROM self_codebase_index")

        # Get schema indexing status
        schema_tables = await conn.fetchval("SELECT COUNT(*) FROM self_schema_index")

        # Get extraction coverage (handle missing table)
        total_vectors = await conn.fetchval("SELECT COUNT(*) FROM vector_content")
        try:
            extracted_vectors = await conn.fetchval(
                "SELECT COUNT(DISTINCT vector_id) FROM fact_extraction_tracking WHERE status = 'completed'"
            ) or 0
        except Exception:
            extracted_vectors = 0
        coverage_pct = (extracted_vectors / total_vectors * 100) if total_vectors > 0 else 0

        # Get graph edges count (handle missing table)
        try:
            graph_edges = await conn.fetchval("SELECT COUNT(*) FROM knowledge_graph")
        except Exception:
            graph_edges = 0

        # Get recent test results
        test_results = await conn.fetch("""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE passed = true) as passed
            FROM self_test_results
            WHERE run_at > NOW() - INTERVAL '2 hours'
        """)

        test_total = test_results[0]['total'] if test_results else 0
        test_passed = test_results[0]['passed'] if test_results else 0
        pass_rate = (test_passed / test_total) if test_total > 0 else 0

        # Get last test run time
        last_test_run = await conn.fetchval(
            "SELECT MAX(run_at) FROM self_test_results"
        )

        # Count regressions
        regressions = await conn.fetchval("""
            SELECT COUNT(*) FROM self_detected_issues
            WHERE issue_type = 'query_regression'
            AND status = 'open'
            AND created_at > NOW() - INTERVAL '24 hours'
        """) or 0

        # Get open issues by severity
        issues = await conn.fetch("""
            SELECT
                severity,
                COUNT(*) as count
            FROM self_detected_issues
            WHERE status = 'open'
            GROUP BY severity
        """)

        issue_counts = {
            'critical': 0,
            'warning': 0,
            'info': 0
        }
        for row in issues:
            issue_counts[row['severity']] = row['count']

        # Get recent critical issues
        recent_issues = await conn.fetch("""
            SELECT title, severity, created_at
            FROM self_detected_issues
            WHERE status = 'open'
            ORDER BY
                CASE severity
                    WHEN 'critical' THEN 1
                    WHEN 'warning' THEN 2
                    ELSE 3
                END,
                created_at DESC
            LIMIT 5
        """)

        # Get resolved issues in last 24h
        resolved_24h = await conn.fetchval("""
            SELECT COUNT(*) FROM self_detected_issues
            WHERE status = 'resolved'
            AND resolved_at > NOW() - INTERVAL '24 hours'
        """) or 0

        await conn.close()

        # Determine overall status
        if issue_counts['critical'] >= 3 or pass_rate < 0.6:
            status = "unhealthy"
        elif issue_counts['critical'] > 0 or pass_rate < 0.9:
            status = "degraded"
        else:
            status = "healthy"

        # Check self-awareness status
        knows_own_code = codebase_files > 0
        knows_own_schema = schema_tables > 0
        monitors_own_logs = 'log_monitor' in worker_status.get('workers', {})
        validates_own_output = test_total > 0

        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "workers": worker_status.get('workers', {}),
            "knowledge": {
                "total_vectors": vector_count,
                "total_facts": facts_count,
                "extraction_coverage_pct": round(coverage_pct, 1),
                "graph_edges": graph_edges,
                "codebase_files_indexed": codebase_files,
                "schema_tables_indexed": schema_tables
            },
            "quality": {
                "last_test_pass_rate": round(pass_rate, 2),
                "tests_run": test_total,
                "tests_passed": test_passed,
                "last_test_run": last_test_run.isoformat() if last_test_run else None,
                "regressions_detected": regressions
            },
            "issues": {
                "open_critical": issue_counts['critical'],
                "open_warning": issue_counts['warning'],
                "open_info": issue_counts['info'],
                "resolved_last_24h": resolved_24h,
                "recent": [
                    {
                        "title": issue['title'][:100],
                        "severity": issue['severity'],
                        "created_at": issue['created_at'].isoformat()
                    }
                    for issue in recent_issues
                ]
            },
            "self_awareness": {
                "knows_own_code": knows_own_code,
                "knows_own_schema": knows_own_schema,
                "monitors_own_logs": monitors_own_logs,
                "validates_own_output": validates_own_output
            }
        }

    except Exception as e:
        logger.error(f"Failed to get detailed health: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Proposal review endpoints - MUST be before frontend mount
@app.get("/api/echo/proposals")
async def list_proposals(status: Optional[str] = None):
    """List improvement proposals with optional status filter."""
    import asyncpg

    try:
        db_url = os.getenv("DATABASE_URL", "postgresql://echo:echo_secure_password_123@localhost/echo_brain")
        conn = await asyncpg.connect(db_url)

        try:
            # Build query based on status filter
            if status:
                proposals = await conn.fetch("""
                    SELECT
                        p.id, p.title, p.target_file, p.risk_assessment,
                        p.status, p.created_at,
                        i.title as issue_title
                    FROM self_improvement_proposals p
                    LEFT JOIN self_detected_issues i ON p.issue_id = i.id
                    WHERE p.status = $1
                    ORDER BY p.created_at DESC
                """, status)
            else:
                proposals = await conn.fetch("""
                    SELECT
                        p.id, p.title, p.target_file, p.risk_assessment,
                        p.status, p.created_at,
                        i.title as issue_title
                    FROM self_improvement_proposals p
                    LEFT JOIN self_detected_issues i ON p.issue_id = i.id
                    ORDER BY p.created_at DESC
                """)

            # Get counts by status
            counts = await conn.fetch("""
                SELECT status, COUNT(*) as count
                FROM self_improvement_proposals
                GROUP BY status
            """)

            count_dict = {row['status']: row['count'] for row in counts}

            await conn.close()

            return {
                "proposals": [
                    {
                        "id": str(p['id']),
                        "title": p['title'],
                        "issue_title": p['issue_title'],
                        "target_file": p['target_file'],
                        "risk_assessment": p['risk_assessment'],
                        "status": p['status'],
                        "created_at": p['created_at'].isoformat()
                    }
                    for p in proposals
                ],
                "counts": {
                    "pending": count_dict.get('pending', 0),
                    "approved": count_dict.get('approved', 0),
                    "rejected": count_dict.get('rejected', 0),
                    "applied": count_dict.get('applied', 0)
                }
            }
        finally:
            await conn.close()

    except Exception as e:
        logger.error(f"Failed to list proposals: {e}")
        return {"error": str(e), "proposals": [], "counts": {}}

@app.get("/api/echo/proposals/{proposal_id}")
async def get_proposal(proposal_id: str):
    """Get full details of a specific proposal."""
    import asyncpg
    from uuid import UUID

    try:
        # Validate UUID
        UUID(proposal_id)

        db_url = os.getenv("DATABASE_URL", "postgresql://echo:echo_secure_password_123@localhost/echo_brain")
        conn = await asyncpg.connect(db_url)

        try:
            proposal = await conn.fetchrow("""
                SELECT
                    p.*,
                    i.title as issue_title,
                    i.description as issue_description
                FROM self_improvement_proposals p
                LEFT JOIN self_detected_issues i ON p.issue_id = i.id
                WHERE p.id = $1
            """, UUID(proposal_id))

            if not proposal:
                return {"error": "Proposal not found"}

            await conn.close()

            return {
                "id": str(proposal['id']),
                "title": proposal['title'],
                "description": proposal['description'],
                "target_file": proposal['target_file'],
                "current_code": proposal['current_code'],
                "proposed_code": proposal['proposed_code'],
                "reasoning": proposal['reasoning'],
                "risk_assessment": proposal['risk_assessment'],
                "status": proposal['status'],
                "reviewed_by": proposal['reviewed_by'],
                "reviewed_at": proposal['reviewed_at'].isoformat() if proposal['reviewed_at'] else None,
                "created_at": proposal['created_at'].isoformat(),
                "issue_title": proposal['issue_title'],
                "issue_description": proposal['issue_description']
            }
        finally:
            await conn.close()

    except ValueError:
        return {"error": "Invalid proposal ID"}
    except Exception as e:
        logger.error(f"Failed to get proposal: {e}")
        return {"error": str(e)}

@app.post("/api/echo/proposals/{proposal_id}/approve")
async def approve_proposal(proposal_id: str):
    """Approve a proposal for implementation."""
    import asyncpg
    from uuid import UUID
    from datetime import datetime

    try:
        UUID(proposal_id)

        db_url = os.getenv("DATABASE_URL", "postgresql://echo:echo_secure_password_123@localhost/echo_brain")
        conn = await asyncpg.connect(db_url)

        try:
            # Update proposal status
            updated = await conn.execute("""
                UPDATE self_improvement_proposals
                SET status = 'approved',
                    reviewed_by = 'patrick',
                    reviewed_at = NOW()
                WHERE id = $1 AND status = 'pending'
            """, UUID(proposal_id))

            if updated == "UPDATE 0":
                return {"error": "Proposal not found or already reviewed"}

            # Create audit log entry
            await conn.execute("""
                INSERT INTO autonomous_audit_log (action, entity_type, entity_id, details)
                VALUES ('proposal_approved', 'improvement_proposal', $1, $2)
            """, proposal_id, {"reviewed_by": "patrick", "action": "approved"})

            await conn.close()

            return {"status": "approved", "id": proposal_id}
        finally:
            await conn.close()

    except ValueError:
        return {"error": "Invalid proposal ID"}
    except Exception as e:
        logger.error(f"Failed to approve proposal: {e}")
        return {"error": str(e)}

@app.post("/api/echo/proposals/{proposal_id}/reject")
async def reject_proposal(proposal_id: str, reason: Optional[str] = None):
    """Reject a proposal."""
    import asyncpg
    from uuid import UUID

    try:
        UUID(proposal_id)

        db_url = os.getenv("DATABASE_URL", "postgresql://echo:echo_secure_password_123@localhost/echo_brain")
        conn = await asyncpg.connect(db_url)

        try:
            # Update proposal status
            updated = await conn.execute("""
                UPDATE self_improvement_proposals
                SET status = 'rejected',
                    reviewed_by = 'patrick',
                    reviewed_at = NOW()
                WHERE id = $1 AND status = 'pending'
            """, UUID(proposal_id))

            if updated == "UPDATE 0":
                return {"error": "Proposal not found or already reviewed"}

            # Create audit log entry
            await conn.execute("""
                INSERT INTO autonomous_audit_log (action, entity_type, entity_id, details)
                VALUES ('proposal_rejected', 'improvement_proposal', $1, $2)
            """, proposal_id, {"reviewed_by": "patrick", "action": "rejected", "reason": reason})

            await conn.close()

            return {"status": "rejected", "id": proposal_id}
        finally:
            await conn.close()

    except ValueError:
        return {"error": "Invalid proposal ID"}
    except Exception as e:
        logger.error(f"Failed to reject proposal: {e}")
        return {"error": str(e)}

# Mount static files for Vue frontend - MUST be last to serve as catch-all
frontend_path = Path("/opt/tower-echo-brain/frontend/dist")
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
    logger.info(f"✅ Frontend mounted from {frontend_path}")
else:
    logger.warning(f"⚠️ Frontend not found at {frontend_path}")

# Initialize Worker Scheduler on startup
@app.on_event("startup")
async def startup_event():
    """Initialize worker scheduler and register workers."""
    try:
        # Add project root to path if needed
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from src.autonomous.worker_scheduler import worker_scheduler

        # Track registered workers
        workers_registered = 0

        # Register existing workers
        try:
            from src.autonomous.workers.fact_extraction_worker import FactExtractionWorker
            worker = FactExtractionWorker()
            worker_scheduler.register_worker("fact_extraction", worker.run_cycle, interval_minutes=30)
            workers_registered += 1
            logger.info("✅ Registered fact_extraction worker (30 min)")
        except Exception as e:
            logger.error(f"❌ Failed to register fact_extraction: {e}")

        try:
            from src.autonomous.workers.conversation_watcher import ConversationWatcher
            worker = ConversationWatcher()
            worker_scheduler.register_worker("conversation_watcher", worker.run_cycle, interval_minutes=10)
            workers_registered += 1
            logger.info("✅ Registered conversation_watcher worker (10 min)")
        except Exception as e:
            logger.error(f"❌ Failed to register conversation_watcher: {e}")

        try:
            from src.autonomous.workers.knowledge_graph_builder import KnowledgeGraphBuilder
            worker = KnowledgeGraphBuilder()
            worker_scheduler.register_worker("knowledge_graph", worker.run_cycle, interval_minutes=1440)  # daily
            workers_registered += 1
            logger.info("✅ Registered knowledge_graph worker (daily)")
        except Exception as e:
            logger.error(f"❌ Failed to register knowledge_graph: {e}")

        # Register new Phase 2a self-awareness workers
        try:
            from src.autonomous.workers.codebase_indexer import CodebaseIndexer
            worker = CodebaseIndexer()
            worker_scheduler.register_worker("codebase_indexer", worker.run_cycle, interval_minutes=360)  # Every 6 hours
            workers_registered += 1
            logger.info("✅ Registered codebase_indexer worker (6 hours)")
        except Exception as e:
            logger.error(f"❌ Failed to register codebase_indexer: {e}")

        try:
            from src.autonomous.workers.schema_indexer import SchemaIndexer
            worker = SchemaIndexer()
            worker_scheduler.register_worker("schema_indexer", worker.run_cycle, interval_minutes=1440)  # Daily
            workers_registered += 1
            logger.info("✅ Registered schema_indexer worker (daily)")
        except Exception as e:
            logger.error(f"❌ Failed to register schema_indexer: {e}")

        try:
            from src.autonomous.workers.log_monitor import LogMonitor
            worker = LogMonitor()
            worker_scheduler.register_worker("log_monitor", worker.run_cycle, interval_minutes=15)  # Every 15 min
            workers_registered += 1
            logger.info("✅ Registered log_monitor worker (15 min)")
        except Exception as e:
            logger.error(f"❌ Failed to register log_monitor: {e}")

        try:
            from src.autonomous.workers.self_test_runner import SelfTestRunner
            worker = SelfTestRunner()
            worker_scheduler.register_worker("self_test_runner", worker.run_cycle, interval_minutes=60)  # Hourly
            workers_registered += 1
            logger.info("✅ Registered self_test_runner worker (hourly)")
        except Exception as e:
            logger.error(f"❌ Failed to register self_test_runner: {e}")

        try:
            from src.autonomous.workers.improvement_engine import ImprovementEngine
            worker = ImprovementEngine()
            worker_scheduler.register_worker("improvement_engine", worker.run_cycle, interval_minutes=120)  # Every 2 hours
            workers_registered += 1
            logger.info("✅ Registered improvement_engine worker (2 hours)")
        except Exception as e:
            logger.error(f"❌ Failed to register improvement_engine: {e}")

        # Start the scheduler
        await worker_scheduler.start()
        logger.info(f"✅ Worker scheduler started with {workers_registered} workers")
    except Exception as e:
        logger.error(f"❌ Failed to start worker scheduler: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop worker scheduler on shutdown."""
    try:
        from src.autonomous.worker_scheduler import worker_scheduler
        await worker_scheduler.stop()
        logger.info("✅ Worker scheduler stopped")
    except Exception as e:
        logger.error(f"❌ Error stopping worker scheduler: {e}")

# Enhanced request logging with metrics
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Skip logging for static assets
    if request.url.path.startswith("/assets/") or request.url.path.endswith(".js") or request.url.path.endswith(".css"):
        return await call_next(request)

    # Generate request ID
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    # Track request start time
    start_time = time.time()

    # Log incoming request
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"[{request_id}] → {request.method} {request.url.path} from {client_ip}")

    try:
        # Process request
        response = await call_next(request)

        # Calculate response time
        duration_ms = (time.time() - start_time) * 1000

        # Update metrics
        REQUEST_METRICS["total_requests"] += 1
        REQUEST_METRICS["response_times"].append(duration_ms)
        REQUEST_METRICS["requests_by_endpoint"][request.url.path] += 1

        # Track errors
        if 400 <= response.status_code < 500:
            REQUEST_METRICS["errors_4xx"] += 1
            REQUEST_METRICS["errors_by_endpoint"][request.url.path] += 1
        elif response.status_code >= 500:
            REQUEST_METRICS["errors_5xx"] += 1
            REQUEST_METRICS["errors_by_endpoint"][request.url.path] += 1

        # Track slow requests
        if duration_ms > 1000:  # Over 1 second
            REQUEST_METRICS["slowest_requests"].append({
                "path": request.url.path,
                "method": request.method,
                "duration_ms": duration_ms,
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id
            })

        # Log response
        logger.info(f"[{request_id}] ← {response.status_code} ({duration_ms:.2f}ms)")

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        return response

    except Exception as e:
        # Log exception
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"[{request_id}] ❌ Exception after {duration_ms:.2f}ms: {str(e)}")

        # Track error
        REQUEST_METRICS["errors_5xx"] += 1
        REQUEST_METRICS["errors_by_endpoint"][request.url.path] += 1

        # Store error for dashboard
        ERROR_LOG.append({
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "error": str(e),
            "traceback": traceback.format_exc()
        })

        # Return error response
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "request_id": request_id}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8309)

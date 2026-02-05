"""
Echo Brain Self-Diagnostic System
Comprehensive introspection of knowledge, codebase, and health.
"""
import asyncio
import asyncpg
import httpx
import os
import glob
import ast
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticResult:
    """Result from a diagnostic check"""
    category: str
    name: str
    status: str  # 'healthy', 'warning', 'critical', 'info'
    message: str
    details: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None


class EchoBrainDiagnostic:
    """
    Comprehensive self-diagnostic for Echo Brain.
    Tests knowledge sources, codebase quality, and provides recommendations.
    """
    
    def __init__(self):
        self.db_config = {
            "host": "localhost",
            "database": "echo_brain",
            "user": "patrick",
            "password": os.getenv("DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE")
        }
        self.results: List[DiagnosticResult] = []
        
    async def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run all diagnostics and return comprehensive report"""
        self.results = []
        
        start_time = datetime.now()
        
        # Run all diagnostic categories
        await self._diagnose_knowledge_sources()
        await self._diagnose_unified_layer()
        await self._diagnose_mcp_integration()
        await self._diagnose_api_endpoints()
        await self._diagnose_codebase()
        await self._diagnose_data_quality()
        await self._analyze_usage_patterns()
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Calculate health score
        health_score = self._calculate_health_score()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "health_score": health_score,
            "summary": self._generate_summary(),
            "results": [asdict(r) for r in self.results],
            "recommendations": recommendations,
            "quick_stats": await self._get_quick_stats()
        }
    
    # ==================== KNOWLEDGE SOURCES ====================
    
    async def _diagnose_knowledge_sources(self):
        """Check all knowledge data sources"""
        
        # PostgreSQL Facts
        try:
            pool = await asyncpg.create_pool(**self.db_config, min_size=1, max_size=2)
            async with pool.acquire() as conn:
                fact_count = await conn.fetchval("SELECT COUNT(*) FROM facts")
                recent_facts = await conn.fetchval(
                    "SELECT COUNT(*) FROM facts WHERE created_at > NOW() - INTERVAL '7 days'"
                )
                
                if fact_count > 5000:
                    status = "healthy"
                    msg = f"{fact_count:,} facts available"
                elif fact_count > 1000:
                    status = "warning"
                    msg = f"Only {fact_count:,} facts - consider extracting more"
                else:
                    status = "critical"
                    msg = f"Low fact count: {fact_count:,}"
                
                self.results.append(DiagnosticResult(
                    category="knowledge",
                    name="PostgreSQL Facts",
                    status=status,
                    message=msg,
                    details={"total": fact_count, "recent_7d": recent_facts},
                    recommendation="Run fact extraction on recent conversations" if recent_facts < 100 else None
                ))
            await pool.close()
        except Exception as e:
            self.results.append(DiagnosticResult(
                category="knowledge",
                name="PostgreSQL Facts",
                status="critical",
                message=f"Database error: {e}",
                recommendation="Check PostgreSQL connection"
            ))
        
        # Qdrant Vectors
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get("http://localhost:6333/collections/echo_memory")
                data = r.json()
                vector_count = data.get("result", {}).get("points_count", 0)
                
                if vector_count > 20000:
                    status = "healthy"
                elif vector_count > 5000:
                    status = "warning"
                else:
                    status = "critical"
                
                self.results.append(DiagnosticResult(
                    category="knowledge",
                    name="Qdrant Vectors",
                    status=status,
                    message=f"{vector_count:,} vectors in echo_memory",
                    details={"vector_count": vector_count, "status": data.get("result", {}).get("status")}
                ))
        except Exception as e:
            self.results.append(DiagnosticResult(
                category="knowledge",
                name="Qdrant Vectors",
                status="critical",
                message=f"Qdrant error: {e}"
            ))
        
        # Conversations
        try:
            pool = await asyncpg.create_pool(**self.db_config, min_size=1, max_size=2)
            async with pool.acquire() as conn:
                conv_count = await conn.fetchval("SELECT COUNT(*) FROM claude_conversations")
                recent_conv = await conn.fetchval(
                    "SELECT COUNT(*) FROM claude_conversations WHERE created_at > NOW() - INTERVAL '24 hours'"
                )
                
                self.results.append(DiagnosticResult(
                    category="knowledge",
                    name="Conversations",
                    status="healthy" if conv_count > 10000 else "warning",
                    message=f"{conv_count:,} conversations indexed",
                    details={"total": conv_count, "last_24h": recent_conv}
                ))
            await pool.close()
        except Exception as e:
            self.results.append(DiagnosticResult(
                category="knowledge",
                name="Conversations",
                status="critical",
                message=f"Error: {e}"
            ))
    
    # ==================== UNIFIED LAYER ====================
    
    async def _diagnose_unified_layer(self):
        """Test the unified knowledge layer integration"""
        
        try:
            from src.core.unified_knowledge import get_unified_knowledge
            knowledge = get_unified_knowledge()
            
            # Test context retrieval with a known question
            test_query = "What port does Echo Brain run on?"
            context = await knowledge.get_context(test_query, max_facts=3, max_vectors=2, max_conversations=2)
            
            # Check if core facts are working
            core_facts_found = any(f.source_type == 'core' for f in context['facts'])
            db_facts_found = any(f.source_type == 'fact' for f in context['facts'])
            vectors_found = len(context['vectors']) > 0
            convs_found = len(context['conversations']) > 0
            
            issues = []
            if not core_facts_found:
                issues.append("Core facts not returning")
            if not db_facts_found:
                issues.append("Database facts not returning")
            if not vectors_found:
                issues.append("Vector search not returning results")
            if not convs_found:
                issues.append("Conversation search not returning results")
            
            # Check if the answer would be correct
            answer_components = [f.content.lower() for f in context['facts']]
            knows_port = any('8309' in c for c in answer_components)
            
            if not issues and knows_port:
                status = "healthy"
                msg = "Unified layer working - all sources connected"
            elif knows_port:
                status = "warning"
                msg = f"Partial issues: {', '.join(issues)}"
            else:
                status = "critical"
                msg = "Unified layer broken - cannot answer basic questions"
            
            self.results.append(DiagnosticResult(
                category="integration",
                name="Unified Knowledge Layer",
                status=status,
                message=msg,
                details={
                    "core_facts": core_facts_found,
                    "db_facts": db_facts_found,
                    "vectors": vectors_found,
                    "conversations": convs_found,
                    "knows_own_port": knows_port,
                    "total_sources": context['total_sources']
                },
                recommendation=issues[0] if issues else None
            ))
            
            await knowledge.close()
            
        except Exception as e:
            self.results.append(DiagnosticResult(
                category="integration",
                name="Unified Knowledge Layer",
                status="critical",
                message=f"Import/execution error: {e}",
                recommendation="Check unified_knowledge.py for syntax errors"
            ))
    
    # ==================== MCP INTEGRATION ====================
    
    async def _diagnose_mcp_integration(self):
        """Test MCP service methods"""
        
        try:
            from src.integrations.mcp_service import mcp_service
            
            # Test search_memory
            memory_results = await mcp_service.search_memory("echo brain", limit=3)
            memory_ok = len(memory_results) > 0
            
            # Test get_facts
            facts_results = await mcp_service.get_facts("echo brain", limit=3)
            facts_ok = len(facts_results) > 0
            
            # Test get_vector_count
            vector_count = mcp_service.get_vector_count()
            count_ok = vector_count > 0
            
            all_ok = memory_ok and facts_ok and count_ok
            
            self.results.append(DiagnosticResult(
                category="integration",
                name="MCP Service",
                status="healthy" if all_ok else "warning",
                message=f"search_memory: {'✓' if memory_ok else '✗'}, get_facts: {'✓' if facts_ok else '✗'}, vector_count: {'✓' if count_ok else '✗'}",
                details={
                    "search_memory": memory_ok,
                    "get_facts": facts_ok,
                    "vector_count": vector_count
                }
            ))
        except Exception as e:
            self.results.append(DiagnosticResult(
                category="integration",
                name="MCP Service",
                status="critical",
                message=f"MCP error: {e}"
            ))
    
    # ==================== API ENDPOINTS ====================
    
    async def _diagnose_api_endpoints(self):
        """Test critical API endpoints"""

        endpoints = [
            ("GET", "/api/echo/health", None),
            ("POST", "/api/echo/ask", {"question": "test", "use_context": False}),
            ("POST", "/api/echo/memory/search", {"query": "test", "limit": 1}),
            ("POST", "/api/echo/intelligence/think", {"query": "test", "depth": 1}),
            ("GET", "/api/echo/system/resources", None),
            ("GET", "/api/echo/brain", None),
        ]
        
        working = 0
        failed = []
        
        async with httpx.AsyncClient(timeout=30) as client:
            for method, path, body in endpoints:
                try:
                    url = f"http://localhost:8309{path}"
                    if method == "GET":
                        r = await client.get(url)
                    else:
                        r = await client.post(url, json=body)
                    
                    if r.status_code == 200:
                        working += 1
                    else:
                        failed.append(f"{path} ({r.status_code})")
                except Exception as e:
                    failed.append(f"{path} ({e})")
        
        total = len(endpoints)
        if working == total:
            status = "healthy"
        elif working >= total * 0.7:
            status = "warning"
        else:
            status = "critical"
        
        self.results.append(DiagnosticResult(
            category="api",
            name="API Endpoints",
            status=status,
            message=f"{working}/{total} endpoints responding",
            details={"working": working, "total": total, "failed": failed},
            recommendation=f"Fix: {failed[0]}" if failed else None
        ))
    
    # ==================== CODEBASE ANALYSIS ====================
    
    async def _diagnose_codebase(self):
        """Analyze codebase for issues"""
        
        base_path = "/opt/tower-echo-brain/src"
        issues = []
        stats = {"files": 0, "lines": 0, "functions": 0, "classes": 0}
        
        for py_file in glob.glob(f"{base_path}/**/*.py", recursive=True):
            if "__pycache__" in py_file:
                continue
            
            stats["files"] += 1
            
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    stats["lines"] += len(content.split('\n'))
                
                # Parse AST to count functions/classes
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                            stats["functions"] += 1
                        elif isinstance(node, ast.ClassDef):
                            stats["classes"] += 1
                except:
                    pass
                
                # Check for common issues
                if "localhost:8309" in content and "localhost" not in py_file:
                    # Hardcoded URLs are fine for internal services
                    pass
                
                if "TODO" in content or "FIXME" in content:
                    count = content.count("TODO") + content.count("FIXME")
                    if count > 5:
                        issues.append(f"{os.path.basename(py_file)}: {count} TODO/FIXME comments")
                
                if "import *" in content:
                    issues.append(f"{os.path.basename(py_file)}: wildcard import")
                    
            except Exception as e:
                issues.append(f"Error reading {py_file}: {e}")
        
        status = "healthy" if len(issues) < 3 else "warning" if len(issues) < 10 else "critical"
        
        self.results.append(DiagnosticResult(
            category="codebase",
            name="Code Quality",
            status=status,
            message=f"{stats['files']} files, {stats['lines']:,} lines, {len(issues)} issues",
            details={**stats, "issues": issues[:5]},  # Limit to 5 issues
            recommendation=issues[0] if issues else None
        ))
    
    # ==================== DATA QUALITY ====================
    
    async def _diagnose_data_quality(self):
        """Check quality of stored data"""
        
        try:
            pool = await asyncpg.create_pool(**self.db_config, min_size=1, max_size=2)
            async with pool.acquire() as conn:
                # Check for duplicate facts
                duplicates = await conn.fetchval("""
                    SELECT COUNT(*) FROM (
                        SELECT subject, predicate, object, COUNT(*)
                        FROM facts
                        GROUP BY subject, predicate, object
                        HAVING COUNT(*) > 1
                    ) as dupes
                """)
                
                # Check for low-confidence facts
                low_conf = await conn.fetchval(
                    "SELECT COUNT(*) FROM facts WHERE confidence < 0.5"
                )
                
                # Check for empty content in conversations
                empty_conv = await conn.fetchval(
                    "SELECT COUNT(*) FROM claude_conversations WHERE content IS NULL OR content = ''"
                )
                
                # Check for stale data
                oldest_fact = await conn.fetchval(
                    "SELECT MIN(created_at) FROM facts"
                )
                
                issues = []
                if duplicates > 100:
                    issues.append(f"{duplicates} duplicate facts")
                if low_conf > 500:
                    issues.append(f"{low_conf} low-confidence facts")
                if empty_conv > 0:
                    issues.append(f"{empty_conv} empty conversations")
                
                status = "healthy" if not issues else "warning"
                
                self.results.append(DiagnosticResult(
                    category="data",
                    name="Data Quality",
                    status=status,
                    message=f"{len(issues)} quality issues found" if issues else "Data quality good",
                    details={
                        "duplicate_facts": duplicates,
                        "low_confidence_facts": low_conf,
                        "empty_conversations": empty_conv,
                        "oldest_fact": oldest_fact.isoformat() if oldest_fact else None
                    },
                    recommendation=f"Clean up: {issues[0]}" if issues else None
                ))
            await pool.close()
        except Exception as e:
            self.results.append(DiagnosticResult(
                category="data",
                name="Data Quality",
                status="warning",
                message=f"Could not analyze: {e}"
            ))
    
    # ==================== USAGE PATTERNS ====================
    
    async def _analyze_usage_patterns(self):
        """Analyze how Echo Brain is being used"""
        
        try:
            pool = await asyncpg.create_pool(**self.db_config, min_size=1, max_size=2)
            async with pool.acquire() as conn:
                # Get recent query patterns
                recent_queries = await conn.fetch("""
                    SELECT 
                        date_trunc('hour', created_at) as hour,
                        COUNT(*) as queries
                    FROM claude_conversations
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                    GROUP BY hour
                    ORDER BY hour DESC
                    LIMIT 24
                """)
                
                # Most common topics in facts
                top_subjects = await conn.fetch("""
                    SELECT subject, COUNT(*) as cnt
                    FROM facts
                    GROUP BY subject
                    ORDER BY cnt DESC
                    LIMIT 5
                """)
                
                total_24h = sum(r['queries'] for r in recent_queries)
                
                self.results.append(DiagnosticResult(
                    category="usage",
                    name="Usage Patterns",
                    status="info",
                    message=f"{total_24h} queries in last 24h",
                    details={
                        "queries_24h": total_24h,
                        "top_subjects": [dict(r) for r in top_subjects],
                        "hourly_breakdown": [dict(r) for r in recent_queries[:6]]
                    }
                ))
            await pool.close()
        except Exception as e:
            self.results.append(DiagnosticResult(
                category="usage",
                name="Usage Patterns",
                status="info",
                message=f"Could not analyze: {e}"
            ))
    
    # ==================== SUMMARY & RECOMMENDATIONS ====================
    
    def _calculate_health_score(self) -> float:
        """Calculate overall health score (0-100)"""
        if not self.results:
            return 0
        
        weights = {"critical": 0, "warning": 0.6, "healthy": 1.0, "info": 1.0}
        scores = [weights.get(r.status, 0.5) for r in self.results if r.status != "info"]
        
        return round(sum(scores) / len(scores) * 100, 1) if scores else 100
    
    def _generate_summary(self) -> Dict[str, int]:
        """Generate status summary"""
        summary = {"healthy": 0, "warning": 0, "critical": 0, "info": 0}
        for r in self.results:
            summary[r.status] = summary.get(r.status, 0) + 1
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate prioritized recommendations"""
        recommendations = []
        
        # Collect recommendations from results
        for r in self.results:
            if r.recommendation and r.status in ("critical", "warning"):
                priority = "🔴" if r.status == "critical" else "🟡"
                recommendations.append(f"{priority} [{r.category}] {r.recommendation}")
        
        # Add general recommendations based on patterns
        facts_result = next((r for r in self.results if r.name == "PostgreSQL Facts"), None)
        if facts_result and facts_result.details:
            recent = facts_result.details.get("recent_7d", 0)
            if recent < 50:
                recommendations.append("🟡 [knowledge] Run fact extraction on recent conversations to keep knowledge fresh")
        
        return recommendations[:10]  # Top 10 recommendations
    
    async def _get_quick_stats(self) -> Dict[str, Any]:
        """Get quick stats for dashboard"""
        stats = {}
        
        for r in self.results:
            if r.name == "PostgreSQL Facts" and r.details:
                stats["facts_total"] = r.details.get("total", 0)
            elif r.name == "Qdrant Vectors" and r.details:
                stats["vectors_total"] = r.details.get("vector_count", 0)
            elif r.name == "Conversations" and r.details:
                stats["conversations_total"] = r.details.get("total", 0)
            elif r.name == "Code Quality" and r.details:
                stats["codebase_lines"] = r.details.get("lines", 0)
        
        return stats


# Singleton instance
_diagnostic = None

def get_diagnostic() -> EchoBrainDiagnostic:
    global _diagnostic
    if not _diagnostic:
        _diagnostic = EchoBrainDiagnostic()
    return _diagnostic


async def run_diagnostic() -> Dict[str, Any]:
    """Convenience function to run full diagnostic"""
    diagnostic = get_diagnostic()
    return await diagnostic.run_full_diagnostic()

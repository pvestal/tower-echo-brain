"""
Echo Brain Deep Self-Diagnostic
Actually tests intelligence, verifies correctness, identifies gaps.
"""
import asyncio
import asyncpg
import httpx
import os
import re
import ast
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A self-test with expected answer"""
    question: str
    expected_contains: List[str]  # Answer must contain these
    expected_not_contains: List[str] = field(default_factory=list)  # Must NOT contain
    category: str = "general"


@dataclass 
class DiagnosticFinding:
    severity: str  # critical, warning, info, success
    category: str
    title: str
    details: str
    fix: Optional[str] = None  # Actionable fix
    auto_fixable: bool = False


class DeepDiagnostic:
    """
    Deep introspective diagnostic that:
    1. Tests if Echo Brain can answer questions about itself correctly
    2. Verifies knowledge retrieval actually improves answers
    3. Identifies knowledge gaps
    4. Analyzes codebase for real issues
    5. Provides specific, actionable fixes
    """
    
    # Questions Echo Brain MUST answer correctly about itself
    SELF_KNOWLEDGE_TESTS = [
        TestCase(
            question="What port does Echo Brain run on?",
            expected_contains=["8309"],
            category="self-knowledge"
        ),
        TestCase(
            question="What database does Echo Brain use?",
            expected_contains=["postgresql", "echo_brain"],
            category="self-knowledge"
        ),
        TestCase(
            question="What vector database does Echo Brain use?",
            expected_contains=["qdrant", "6333"],
            category="self-knowledge"
        ),
        TestCase(
            question="What embedding model does Echo Brain use?",
            expected_contains=["mxbai"],
            category="self-knowledge"
        ),
        TestCase(
            question="What LLM models can Echo Brain use?",
            expected_contains=["mistral", "ollama"],
            category="self-knowledge"
        ),
        TestCase(
            question="How many vectors does Echo Brain have?",
            expected_contains=["24", "000"],  # Should mention 24k+
            category="self-knowledge"
        ),
        TestCase(
            question="What is the Echo Brain API prefix?",
            expected_contains=["/api/echo"],
            category="self-knowledge"
        ),
    ]
    
    def __init__(self):
        self.db_config = {
            "host": "localhost",
            "database": "echo_brain",
            "user": "patrick",
            "password": os.getenv("DB_PASSWORD", "RP78eIrW7cI2jYvL5akt1yurE")
        }
        self.findings: List[DiagnosticFinding] = []
        self.test_results: List[Dict] = []
        
    async def run_deep_diagnostic(self) -> Dict[str, Any]:
        """Run comprehensive deep diagnostic"""
        self.findings = []
        self.test_results = []
        start = datetime.now()
        
        # 1. Test self-knowledge (can Echo Brain answer questions about itself?)
        await self._test_self_knowledge()
        
        # 2. Test context impact (does context actually help?)
        await self._test_context_impact()
        
        # 3. Identify knowledge gaps
        await self._identify_knowledge_gaps()
        
        # 4. Analyze fact quality
        await self._analyze_fact_quality()
        
        # 5. Check for stale/outdated knowledge
        await self._check_knowledge_freshness()
        
        # 6. Analyze codebase for real issues
        await self._deep_code_analysis()
        
        # 7. Check endpoint consistency
        await self._check_endpoint_consistency()
        
        # 8. Analyze error patterns
        await self._analyze_error_patterns()
        
        # Calculate scores
        self_knowledge_score = self._calculate_self_knowledge_score()
        
        elapsed = (datetime.now() - start).total_seconds()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "scores": {
                "self_knowledge": self_knowledge_score,
                "overall": self._calculate_overall_score()
            },
            "test_results": self.test_results,
            "findings": [asdict(f) for f in self.findings],
            "summary": self._generate_summary(),
            "action_items": self._generate_action_items()
        }
    
    # ==================== SELF-KNOWLEDGE TESTS ====================
    
    async def _test_self_knowledge(self):
        """Test if Echo Brain can correctly answer questions about itself"""
        
        async with httpx.AsyncClient(timeout=60) as client:
            for test in self.SELF_KNOWLEDGE_TESTS:
                try:
                    # Ask with context
                    r = await client.post(
                        "http://localhost:8309/api/echo/ask",
                        json={"question": test.question, "use_context": True, "verbose": True}
                    )
                    
                    if r.status_code != 200:
                        self.test_results.append({
                            "question": test.question,
                            "passed": False,
                            "error": f"HTTP {r.status_code}",
                            "category": test.category
                        })
                        continue
                    
                    data = r.json()
                    answer = data.get("answer", "").lower()
                    
                    # Check expected content
                    missing = [exp for exp in test.expected_contains if exp.lower() not in answer]
                    forbidden = [exp for exp in test.expected_not_contains if exp.lower() in answer]
                    
                    passed = len(missing) == 0 and len(forbidden) == 0
                    
                    self.test_results.append({
                        "question": test.question,
                        "passed": passed,
                        "answer_preview": answer[:200],
                        "missing": missing,
                        "forbidden_found": forbidden,
                        "sources_used": len(data.get("sources", [])),
                        "category": test.category
                    })
                    
                    if not passed:
                        self.findings.append(DiagnosticFinding(
                            severity="warning",
                            category="self-knowledge",
                            title=f"Cannot answer: {test.question}",
                            details=f"Missing: {missing}, Answer: {answer[:100]}...",
                            fix=f"Add fact: Echo Brain {' '.join(test.expected_contains)}"
                        ))
                        
                except Exception as e:
                    self.test_results.append({
                        "question": test.question,
                        "passed": False,
                        "error": str(e),
                        "category": test.category
                    })
    
    # ==================== CONTEXT IMPACT TEST ====================
    
    async def _test_context_impact(self):
        """Test if using context actually improves answers"""
        
        test_question = "What port does Echo Brain run on?"
        
        async with httpx.AsyncClient(timeout=60) as client:
            # Without context
            r1 = await client.post(
                "http://localhost:8309/api/echo/ask",
                json={"question": test_question, "use_context": False}
            )
            answer_no_ctx = r1.json().get("answer", "").lower() if r1.status_code == 200 else ""
            
            # With context
            r2 = await client.post(
                "http://localhost:8309/api/echo/ask",
                json={"question": test_question, "use_context": True}
            )
            data_ctx = r2.json() if r2.status_code == 200 else {}
            answer_ctx = data_ctx.get("answer", "").lower()
            sources = len(data_ctx.get("sources", []))
        
        # Check if context helped
        correct_no_ctx = "8309" in answer_no_ctx
        correct_ctx = "8309" in answer_ctx
        
        if correct_ctx and not correct_no_ctx:
            self.findings.append(DiagnosticFinding(
                severity="success",
                category="context-impact",
                title="Context improves answers",
                details=f"Without context: wrong. With context ({sources} sources): correct.",
                fix=None
            ))
        elif correct_ctx and correct_no_ctx:
            self.findings.append(DiagnosticFinding(
                severity="info",
                category="context-impact",
                title="LLM already knows this",
                details="Answer correct with and without context",
                fix=None
            ))
        elif not correct_ctx:
            self.findings.append(DiagnosticFinding(
                severity="critical",
                category="context-impact",
                title="Context not helping",
                details=f"Even with {sources} sources, answer is wrong: {answer_ctx[:100]}",
                fix="Check if facts contain '8309' and if they're being retrieved"
            ))
    
    # ==================== KNOWLEDGE GAPS ====================
    
    async def _identify_knowledge_gaps(self):
        """Identify what Echo Brain should know but doesn't"""
        
        # Things Echo Brain should have facts about
        expected_subjects = [
            "Echo Brain",
            "Tower System", 
            "Qdrant",
            "PostgreSQL",
            "Ollama",
            "unified knowledge",
            "MCP",
            "vector",
            "embedding"
        ]
        
        try:
            pool = await asyncpg.create_pool(**self.db_config, min_size=1, max_size=2)
            async with pool.acquire() as conn:
                gaps = []
                for subject in expected_subjects:
                    count = await conn.fetchval(
                        "SELECT COUNT(*) FROM facts WHERE subject ILIKE $1 OR object ILIKE $1",
                        f"%{subject}%"
                    )
                    if count < 10:
                        gaps.append((subject, count))
                
                if gaps:
                    self.findings.append(DiagnosticFinding(
                        severity="warning",
                        category="knowledge-gaps",
                        title=f"Knowledge gaps in {len(gaps)} areas",
                        details=f"Low fact counts: {', '.join([f'{s}({c})' for s,c in gaps])}",
                        fix=f"Extract more facts about: {', '.join([s for s,c in gaps])}"
                    ))
                else:
                    self.findings.append(DiagnosticFinding(
                        severity="success",
                        category="knowledge-gaps",
                        title="No major knowledge gaps",
                        details="All expected subjects have adequate fact coverage",
                        fix=None
                    ))
            await pool.close()
        except Exception as e:
            logger.error(f"Knowledge gap analysis failed: {e}")
    
    # ==================== FACT QUALITY ====================
    
    async def _analyze_fact_quality(self):
        """Analyze quality of stored facts"""
        
        try:
            pool = await asyncpg.create_pool(**self.db_config, min_size=1, max_size=2)
            async with pool.acquire() as conn:
                # Check for garbage facts (too short, too generic)
                short_facts = await conn.fetchval(
                    "SELECT COUNT(*) FROM facts WHERE LENGTH(object) < 5"
                )
                
                # Check for duplicate facts
                duplicates = await conn.fetchval("""
                    SELECT COUNT(*) FROM (
                        SELECT subject, predicate, object
                        FROM facts
                        GROUP BY subject, predicate, object
                        HAVING COUNT(*) > 1
                    ) as dupes
                """)
                
                # Check for contradictory facts (same subject+predicate, different object)
                contradictions = await conn.fetch("""
                    SELECT subject, predicate, COUNT(DISTINCT object) as variants
                    FROM facts
                    WHERE confidence > 0.7
                    GROUP BY subject, predicate
                    HAVING COUNT(DISTINCT object) > 1
                    LIMIT 5
                """)
                
                # Check for low confidence facts
                low_conf = await conn.fetchval(
                    "SELECT COUNT(*) FROM facts WHERE confidence < 0.5"
                )
                
                issues = []
                if short_facts > 100:
                    issues.append(f"{short_facts} very short facts (< 5 chars)")
                if duplicates > 50:
                    issues.append(f"{duplicates} duplicate facts")
                if len(contradictions) > 0:
                    issues.append(f"{len(contradictions)} potential contradictions")
                if low_conf > 500:
                    issues.append(f"{low_conf} low-confidence facts")
                
                if issues:
                    self.findings.append(DiagnosticFinding(
                        severity="warning",
                        category="fact-quality",
                        title="Fact quality issues",
                        details="; ".join(issues),
                        fix="Run: DELETE FROM facts WHERE LENGTH(object) < 5 OR confidence < 0.3",
                        auto_fixable=True
                    ))
                    
                    if contradictions:
                        for c in contradictions[:3]:
                            self.findings.append(DiagnosticFinding(
                                severity="info",
                                category="fact-quality",
                                title=f"Contradiction: {c['subject']} {c['predicate']}",
                                details=f"Has {c['variants']} different values",
                                fix=f"Review facts WHERE subject='{c['subject']}' AND predicate='{c['predicate']}'"
                            ))
                            
            await pool.close()
        except Exception as e:
            logger.error(f"Fact quality analysis failed: {e}")
    
    # ==================== KNOWLEDGE FRESHNESS ====================
    
    async def _check_knowledge_freshness(self):
        """Check if knowledge is stale"""
        
        try:
            pool = await asyncpg.create_pool(**self.db_config, min_size=1, max_size=2)
            async with pool.acquire() as conn:
                # Facts added in last 7 days
                recent_facts = await conn.fetchval(
                    "SELECT COUNT(*) FROM facts WHERE created_at > NOW() - INTERVAL '7 days'"
                )
                
                # Conversations in last 24 hours
                recent_convs = await conn.fetchval(
                    "SELECT COUNT(*) FROM claude_conversations WHERE created_at > NOW() - INTERVAL '24 hours'"
                )
                
                # Oldest fact
                oldest = await conn.fetchval("SELECT MIN(created_at) FROM facts")
                
                if recent_facts < 10:
                    self.findings.append(DiagnosticFinding(
                        severity="warning",
                        category="freshness",
                        title="Knowledge going stale",
                        details=f"Only {recent_facts} facts added in last 7 days",
                        fix="Run fact extraction on recent conversations"
                    ))
                    
                if recent_convs > 100 and recent_facts < 10:
                    self.findings.append(DiagnosticFinding(
                        severity="warning",
                        category="freshness",
                        title="Conversations not being processed",
                        details=f"{recent_convs} conversations in 24h but only {recent_facts} facts extracted",
                        fix="Enable continuous fact extraction"
                    ))
                    
            await pool.close()
        except Exception as e:
            logger.error(f"Freshness check failed: {e}")
    
    # ==================== DEEP CODE ANALYSIS ====================
    
    async def _deep_code_analysis(self):
        """Analyze codebase for real architectural issues"""
        
        base_path = "/opt/tower-echo-brain/src"
        issues = []
        
        # Check for common problems
        patterns_to_find = {
            r"localhost:8309": "Hardcoded port (acceptable for internal)",
            r"password\s*=\s*['\"]": "Hardcoded password",
            r"except:\s*$": "Bare except clause",
            r"# TODO|# FIXME|# HACK": "Technical debt marker",
            r"print\(": "Debug print statement",
            r"import \*": "Wildcard import",
        }
        
        critical_issues = []
        warnings = []
        
        for root, dirs, files in os.walk(base_path):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            
            for file in files:
                if not file.endswith(".py"):
                    continue
                    
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, base_path)
                
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        lines = content.split('\n')
                    
                    for pattern, desc in patterns_to_find.items():
                        matches = re.findall(pattern, content, re.MULTILINE)
                        if matches:
                            if "password" in pattern.lower():
                                critical_issues.append(f"{rel_path}: {desc} ({len(matches)}x)")
                            elif "except:" in pattern or "print(" in pattern:
                                warnings.append(f"{rel_path}: {desc} ({len(matches)}x)")
                    
                    # Check for unused imports using AST
                    try:
                        tree = ast.parse(content)
                        imports = set()
                        used_names = set()
                        
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    imports.add(alias.asname or alias.name)
                            elif isinstance(node, ast.ImportFrom):
                                for alias in node.names:
                                    if alias.name != '*':
                                        imports.add(alias.asname or alias.name)
                            elif isinstance(node, ast.Name):
                                used_names.add(node.id)
                        
                        unused = imports - used_names
                        if len(unused) > 5:
                            warnings.append(f"{rel_path}: {len(unused)} potentially unused imports")
                    except:
                        pass
                        
                except Exception as e:
                    pass
        
        if critical_issues:
            self.findings.append(DiagnosticFinding(
                severity="critical",
                category="code-quality",
                title=f"{len(critical_issues)} critical code issues",
                details="; ".join(critical_issues[:3]),
                fix="Move credentials to environment variables"
            ))
        
        if warnings:
            self.findings.append(DiagnosticFinding(
                severity="warning",
                category="code-quality",
                title=f"{len(warnings)} code warnings",
                details="; ".join(warnings[:5]),
                fix="Review and clean up flagged code patterns"
            ))
    
    # ==================== ENDPOINT CONSISTENCY ====================
    
    async def _check_endpoint_consistency(self):
        """Check if all endpoints return consistent response formats"""
        
        endpoints = [
            ("POST", "/api/echo/ask", {"question": "test"}, ["answer", "model"]),
            ("POST", "/api/echo/memory/search", {"query": "test", "limit": 1}, []),
            ("GET", "/api/echo/health", None, ["status"]),
            ("GET", "/api/echo/brain", None, ["activity"]),
        ]
        
        inconsistencies = []
        
        async with httpx.AsyncClient(timeout=30) as client:
            for method, path, body, expected_keys in endpoints:
                try:
                    url = f"http://localhost:8309{path}"
                    if method == "GET":
                        r = await client.get(url)
                    else:
                        r = await client.post(url, json=body)
                    
                    if r.status_code == 200:
                        data = r.json()
                        missing = [k for k in expected_keys if k not in data]
                        if missing:
                            inconsistencies.append(f"{path}: missing {missing}")
                    else:
                        inconsistencies.append(f"{path}: HTTP {r.status_code}")
                        
                except Exception as e:
                    inconsistencies.append(f"{path}: {str(e)[:50]}")
        
        if inconsistencies:
            self.findings.append(DiagnosticFinding(
                severity="warning",
                category="api-consistency",
                title="API response inconsistencies",
                details="; ".join(inconsistencies),
                fix="Standardize response formats"
            ))
    
    # ==================== ERROR PATTERNS ====================
    
    async def _analyze_error_patterns(self):
        """Analyze recent errors for patterns"""
        
        import subprocess
        
        try:
            result = subprocess.run(
                ["sudo", "journalctl", "-u", "tower-echo-brain", "-n", "500", "--no-pager"],
                capture_output=True, text=True, timeout=10
            )
            
            logs = result.stdout
            
            # Count error types
            error_patterns = {
                "timeout": len(re.findall(r"timeout|timed out", logs, re.I)),
                "connection": len(re.findall(r"connection refused|cannot connect", logs, re.I)),
                "memory": len(re.findall(r"out of memory|memory error", logs, re.I)),
                "database": len(re.findall(r"database error|sql error|asyncpg", logs, re.I)),
                "import": len(re.findall(r"import error|module not found", logs, re.I)),
            }
            
            significant = {k: v for k, v in error_patterns.items() if v > 5}
            
            if significant:
                worst = max(significant.items(), key=lambda x: x[1])
                self.findings.append(DiagnosticFinding(
                    severity="warning",
                    category="error-patterns",
                    title=f"Recurring {worst[0]} errors ({worst[1]}x)",
                    details=f"Error counts: {significant}",
                    fix=f"Investigate {worst[0]} errors in logs"
                ))
                
        except Exception as e:
            pass
    
    # ==================== SCORING & SUMMARY ====================
    
    def _calculate_self_knowledge_score(self) -> float:
        """Calculate score for self-knowledge tests"""
        if not self.test_results:
            return 0
        
        passed = sum(1 for t in self.test_results if t.get("passed"))
        return round(passed / len(self.test_results) * 100, 1)
    
    def _calculate_overall_score(self) -> float:
        """Calculate overall health score"""
        if not self.findings:
            return 100
        
        weights = {"critical": -20, "warning": -5, "info": 0, "success": 5}
        score = 100
        for f in self.findings:
            score += weights.get(f.severity, 0)
        
        return max(0, min(100, round(score, 1)))
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate findings summary"""
        return {
            "critical": len([f for f in self.findings if f.severity == "critical"]),
            "warning": len([f for f in self.findings if f.severity == "warning"]),
            "info": len([f for f in self.findings if f.severity == "info"]),
            "success": len([f for f in self.findings if f.severity == "success"]),
            "tests_passed": sum(1 for t in self.test_results if t.get("passed")),
            "tests_failed": sum(1 for t in self.test_results if not t.get("passed")),
        }
    
    def _generate_action_items(self) -> List[Dict[str, Any]]:
        """Generate prioritized action items"""
        actions = []
        
        # Add actions from findings with fixes
        for f in self.findings:
            if f.fix and f.severity in ("critical", "warning"):
                actions.append({
                    "priority": 1 if f.severity == "critical" else 2,
                    "category": f.category,
                    "action": f.fix,
                    "auto_fixable": f.auto_fixable
                })
        
        # Add actions from failed tests
        for t in self.test_results:
            if not t.get("passed") and t.get("missing"):
                actions.append({
                    "priority": 2,
                    "category": "self-knowledge",
                    "action": f"Add fact containing: {', '.join(t['missing'])}",
                    "auto_fixable": False
                })
        
        # Sort by priority
        actions.sort(key=lambda x: x["priority"])
        
        return actions[:10]  # Top 10 actions


async def run_deep_diagnostic() -> Dict[str, Any]:
    """Run deep diagnostic"""
    diag = DeepDiagnostic()
    return await diag.run_deep_diagnostic()

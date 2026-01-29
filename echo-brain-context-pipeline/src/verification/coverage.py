"""
Verification & Coverage Testing

Ensures complete ingestion and fact extraction across all sources.
This is the "trust but verify" layer that catches gaps.
"""

import asyncio
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import asyncpg

from ..context_assembly.models import CoverageReport, IngestionGap, Domain, SourceType


@dataclass
class VerificationConfig:
    """Configuration for verification runs."""
    postgres_dsn: str
    
    # Directories to scan for sources
    source_directories: dict[str, str] = None  # {"documents": "/path/to/docs", ...}
    
    # Quality sampling
    sample_size: int = 100  # How many vectors to sample for quality checks
    min_facts_per_vector: int = 1  # Minimum expected facts per vector
    
    # Alerting thresholds
    coverage_warning_threshold: float = 0.8  # Warn if below 80%
    coverage_critical_threshold: float = 0.5  # Critical if below 50%
    
    def __post_init__(self):
        if self.source_directories is None:
            self.source_directories = {
                "documents": "/home/patrick/echo-brain/documents",
                "code": "/home/patrick/tower-echo-brain",
                "conversations": "/home/patrick/echo-brain/conversations"
            }


class CoverageVerifier:
    """
    Verifies ingestion completeness and fact extraction coverage.
    
    Key functions:
    1. Compare filesystem to database (find untracked files)
    2. Verify all tracked sources have been vectorized
    3. Verify all vectors have facts extracted
    4. Sample quality of extracted facts
    """
    
    def __init__(self, config: VerificationConfig):
        self.config = config
        self._pool: Optional[asyncpg.Pool] = None
    
    async def initialize(self):
        """Initialize database connection."""
        self._pool = await asyncpg.create_pool(
            self.config.postgres_dsn,
            min_size=2,
            max_size=5
        )
    
    async def close(self):
        """Clean up resources."""
        if self._pool:
            await self._pool.close()
    
    async def run_full_verification(self) -> dict:
        """
        Run all verification checks and return comprehensive report.
        """
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "coverage": None,
            "gaps": [],
            "quality_sample": None,
            "recommendations": []
        }
        
        # 1. Get coverage metrics
        report["coverage"] = await self.get_coverage_report()
        
        # 2. Find gaps
        report["gaps"] = await self.find_all_gaps()
        
        # 3. Quality sample
        report["quality_sample"] = await self.sample_fact_quality()
        
        # 4. Generate recommendations
        report["recommendations"] = self._generate_recommendations(report)
        
        return report
    
    async def get_coverage_report(self) -> CoverageReport:
        """
        Generate comprehensive coverage metrics.
        """
        async with self._pool.acquire() as conn:
            # Overall stats
            total_vectors = await conn.fetchval(
                "SELECT COUNT(*) FROM ingestion_tracking WHERE vector_id IS NOT NULL"
            )
            
            vectors_with_facts = await conn.fetchval(
                "SELECT COUNT(*) FROM ingestion_tracking WHERE fact_extracted = TRUE"
            )
            
            # By domain
            domain_stats = await conn.fetch("""
                SELECT 
                    domain,
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE fact_extracted = TRUE) as with_facts
                FROM ingestion_tracking
                WHERE vector_id IS NOT NULL
                GROUP BY domain
            """)
            
            domain_coverage = {}
            for row in domain_stats:
                domain = row["domain"] or "unknown"
                total = row["total"]
                with_facts = row["with_facts"]
                domain_coverage[domain] = {
                    "total": total,
                    "with_facts": with_facts,
                    "pct": (with_facts / total * 100) if total > 0 else 0
                }
            
            # By source type
            source_stats = await conn.fetch("""
                SELECT 
                    source_type,
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE fact_extracted = TRUE) as with_facts
                FROM ingestion_tracking
                WHERE vector_id IS NOT NULL
                GROUP BY source_type
            """)
            
            source_coverage = {}
            for row in source_stats:
                source_type = row["source_type"]
                total = row["total"]
                with_facts = row["with_facts"]
                source_coverage[source_type] = {
                    "total": total,
                    "with_facts": with_facts,
                    "pct": (with_facts / total * 100) if total > 0 else 0
                }
            
            # Gaps
            sources_missing_facts = await conn.fetchval(
                "SELECT COUNT(*) FROM ingestion_tracking WHERE fact_extracted = FALSE AND vector_id IS NOT NULL"
            )
            
            oldest_unprocessed = await conn.fetchval("""
                SELECT MIN(created_at) 
                FROM ingestion_tracking 
                WHERE fact_extracted = FALSE AND vector_id IS NOT NULL
            """)
            
            overall_pct = (vectors_with_facts / total_vectors * 100) if total_vectors > 0 else 0
            
            return CoverageReport(
                total_vectors=total_vectors,
                vectors_with_facts=vectors_with_facts,
                overall_coverage_pct=overall_pct,
                domain_coverage=domain_coverage,
                source_type_coverage=source_coverage,
                sources_missing_facts=sources_missing_facts,
                oldest_unprocessed=oldest_unprocessed
            )
    
    async def find_all_gaps(self) -> list[IngestionGap]:
        """
        Find all gaps in ingestion pipeline.
        """
        gaps = []
        
        # 1. Files not tracked
        untracked = await self._find_untracked_files()
        for path, source_type in untracked:
            gaps.append(IngestionGap(
                source_path=path,
                source_type=source_type,
                reason="not_tracked"
            ))
        
        # 2. Tracked but not vectorized
        async with self._pool.acquire() as conn:
            not_vectorized = await conn.fetch("""
                SELECT source_path, source_type
                FROM ingestion_tracking
                WHERE vector_id IS NULL
                LIMIT 100
            """)
            
            for row in not_vectorized:
                gaps.append(IngestionGap(
                    source_path=row["source_path"],
                    source_type=SourceType(row["source_type"]),
                    reason="not_vectorized"
                ))
            
            # 3. Vectorized but facts not extracted
            no_facts = await conn.fetch("""
                SELECT source_path, source_type
                FROM ingestion_tracking
                WHERE vector_id IS NOT NULL
                AND fact_extracted = FALSE
                LIMIT 100
            """)
            
            for row in no_facts:
                gaps.append(IngestionGap(
                    source_path=row["source_path"],
                    source_type=SourceType(row["source_type"]),
                    reason="facts_not_extracted"
                ))
        
        return gaps
    
    async def _find_untracked_files(self) -> list[tuple[str, SourceType]]:
        """
        Scan source directories and find files not in tracking table.
        """
        untracked = []
        
        # Get all tracked paths
        async with self._pool.acquire() as conn:
            tracked = await conn.fetch("SELECT source_path FROM ingestion_tracking")
            tracked_paths = {row["source_path"] for row in tracked}
        
        # Scan each directory
        for source_type_name, directory in self.config.source_directories.items():
            source_type = SourceType(source_type_name)
            
            if not os.path.exists(directory):
                continue
            
            for root, _, files in os.walk(directory):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    
                    # Skip hidden files and common excludes
                    if filename.startswith('.'):
                        continue
                    if '__pycache__' in filepath or 'node_modules' in filepath:
                        continue
                    if '.git' in filepath:
                        continue
                    
                    # Check if tracked
                    if filepath not in tracked_paths:
                        untracked.append((filepath, source_type))
        
        return untracked
    
    async def sample_fact_quality(self) -> dict:
        """
        Sample vectors and their facts to assess extraction quality.
        """
        async with self._pool.acquire() as conn:
            # Get random sample of vectors with facts
            samples = await conn.fetch("""
                SELECT 
                    it.id,
                    it.source_path,
                    it.source_type,
                    it.facts_count,
                    vc.content
                FROM ingestion_tracking it
                JOIN vector_content vc ON vc.tracking_id = it.id
                WHERE it.fact_extracted = TRUE
                ORDER BY RANDOM()
                LIMIT $1
            """, self.config.sample_size)
            
            quality_metrics = {
                "samples_checked": len(samples),
                "avg_facts_per_vector": 0,
                "vectors_with_zero_facts": 0,
                "vectors_below_minimum": 0,
                "fact_density_by_type": {},
                "sample_details": []
            }
            
            total_facts = 0
            
            for sample in samples:
                facts_count = sample["facts_count"] or 0
                total_facts += facts_count
                
                if facts_count == 0:
                    quality_metrics["vectors_with_zero_facts"] += 1
                
                if facts_count < self.config.min_facts_per_vector:
                    quality_metrics["vectors_below_minimum"] += 1
                
                # Get facts for this vector
                facts = await conn.fetch("""
                    SELECT fact_type, COUNT(*) as count
                    FROM facts
                    WHERE source_id = $1
                    GROUP BY fact_type
                """, sample["id"])
                
                for fact in facts:
                    fact_type = fact["fact_type"]
                    if fact_type not in quality_metrics["fact_density_by_type"]:
                        quality_metrics["fact_density_by_type"][fact_type] = 0
                    quality_metrics["fact_density_by_type"][fact_type] += fact["count"]
                
                # Store sample detail for review
                quality_metrics["sample_details"].append({
                    "source_path": sample["source_path"],
                    "source_type": sample["source_type"],
                    "facts_count": facts_count,
                    "content_preview": sample["content"][:200] + "..."
                })
            
            if samples:
                quality_metrics["avg_facts_per_vector"] = total_facts / len(samples)
            
            return quality_metrics
    
    def _generate_recommendations(self, report: dict) -> list[str]:
        """
        Generate actionable recommendations based on verification results.
        """
        recommendations = []
        coverage = report["coverage"]
        
        # Coverage recommendations
        if coverage.overall_coverage_pct < self.config.coverage_critical_threshold * 100:
            recommendations.append(
                f"CRITICAL: Fact extraction coverage is {coverage.overall_coverage_pct:.1f}%. "
                f"Run scripts/extract_all_facts.py immediately."
            )
        elif coverage.overall_coverage_pct < self.config.coverage_warning_threshold * 100:
            recommendations.append(
                f"WARNING: Fact extraction coverage is {coverage.overall_coverage_pct:.1f}%. "
                f"Schedule fact extraction job."
            )
        
        # Gap recommendations
        gaps_by_reason = {}
        for gap in report["gaps"]:
            if gap.reason not in gaps_by_reason:
                gaps_by_reason[gap.reason] = 0
            gaps_by_reason[gap.reason] += 1
        
        if gaps_by_reason.get("not_tracked", 0) > 0:
            recommendations.append(
                f"Found {gaps_by_reason['not_tracked']} untracked files. "
                f"Run scripts/backfill_tracking.py to add them."
            )
        
        if gaps_by_reason.get("not_vectorized", 0) > 0:
            recommendations.append(
                f"Found {gaps_by_reason['not_vectorized']} tracked files without vectors. "
                f"Check embedding pipeline."
            )
        
        # Quality recommendations
        quality = report["quality_sample"]
        if quality["avg_facts_per_vector"] < self.config.min_facts_per_vector:
            recommendations.append(
                f"Average facts per vector ({quality['avg_facts_per_vector']:.1f}) is below minimum. "
                f"Consider rerunning extraction with improved prompts."
            )
        
        if quality["vectors_with_zero_facts"] > quality["samples_checked"] * 0.1:
            recommendations.append(
                f"{quality['vectors_with_zero_facts']} vectors have zero facts. "
                f"Review extraction failures."
            )
        
        return recommendations


class IntegrationTester:
    """
    End-to-end integration tests for the context assembly pipeline.
    """
    
    def __init__(self, assembler):
        """
        Args:
            assembler: ContextAssembler instance
        """
        self.assembler = assembler
    
    async def run_all_tests(self) -> dict:
        """Run all integration tests."""
        results = {
            "passed": 0,
            "failed": 0,
            "tests": []
        }
        
        test_cases = [
            # Domain isolation tests
            {
                "name": "technical_domain_isolation",
                "query": "How do I fix the PostgreSQL connection pooling issue?",
                "expected_domain": Domain.TECHNICAL,
                "should_contain": ["database", "connection", "postgres"],
                "should_not_contain": ["anime", "lora", "tokyo debt"]
            },
            {
                "name": "anime_domain_isolation",
                "query": "What LoRA settings work best for character consistency in Tokyo Debt Desire?",
                "expected_domain": Domain.ANIME,
                "should_contain": ["lora", "character"],
                "should_not_contain": ["postgresql", "fastapi", "docker"]
            },
            {
                "name": "personal_domain_isolation",
                "query": "What are the Victron MultiPlus settings for my RV?",
                "expected_domain": Domain.PERSONAL,
                "should_contain": ["victron", "rv"],
                "should_not_contain": []
            },
            
            # Context assembly tests
            {
                "name": "context_has_facts",
                "query": "What models does Echo Brain use for inference?",
                "expected_domain": Domain.TECHNICAL,
                "check_facts_present": True,
                "min_facts": 1
            },
            {
                "name": "context_has_code",
                "query": "Show me the retriever implementation",
                "expected_domain": Domain.TECHNICAL,
                "check_code_present": True
            },
            
            # Token budget tests
            {
                "name": "token_budget_respected",
                "query": "Give me a comprehensive overview of the entire Echo Brain architecture",
                "max_tokens": 8192,
                "check_token_budget": True
            }
        ]
        
        for test_case in test_cases:
            result = await self._run_test(test_case)
            results["tests"].append(result)
            if result["passed"]:
                results["passed"] += 1
            else:
                results["failed"] += 1
        
        return results
    
    async def _run_test(self, test_case: dict) -> dict:
        """Run a single test case."""
        result = {
            "name": test_case["name"],
            "passed": True,
            "errors": []
        }
        
        try:
            context = await self.assembler.assemble(test_case["query"])
            
            # Check domain
            if "expected_domain" in test_case:
                if context.domain != test_case["expected_domain"]:
                    result["passed"] = False
                    result["errors"].append(
                        f"Expected domain {test_case['expected_domain'].value}, "
                        f"got {context.domain.value}"
                    )
            
            # Check content contains expected terms
            if "should_contain" in test_case:
                context_text = context.to_prompt_components()["context"].lower()
                for term in test_case["should_contain"]:
                    if term.lower() not in context_text:
                        result["passed"] = False
                        result["errors"].append(f"Missing expected term: {term}")
            
            # Check content doesn't contain forbidden terms
            if "should_not_contain" in test_case:
                context_text = context.to_prompt_components()["context"].lower()
                for term in test_case["should_not_contain"]:
                    if term.lower() in context_text:
                        result["passed"] = False
                        result["errors"].append(f"Contains forbidden term: {term}")
            
            # Check facts present
            if test_case.get("check_facts_present"):
                if len(context.facts) < test_case.get("min_facts", 1):
                    result["passed"] = False
                    result["errors"].append(
                        f"Expected at least {test_case.get('min_facts', 1)} facts, "
                        f"got {len(context.facts)}"
                    )
            
            # Check code present
            if test_case.get("check_code_present"):
                if not context.code_context:
                    result["passed"] = False
                    result["errors"].append("Expected code context but none found")
            
            # Check token budget
            if test_case.get("check_token_budget"):
                if context.token_count > test_case.get("max_tokens", 8192):
                    result["passed"] = False
                    result["errors"].append(
                        f"Token count {context.token_count} exceeds budget "
                        f"{test_case.get('max_tokens', 8192)}"
                    )
        
        except Exception as e:
            result["passed"] = False
            result["errors"].append(f"Exception: {str(e)}")
        
        return result


# ============================================================================
# CLI scripts
# ============================================================================

async def run_verification(postgres_dsn: str):
    """Run full verification and print report."""
    config = VerificationConfig(postgres_dsn=postgres_dsn)
    verifier = CoverageVerifier(config)
    
    await verifier.initialize()
    
    try:
        report = await verifier.run_full_verification()
        
        print("\n" + "=" * 60)
        print("ECHO BRAIN VERIFICATION REPORT")
        print("=" * 60)
        print(f"Generated: {report['timestamp']}")
        
        coverage = report["coverage"]
        print(f"\nCOVERAGE: {coverage.overall_coverage_pct:.1f}%")
        print(f"  Total vectors: {coverage.total_vectors}")
        print(f"  With facts: {coverage.vectors_with_facts}")
        print(f"  Missing facts: {coverage.sources_missing_facts}")
        
        print("\nBY DOMAIN:")
        for domain, stats in coverage.domain_coverage.items():
            print(f"  {domain}: {stats['with_facts']}/{stats['total']} ({stats['pct']:.1f}%)")
        
        print("\nBY SOURCE TYPE:")
        for source_type, stats in coverage.source_type_coverage.items():
            print(f"  {source_type}: {stats['with_facts']}/{stats['total']} ({stats['pct']:.1f}%)")
        
        print(f"\nGAPS FOUND: {len(report['gaps'])}")
        gaps_by_reason = {}
        for gap in report["gaps"]:
            if gap.reason not in gaps_by_reason:
                gaps_by_reason[gap.reason] = 0
            gaps_by_reason[gap.reason] += 1
        for reason, count in gaps_by_reason.items():
            print(f"  {reason}: {count}")
        
        quality = report["quality_sample"]
        print(f"\nQUALITY SAMPLE ({quality['samples_checked']} vectors):")
        print(f"  Avg facts per vector: {quality['avg_facts_per_vector']:.1f}")
        print(f"  Vectors with zero facts: {quality['vectors_with_zero_facts']}")
        
        print("\nRECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")
        
    finally:
        await verifier.close()


if __name__ == "__main__":
    import sys
    dsn = sys.argv[1] if len(sys.argv) > 1 else "postgresql://localhost/echo_brain"
    asyncio.run(run_verification(dsn))

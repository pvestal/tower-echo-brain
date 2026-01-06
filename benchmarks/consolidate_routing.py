#!/usr/bin/env python3
"""
Model Routing Consolidation Script
==================================
Uses benchmark results to create a single source of truth for model routing.

This script:
1. Analyzes current routing chaos (46+ files with model refs)
2. Creates a unified routing configuration based on benchmark data
3. Generates migration script to update database routing tables
4. Provides recommendations for removing hardcoded references

Based on our diagnostic findings:
- 46 files contain conflicting model routing logic
- 8 database tables for routing/models
- No single source of truth
"""

import json
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import tempfile

@dataclass
class BenchmarkSummary:
    """Summary of benchmark results for routing decisions."""
    model: str
    category: str
    avg_ttft_ms: float
    avg_total_ms: float
    avg_tps: float
    accuracy: Optional[float]
    vram_mb: Optional[float]

    def speed_score(self) -> float:
        """Calculate speed score (lower TTFT is better)."""
        return max(0, (1000 - self.avg_ttft_ms) / 1000)

    def composite_score(self) -> float:
        """Calculate composite score for ranking."""
        if self.category == 'classification':
            # For classification, speed and accuracy matter most
            speed = self.speed_score()
            acc = self.accuracy if self.accuracy else 0.5
            return speed * 0.6 + acc * 0.4
        else:
            # For other categories, balance speed and capability
            speed = self.speed_score()
            capability = min(self.avg_tps / 15.0, 1.0)  # Normalize TPS
            return speed * 0.4 + capability * 0.6

class RoutingConsolidator:
    """Consolidates Echo Brain's model routing based on benchmark data."""

    def __init__(self, benchmark_file: str = "/opt/tower-echo-brain/benchmarks/benchmark_results_latest.json"):
        self.benchmark_file = Path(benchmark_file)
        self.results = self.load_benchmark_results()
        self.recommendations = self.analyze_results()

    def load_benchmark_results(self) -> List[dict]:
        """Load benchmark results from JSON file."""
        if not self.benchmark_file.exists():
            raise FileNotFoundError(f"Benchmark results not found: {self.benchmark_file}")

        with open(self.benchmark_file) as f:
            return json.load(f)

    def analyze_results(self) -> Dict[str, BenchmarkSummary]:
        """Analyze benchmark results and create recommendations."""
        # Group results by model and category
        model_category_stats = {}

        for result in self.results:
            if result.get('error'):
                continue

            key = (result['model'], result['category'])
            if key not in model_category_stats:
                model_category_stats[key] = []
            model_category_stats[key].append(result)

        # Calculate averages and scores
        summaries = {}
        for (model, category), results in model_category_stats.items():
            ttft_times = [r['time_to_first_token_ms'] for r in results]
            total_times = [r['total_generation_time_ms'] for r in results]
            tps_values = [r['tokens_per_second'] for r in results]
            vram_values = [r.get('vram_during_mb') for r in results if r.get('vram_during_mb')]

            # Calculate accuracy for classification
            accuracy = None
            if category == 'classification':
                correct_results = [r for r in results if r.get('classification_correct') is not None]
                if correct_results:
                    accuracy = sum(1 for r in correct_results if r['classification_correct']) / len(correct_results)

            summary = BenchmarkSummary(
                model=model,
                category=category,
                avg_ttft_ms=sum(ttft_times) / len(ttft_times),
                avg_total_ms=sum(total_times) / len(total_times),
                avg_tps=sum(tps_values) / len(tps_values),
                accuracy=accuracy,
                vram_mb=sum(vram_values) / len(vram_values) if vram_values else None
            )

            summaries[f"{model}:{category}"] = summary

        return summaries

    def get_best_model_per_category(self) -> Dict[str, BenchmarkSummary]:
        """Get the best model for each category based on composite scores."""
        best_per_category = {}

        for key, summary in self.recommendations.items():
            category = summary.category
            if category not in best_per_category:
                best_per_category[category] = summary
            elif summary.composite_score() > best_per_category[category].composite_score():
                best_per_category[category] = summary

        return best_per_category

    def generate_routing_config(self) -> Dict:
        """Generate unified routing configuration."""
        best_models = self.get_best_model_per_category()

        config = {
            "routing_strategy": "benchmark_optimized",
            "fallback_model": "llama3.2:3b",  # Smallest reliable model
            "categories": {},
            "hardware_limits": {
                "max_vram_mb": 12000,  # RTX 3060 limit
                "concurrent_models": 1
            },
            "performance_thresholds": {
                "classification_max_ttft_ms": 150,
                "coding_min_tps": 5.0,
                "reasoning_min_tps": 3.0
            }
        }

        # Add category-specific routing
        for category, summary in best_models.items():
            config["categories"][category] = {
                "primary_model": summary.model,
                "expected_ttft_ms": round(summary.avg_ttft_ms),
                "expected_tps": round(summary.avg_tps, 1),
                "accuracy": summary.accuracy,
                "vram_requirement_mb": summary.vram_mb,
                "selection_reason": f"Best composite score: {summary.composite_score():.3f}",
                "benchmark_date": "2026-01-06"
            }

        return config

    def generate_database_migration(self) -> str:
        """Generate SQL migration script to update database routing."""
        best_models = self.get_best_model_per_category()

        sql_lines = [
            "-- Echo Brain Model Routing Consolidation",
            "-- Generated from benchmark results: 2026-01-06",
            "-- Replaces hardcoded routing with data-driven decisions",
            "",
            "BEGIN;",
            "",
            "-- Clean up existing routing chaos",
            "TRUNCATE TABLE intent_model_mapping CASCADE;",
            "TRUNCATE TABLE model_routing CASCADE;",
            "",
            "-- Insert benchmark-optimized routing",
        ]

        for category, summary in best_models.items():
            sql_lines.extend([
                f"INSERT INTO intent_model_mapping (intent, recommended_model, confidence, reasoning)",
                f"VALUES ('{category}', '{summary.model}', {summary.composite_score():.3f}, ",
                f"        'Benchmark winner: {summary.avg_ttft_ms:.0f}ms TTFT, {summary.avg_tps:.1f} TPS');",
                ""
            ])

        sql_lines.extend([
            "-- Update model capabilities table",
            "INSERT INTO model_capabilities (model_name, category, ttft_ms, tps, vram_mb, last_benchmarked)",
            "VALUES"
        ])

        capability_rows = []
        for key, summary in self.recommendations.items():
            capability_rows.append(
                f"  ('{summary.model}', '{summary.category}', {summary.avg_ttft_ms:.0f}, "
                f"{summary.avg_tps:.1f}, {summary.vram_mb or 'NULL'}, NOW())"
            )

        sql_lines.append(",\n".join(capability_rows) + ";")

        sql_lines.extend([
            "",
            "COMMIT;",
            "",
            "-- Verification queries:",
            "SELECT intent, recommended_model, confidence FROM intent_model_mapping;",
            "SELECT model_name, category, ttft_ms, tps FROM model_capabilities ORDER BY category, ttft_ms;",
        ])

        return "\n".join(sql_lines)

    def find_hardcoded_references(self) -> List[str]:
        """Find files with hardcoded model references."""
        try:
            # Search for files with model references
            result = subprocess.run([
                "find", "/opt/tower-echo-brain/src", "-name", "*.py",
                "-exec", "grep", "-l",
                "-E", "(qwen2\.5|deepseek|llama3|gemma2|minicpm|mistral).*:",
                "{}", "+"
            ], capture_output=True, text=True)

            return result.stdout.strip().split('\n') if result.stdout.strip() else []
        except Exception as e:
            print(f"Error finding hardcoded references: {e}")
            return []

    def generate_cleanup_recommendations(self) -> List[str]:
        """Generate recommendations for cleaning up hardcoded references."""
        files_with_refs = self.find_hardcoded_references()
        recommendations = [
            "🧹 CLEANUP RECOMMENDATIONS",
            "=" * 50,
            "",
            f"Found {len(files_with_refs)} files with hardcoded model references.",
            "",
            "Priority cleanup order:",
            "",
            "1. HIGH PRIORITY - Core routing files:",
            "   - src/core/db_model_router.py",
            "   - src/model_router.py",
            "   - src/core/intelligence.py",
            "",
            "2. MEDIUM PRIORITY - API endpoints:",
            "   - src/api/echo.py",
            "   - src/api/echo_refactored.py",
            "",
            "3. LOW PRIORITY - Individual modules:",
        ]

        # Add individual files
        for file_path in sorted(files_with_refs):
            if any(priority in file_path for priority in ['router', 'intelligence', 'api/echo']):
                continue  # Already listed above
            recommendations.append(f"   - {file_path}")

        recommendations.extend([
            "",
            "CLEANUP STRATEGY:",
            "",
            "1. Replace hardcoded model names with database lookup:",
            "   FROM: model = 'qwen2.5:3b'",
            "   TO:   model = get_model_for_category('classification')",
            "",
            "2. Use the unified db_model_router.py as single source:",
            "   FROM: Multiple routing systems",
            "   TO:   All imports from src.core.db_model_router",
            "",
            "3. Update imports:",
            "   FROM: from src.core.complexity_analyzer import TIER_TO_MODEL",
            "   TO:   from src.core.db_model_router import get_model_for_intent",
        ])

        return recommendations

    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report."""
        best_models = self.get_best_model_per_category()

        lines = [
            "🧠 ECHO BRAIN MODEL ROUTING CONSOLIDATION REPORT",
            "=" * 60,
            f"Generated: 2026-01-06",
            f"Based on benchmark results: {len(self.results)} test runs",
            "",
            "📊 BENCHMARK-OPTIMIZED RECOMMENDATIONS",
            "=" * 40,
        ]

        for category, summary in best_models.items():
            lines.extend([
                f"\n{category.upper()}:",
                f"  🏆 Winner: {summary.model}",
                f"  ⚡ TTFT: {summary.avg_ttft_ms:.0f}ms",
                f"  🚀 TPS: {summary.avg_tps:.1f}",
                f"  🎯 Accuracy: {summary.accuracy:.1%}" if summary.accuracy else "  🎯 Accuracy: N/A",
                f"  💾 VRAM: {summary.vram_mb:.0f}MB" if summary.vram_mb else "  💾 VRAM: Unknown",
                f"  📈 Score: {summary.composite_score():.3f}",
            ])

        lines.extend([
            "",
            "🔧 IMPLEMENTATION STEPS",
            "=" * 30,
            "",
            "1. Apply database migration:",
            "   psql -f routing_migration.sql",
            "",
            "2. Update db_model_router.py to use new config",
            "",
            "3. Clean up hardcoded references (58 files found)",
            "",
            "4. Test with sample queries",
            "",
            "5. Monitor performance in production",
            "",
            "🎯 EXPECTED IMPROVEMENTS",
            "=" * 30,
            "",
            "- Eliminate conflicts between 46 routing files",
            "- Reduce classification time to <150ms",
            "- Increase coding throughput to >5 TPS",
            "- Remove anime bias from fallback context",
            "- Enable data-driven model selection",
        ])

        return "\n".join(lines)

    def save_all_outputs(self, output_dir: str = "/opt/tower-echo-brain/benchmarks"):
        """Save all generated files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save routing configuration
        config = self.generate_routing_config()
        with open(output_path / "routing_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save database migration
        migration_sql = self.generate_database_migration()
        with open(output_path / "routing_migration.sql", "w") as f:
            f.write(migration_sql)

        # Save cleanup recommendations
        cleanup = self.generate_cleanup_recommendations()
        with open(output_path / "cleanup_recommendations.txt", "w") as f:
            f.write("\n".join(cleanup))

        # Save summary report
        summary = self.generate_summary_report()
        with open(output_path / "consolidation_report.txt", "w") as f:
            f.write(summary)

        print(f"✅ All consolidation files saved to {output_path}")
        return output_path

def main():
    """Main execution."""
    print("🧠 Echo Brain Model Routing Consolidation")
    print("=" * 50)

    try:
        consolidator = RoutingConsolidator()

        # Generate and save all outputs
        output_dir = consolidator.save_all_outputs()

        # Print summary
        print(consolidator.generate_summary_report())

        print(f"\n📁 Files generated:")
        for file_path in output_dir.glob("*"):
            print(f"   - {file_path.name}")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nRun the benchmark first:")
        print("   cd /opt/tower-echo-brain/benchmarks")
        print("   python3 benchmark.py --models qwen2.5:3b qwen2.5-coder:7b")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
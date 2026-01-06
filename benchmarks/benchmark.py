#!/usr/bin/env python3
"""
Echo Brain Model Benchmarking Suite
====================================
Systematically evaluates LLM models for Echo Brain's specific use cases.

Use Cases Tested:
1. Classification - Fast intent routing (speed critical)
2. Coding - Generation, debugging, refactoring (quality critical)
3. Reasoning - Multi-step problem solving (chain quality)
4. General - Knowledge and conversation (balance)

Metrics Collected:
- Latency (time to first token, total generation time)
- VRAM usage during inference
- Output quality (automated + manual scoring)
- Tokens per second
- Context handling capability

Author: Patrick's Echo Brain Project
"""

import asyncio
import json
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import statistics
import csv

import httpx


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Central configuration for benchmark runs."""

    # Ollama endpoint
    ollama_base_url: str = "http://localhost:11434"

    # Models to benchmark (grouped by intended use)
    classification_models: list = field(default_factory=lambda: [
        "qwen2.5:1.5b",
        "qwen2.5:3b",
        "llama3.2:3b",
        "phi3:mini",
    ])

    coding_models: list = field(default_factory=lambda: [
        "qwen2.5-coder:7b",
        "qwen2.5-coder:14b",
        "deepseek-coder-v2:16b",
        "codellama:13b",
    ])

    reasoning_models: list = field(default_factory=lambda: [
        "deepseek-r1:8b",
        "qwen2.5:14b",
        "llama3.1:8b",
        "phi3:14b",
    ])

    general_models: list = field(default_factory=lambda: [
        "llama3.1:8b",
        "qwen2.5:7b",
        "mistral:7b",
    ])

    # Benchmark settings
    warmup_runs: int = 1          # Warm up model before timing
    timed_runs: int = 3           # Number of timed runs per test
    timeout_seconds: int = 120    # Max time per generation

    # Output directory
    output_dir: str = "/opt/tower-echo-brain/benchmarks"


# =============================================================================
# TEST CASES - Based on Echo Brain's Actual Query Patterns
# =============================================================================

CLASSIFICATION_TESTS = [
    {
        "id": "cls_001",
        "name": "Simple greeting",
        "prompt": "Classify this user intent: 'Hey, how are you doing today?'\nCategories: greeting, question, coding, reasoning, anime, financial\nRespond with only the category name.",
        "expected": "greeting",
        "max_tokens": 10,
    },
    {
        "id": "cls_002",
        "name": "Coding request",
        "prompt": "Classify this user intent: 'Can you fix the bug in my Python function that handles database connections?'\nCategories: greeting, question, coding, reasoning, anime, financial\nRespond with only the category name.",
        "expected": "coding",
        "max_tokens": 10,
    },
    {
        "id": "cls_003",
        "name": "Reasoning request",
        "prompt": "Classify this user intent: 'Help me think through the architecture for a distributed task queue system'\nCategories: greeting, question, coding, reasoning, anime, financial\nRespond with only the category name.",
        "expected": "reasoning",
        "max_tokens": 10,
    },
    {
        "id": "cls_004",
        "name": "Financial query",
        "prompt": "Classify this user intent: 'Show me my credit card transactions from last week'\nCategories: greeting, question, coding, reasoning, anime, financial\nRespond with only the category name.",
        "expected": "financial",
        "max_tokens": 10,
    },
    {
        "id": "cls_005",
        "name": "Ambiguous technical",
        "prompt": "Classify this user intent: 'Why is the model routing not working correctly?'\nCategories: greeting, question, coding, reasoning, anime, financial\nRespond with only the category name.",
        "expected": "coding",  # Could be reasoning, tests ambiguity handling
        "max_tokens": 10,
    },
]

CODING_TESTS = [
    {
        "id": "code_001",
        "name": "Simple function generation",
        "prompt": """Write a Python function that reverses a string without using built-in reverse methods.
Include type hints and a docstring.""",
        "max_tokens": 300,
        "quality_criteria": ["has_type_hints", "has_docstring", "correct_logic", "no_builtin_reverse"],
    },
    {
        "id": "code_002",
        "name": "Bug fix identification",
        "prompt": """Find and fix the bug in this Python code:

```python
def get_user_by_id(user_id: int, users: list[dict]) -> dict | None:
    for user in users:
        if user['id'] == user_id:
            return user
        return None
```

Explain the bug and provide the corrected code.""",
        "max_tokens": 400,
        "quality_criteria": ["identifies_indentation_bug", "correct_fix", "clear_explanation"],
    },
    {
        "id": "code_003",
        "name": "Database query optimization",
        "prompt": """Refactor this SQLAlchemy query for better performance:

```python
def get_active_users_with_orders(session):
    users = session.query(User).all()
    result = []
    for user in users:
        if user.is_active:
            orders = session.query(Order).filter(Order.user_id == user.id).all()
            if len(orders) > 0:
                result.append({"user": user, "order_count": len(orders)})
    return result
```

Provide the optimized version with explanation.""",
        "max_tokens": 600,
        "quality_criteria": ["uses_join", "single_query", "avoids_n_plus_1", "clear_explanation"],
    },
    {
        "id": "code_004",
        "name": "Async pattern implementation",
        "prompt": """Write a Python async function that:
1. Fetches data from multiple URLs concurrently
2. Has a configurable timeout per request
3. Returns results as they complete (not waiting for all)
4. Handles individual request failures gracefully

Use aiohttp and asyncio.""",
        "max_tokens": 800,
        "quality_criteria": ["uses_gather_or_taskgroup", "has_timeout", "error_handling", "streaming_results"],
    },
    {
        "id": "code_005",
        "name": "Echo Brain specific - Model router",
        "prompt": """Write a Python class for routing LLM queries to appropriate models based on:
- Query complexity (simple/medium/complex)
- Query type (coding/reasoning/general)
- Available model capabilities from a database table

The router should:
1. Score query complexity using heuristics
2. Match to best available model
3. Fall back gracefully if preferred model unavailable
4. Log routing decisions for analysis

Use dataclasses and type hints.""",
        "max_tokens": 1200,
        "quality_criteria": ["has_complexity_scoring", "database_integration", "fallback_logic", "logging", "type_hints"],
    },
]

REASONING_TESTS = [
    {
        "id": "reason_001",
        "name": "Architecture decision",
        "prompt": """I need to decide between these approaches for Echo Brain's model routing:

Option A: All routing logic in Python code with a configuration dict
Option B: Database-driven routing with PostgreSQL tables
Option C: Hybrid - database for model configs, Python for routing logic

Consider: maintainability, performance, flexibility, debugging ease.

Think through each option step by step, then recommend one with justification.""",
        "max_tokens": 1000,
        "quality_criteria": ["considers_all_options", "structured_analysis", "clear_recommendation", "reasoning_visible"],
    },
    {
        "id": "reason_002",
        "name": "Debugging strategy",
        "prompt": """Echo Brain is returning anime-related responses to unrelated queries.

Known facts:
- Database separation (tower_anime vs tower_consolidated) was completed
- resilient_context.py has anime fallback at line 146
- Memory middleware searches all collections without filtering
- Query classification sometimes misroutes

Develop a systematic debugging plan. What do you check first, second, third? Why that order?""",
        "max_tokens": 800,
        "quality_criteria": ["systematic_approach", "prioritized_steps", "explains_reasoning", "actionable"],
    },
    {
        "id": "reason_003",
        "name": "Trade-off analysis",
        "prompt": """For a self-hosted AI assistant running on consumer hardware (12GB VRAM):

Trade-off: Using a 7B model (fast, fits in VRAM) vs 14B model (better quality, requires quantization or offloading)

The assistant handles: coding help, reasoning, general questions, task automation.

Analyze this trade-off. Consider real-world usage patterns, not just benchmarks.""",
        "max_tokens": 800,
        "quality_criteria": ["quantifies_tradeoffs", "considers_usage_patterns", "practical_recommendation"],
    },
    {
        "id": "reason_004",
        "name": "Multi-step problem decomposition",
        "prompt": """Problem: Echo Brain's autonomous workers exist but aren't coordinated.

Components available:
- Celery task queue with Redis
- DeepSeek reasoning model
- DeepSeek coding model
- Codebase search API
- Git operations module

Goal: When a user asks "improve the error handling in the model router", Echo Brain should:
1. Understand the request
2. Find relevant code
3. Reason about improvements
4. Generate code changes
5. Validate changes
6. Commit if approved

Design the coordination flow. What calls what, in what order, with what data?""",
        "max_tokens": 1200,
        "quality_criteria": ["complete_flow", "clear_data_flow", "handles_failures", "practical"],
    },
]

GENERAL_TESTS = [
    {
        "id": "gen_001",
        "name": "Knowledge recall",
        "prompt": "Explain the difference between PostgreSQL's JSONB and JSON column types. When would you use each?",
        "max_tokens": 400,
        "quality_criteria": ["accurate", "practical_guidance", "concise"],
    },
    {
        "id": "gen_002",
        "name": "Conversational context",
        "prompt": """Previous context: User is building a personal AI assistant called Echo Brain that runs on Ollama with PostgreSQL and Qdrant.

User message: "What should I focus on next?"

Provide a helpful response that acknowledges context and asks clarifying questions.""",
        "max_tokens": 300,
        "quality_criteria": ["uses_context", "asks_clarification", "helpful_tone"],
    },
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model: str
    test_id: str
    test_name: str
    category: str

    # Timing metrics
    time_to_first_token_ms: float
    total_generation_time_ms: float
    tokens_generated: int
    tokens_per_second: float

    # Resource metrics
    vram_before_mb: Optional[float] = None
    vram_during_mb: Optional[float] = None

    # Quality metrics (to be filled manually or by automated checks)
    output: str = ""
    classification_correct: Optional[bool] = None
    quality_score: Optional[int] = None  # 1-5 manual rating
    quality_notes: str = ""

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None


# =============================================================================
# BENCHMARK ENGINE
# =============================================================================

class ModelBenchmark:
    """
    Executes benchmarks against Ollama models and collects metrics.

    Design Philosophy:
    - Each test runs multiple times for statistical validity
    - VRAM monitoring via nvidia-smi subprocess
    - Results saved incrementally (don't lose data on crash)
    - Supports both automated and manual quality assessment
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: list[BenchmarkResult] = []
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def check_model_available(self, model: str) -> bool:
        """Verify model is pulled and available in Ollama."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.config.ollama_base_url}/api/tags",
                    timeout=10.0
                )
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    available = [m["name"] for m in models]
                    # Check for exact match or match without tag
                    return any(
                        model == m or model == m.split(":")[0]
                        for m in available
                    )
        except Exception as e:
            print(f"  ⚠️  Error checking model availability: {e}")
        return False

    def get_vram_usage_mb(self) -> Optional[float]:
        """Get current VRAM usage via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Sum all GPUs (in case of multi-GPU)
                values = [float(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
                return sum(values)
        except Exception:
            pass
        return None

    async def run_inference(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        system_prompt: Optional[str] = None
    ) -> tuple[str, float, float, int]:
        """
        Run inference and return (output, ttft_ms, total_ms, token_count).

        Uses streaming to measure time-to-first-token accurately.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
            }
        }

        output_chunks = []
        first_token_time = None
        start_time = time.perf_counter()

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.config.ollama_base_url}/api/chat",
                    json=request_body,
                    timeout=self.config.timeout_seconds
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    content = data["message"]["content"]
                                    if content and first_token_time is None:
                                        first_token_time = time.perf_counter()
                                    output_chunks.append(content)
                            except json.JSONDecodeError:
                                continue
        except httpx.TimeoutException:
            return "", 0, self.config.timeout_seconds * 1000, 0
        except Exception as e:
            return f"ERROR: {e}", 0, 0, 0

        end_time = time.perf_counter()

        output = "".join(output_chunks)
        total_ms = (end_time - start_time) * 1000
        ttft_ms = ((first_token_time - start_time) * 1000) if first_token_time else total_ms

        # Rough token count (Ollama doesn't always return this in streaming)
        token_count = len(output.split()) * 1.3  # Rough approximation

        return output, ttft_ms, total_ms, int(token_count)

    async def benchmark_single_test(
        self,
        model: str,
        test: dict,
        category: str
    ) -> BenchmarkResult:
        """Run a single test against a model, with warmup and multiple runs."""

        print(f"    Testing: {test['name']}")

        # Warmup run (not timed)
        for _ in range(self.config.warmup_runs):
            await self.run_inference(model, test["prompt"], test["max_tokens"])

        # Timed runs
        ttft_times = []
        total_times = []
        token_counts = []
        final_output = ""
        vram_during = None

        for run in range(self.config.timed_runs):
            vram_before = self.get_vram_usage_mb()

            output, ttft, total, tokens = await self.run_inference(
                model,
                test["prompt"],
                test["max_tokens"]
            )

            vram_after = self.get_vram_usage_mb()
            if vram_after and (vram_during is None or vram_after > vram_during):
                vram_during = vram_after

            ttft_times.append(ttft)
            total_times.append(total)
            token_counts.append(tokens)
            final_output = output  # Keep last output for quality review

        # Calculate statistics
        avg_ttft = statistics.mean(ttft_times)
        avg_total = statistics.mean(total_times)
        avg_tokens = statistics.mean(token_counts)
        tps = (avg_tokens / (avg_total / 1000)) if avg_total > 0 else 0

        # Check classification correctness if applicable
        classification_correct = None
        if category == "classification" and "expected" in test:
            classification_correct = test["expected"].lower() in final_output.lower()

        result = BenchmarkResult(
            model=model,
            test_id=test["id"],
            test_name=test["name"],
            category=category,
            time_to_first_token_ms=round(avg_ttft, 2),
            total_generation_time_ms=round(avg_total, 2),
            tokens_generated=int(avg_tokens),
            tokens_per_second=round(tps, 2),
            vram_during_mb=vram_during,
            output=final_output,
            classification_correct=classification_correct,
        )

        # Print quick summary
        correct_str = ""
        if classification_correct is not None:
            correct_str = " ✅" if classification_correct else " ❌"
        print(f"      TTFT: {avg_ttft:.0f}ms | Total: {avg_total:.0f}ms | TPS: {tps:.1f}{correct_str}")

        return result

    async def benchmark_model_category(
        self,
        models: list[str],
        tests: list[dict],
        category: str
    ):
        """Benchmark all models in a category against all tests."""

        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {category.upper()}")
        print(f"{'='*60}")

        for model in models:
            print(f"\n  Model: {model}")

            # Check if model is available
            if not await self.check_model_available(model):
                print(f"    ⚠️  Model not available, skipping. Pull with: ollama pull {model}")
                continue

            for test in tests:
                try:
                    result = await self.benchmark_single_test(model, test, category)
                    self.results.append(result)

                    # Save incrementally
                    self.save_results()

                except Exception as e:
                    print(f"    ❌ Error on {test['name']}: {e}")
                    self.results.append(BenchmarkResult(
                        model=model,
                        test_id=test["id"],
                        test_name=test["name"],
                        category=category,
                        time_to_first_token_ms=0,
                        total_generation_time_ms=0,
                        tokens_generated=0,
                        tokens_per_second=0,
                        error=str(e),
                    ))

    def save_results(self):
        """Save results to JSON and CSV files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON (full data)
        json_path = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        # CSV (for spreadsheet analysis)
        csv_path = self.output_dir / f"benchmark_results_{timestamp}.csv"
        if self.results:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
                writer.writeheader()
                for r in self.results:
                    writer.writerow(asdict(r))

        # Also save "latest" symlink-style files
        latest_json = self.output_dir / "benchmark_results_latest.json"
        latest_csv = self.output_dir / "benchmark_results_latest.csv"

        with open(latest_json, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        if self.results:
            with open(latest_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
                writer.writeheader()
                for r in self.results:
                    writer.writerow(asdict(r))

    def generate_report(self) -> str:
        """Generate a summary report of benchmark results."""

        report_lines = [
            "=" * 70,
            "ECHO BRAIN MODEL BENCHMARK REPORT",
            f"Generated: {datetime.now().isoformat()}",
            "=" * 70,
            "",
        ]

        # Group by category
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = []
            categories[r.category].append(r)

        for category, results in categories.items():
            report_lines.append(f"\n{'─'*70}")
            report_lines.append(f"CATEGORY: {category.upper()}")
            report_lines.append(f"{'─'*70}")

            # Group by model within category
            models = {}
            for r in results:
                if r.model not in models:
                    models[r.model] = []
                models[r.model].append(r)

            # Summary table
            report_lines.append(f"\n{'Model':<30} {'Avg TTFT':<12} {'Avg Total':<12} {'Avg TPS':<10} {'Accuracy':<10}")
            report_lines.append("-" * 74)

            for model, model_results in models.items():
                valid_results = [r for r in model_results if r.error is None]
                if not valid_results:
                    report_lines.append(f"{model:<30} {'ERROR':<12}")
                    continue

                avg_ttft = statistics.mean([r.time_to_first_token_ms for r in valid_results])
                avg_total = statistics.mean([r.total_generation_time_ms for r in valid_results])
                avg_tps = statistics.mean([r.tokens_per_second for r in valid_results])

                # Classification accuracy
                accuracy_str = "N/A"
                correct_results = [r for r in valid_results if r.classification_correct is not None]
                if correct_results:
                    correct_count = sum(1 for r in correct_results if r.classification_correct)
                    accuracy_str = f"{correct_count}/{len(correct_results)}"

                report_lines.append(
                    f"{model:<30} {avg_ttft:>8.0f}ms   {avg_total:>8.0f}ms   {avg_tps:>6.1f}     {accuracy_str:<10}"
                )

            # Recommendation
            report_lines.append(f"\n  📊 Recommendation for {category}:")

            if category == "classification":
                # Prioritize speed and accuracy
                best = min(
                    [(m, r) for m, results in models.items() for r in results if r.error is None],
                    key=lambda x: x[1].time_to_first_token_ms,
                    default=None
                )
                if best:
                    report_lines.append(f"     Fastest: {best[0]} ({best[1].time_to_first_token_ms:.0f}ms TTFT)")

            elif category == "coding":
                # Balance speed and capability (TPS as proxy)
                valid_models = [
                    (m, statistics.mean([r.tokens_per_second for r in results if r.error is None]))
                    for m, results in models.items()
                    if any(r.error is None for r in results)
                ]
                if valid_models:
                    best = max(valid_models, key=lambda x: x[1])
                    report_lines.append(f"     Highest throughput: {best[0]} ({best[1]:.1f} TPS)")

            elif category == "reasoning":
                # Note about manual review needed
                report_lines.append("     ⚠️  Reasoning quality requires manual review of outputs")
                report_lines.append("     Check benchmark_results_latest.json for full outputs")

        report_lines.append(f"\n{'='*70}")
        report_lines.append("NEXT STEPS:")
        report_lines.append("1. Review outputs in benchmark_results_latest.json")
        report_lines.append("2. Score quality manually (1-5) for coding/reasoning tests")
        report_lines.append("3. Run quality_scorer.py to update scores and regenerate report")
        report_lines.append("=" * 70)

        report = "\n".join(report_lines)

        # Save report
        report_path = self.output_dir / "benchmark_report.txt"
        with open(report_path, "w") as f:
            f.write(report)

        return report

    async def run_full_benchmark(self):
        """Execute the complete benchmark suite."""

        print("🧠 Echo Brain Model Benchmark Suite")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"Warmup runs: {self.config.warmup_runs}")
        print(f"Timed runs: {self.config.timed_runs}")
        print()

        # Classification benchmarks
        await self.benchmark_model_category(
            self.config.classification_models,
            CLASSIFICATION_TESTS,
            "classification"
        )

        # Coding benchmarks
        await self.benchmark_model_category(
            self.config.coding_models,
            CODING_TESTS,
            "coding"
        )

        # Reasoning benchmarks
        await self.benchmark_model_category(
            self.config.reasoning_models,
            REASONING_TESTS,
            "reasoning"
        )

        # General benchmarks
        await self.benchmark_model_category(
            self.config.general_models,
            GENERAL_TESTS,
            "general"
        )

        # Generate final report
        report = self.generate_report()
        print("\n" + report)

        print(f"\n✅ Benchmark complete! Results saved to: {self.output_dir}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

async def main():
    """Main entry point for benchmark execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Echo Brain Model Benchmark Suite")
    parser.add_argument(
        "--output-dir",
        default="/opt/tower-echo-brain/benchmarks",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of timed runs per test (default: 3)"
    )
    parser.add_argument(
        "--category",
        choices=["all", "classification", "coding", "reasoning", "general"],
        default="all",
        help="Run only specific category (default: all)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Override models to test (space-separated)"
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        output_dir=args.output_dir,
        timed_runs=args.runs,
    )

    # Override models if specified
    if args.models:
        config.classification_models = args.models
        config.coding_models = args.models
        config.reasoning_models = args.models
        config.general_models = args.models

    benchmark = ModelBenchmark(config)

    if args.category == "all":
        await benchmark.run_full_benchmark()
    else:
        # Run single category
        category_map = {
            "classification": (config.classification_models, CLASSIFICATION_TESTS),
            "coding": (config.coding_models, CODING_TESTS),
            "reasoning": (config.reasoning_models, REASONING_TESTS),
            "general": (config.general_models, GENERAL_TESTS),
        }
        models, tests = category_map[args.category]
        await benchmark.benchmark_model_category(models, tests, args.category)
        report = benchmark.generate_report()
        print("\n" + report)


if __name__ == "__main__":
    asyncio.run(main())
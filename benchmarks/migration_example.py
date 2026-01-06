#!/usr/bin/env python3
"""
Migration Example - Replacing Hardcoded Model References
========================================================
Shows how to update Echo Brain modules to use the unified router
instead of hardcoded model names.

Run this to see examples of the changes needed across the 52 files
with hardcoded model references.
"""

def show_migration_examples():
    """Show before/after examples for migrating hardcoded models."""

    examples = [
        {
            "file": "src/api/echo.py",
            "before": '''
# OLD: Hardcoded model selection
model = "qwen2.5:3b"
if complexity > 50:
    model = "deepseek-r1:8b"
''',
            "after": '''
# NEW: Use unified router
from src.core.unified_model_router import select_model_for_query
model = select_model_for_query(query)
'''
        },
        {
            "file": "src/core/complexity_analyzer.py",
            "before": '''
# OLD: TIER_TO_MODEL dictionary
TIER_TO_MODEL = {
    "simple": "llama3.2:3b",
    "medium": "qwen2.5:7b",
    "complex": "deepseek-r1:8b"
}
''',
            "after": '''
# NEW: Use unified router
from src.core.unified_model_router import get_model_for_category

def get_model_for_tier(tier: str) -> str:
    return get_model_for_category(tier)
'''
        },
        {
            "file": "src/agents/coding_agent.py",
            "before": '''
# OLD: Agent-specific model
class CodingAgent:
    def __init__(self):
        self.model = "qwen2.5-coder:7b"
''',
            "after": '''
# NEW: Dynamic model selection
from src.core.unified_model_router import get_model_for_intent

class CodingAgent:
    def __init__(self):
        self.model = get_model_for_intent("coding")
'''
        },
        {
            "file": "src/core/intelligence.py",
            "before": '''
# OLD: model_hierarchy with hardcoded names
model_hierarchy = {
    "fast": ["llama3.2:3b", "qwen2.5:3b"],
    "balanced": ["llama3.1:8b", "qwen2.5:7b"],
    "powerful": ["deepseek-r1:8b", "qwen2.5:14b"]
}
''',
            "after": '''
# NEW: Query-based selection
from src.core.unified_model_router import select_model_for_query

def get_model_for_requirement(query: str, requirement: str = "balanced") -> str:
    """Get model optimized for specific requirement."""
    return select_model_for_query(f"{requirement}: {query}")
'''
        }
    ]

    print("🔄 MIGRATION EXAMPLES")
    print("=" * 50)
    print("How to replace hardcoded model references with unified router:\n")

    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['file']}")
        print("   " + "=" * len(example['file']))
        print("\n   BEFORE:")
        print("   " + "\n   ".join(line.strip() for line in example['before'].strip().split('\n')))
        print("\n   AFTER:")
        print("   " + "\n   ".join(line.strip() for line in example['after'].strip().split('\n')))
        print("\n")

    print("🚀 MIGRATION BENEFITS")
    print("=" * 30)
    print("✅ Single source of truth for model selection")
    print("✅ Benchmark-optimized performance")
    print("✅ Database-driven routing (can update without code changes)")
    print("✅ Automatic fallback if database unavailable")
    print("✅ Consistent model selection across all modules")
    print("✅ Easy to test and validate model choices")

def show_priority_files():
    """Show which files should be migrated first."""

    print("\n📋 MIGRATION PRIORITY")
    print("=" * 30)

    priority_files = [
        ("HIGH", [
            "src/core/db_model_router.py",
            "src/model_router.py",
            "src/core/intelligence.py"
        ]),
        ("MEDIUM", [
            "src/api/echo.py",
            "src/api/echo_refactored.py",
            "src/core/complexity_analyzer.py"
        ]),
        ("LOW", [
            "src/agents/*.py (5 files)",
            "src/misc/*.py (12 files)",
            "Other modules (35 files)"
        ])
    ]

    for priority, files in priority_files:
        print(f"\n{priority} PRIORITY:")
        for file in files:
            print(f"  • {file}")

def show_testing_strategy():
    """Show how to test the migration."""

    print("\n🧪 TESTING STRATEGY")
    print("=" * 30)
    print("""
1. UNIT TESTS:
   Test unified router with known inputs

2. INTEGRATION TESTS:
   Test actual queries through Echo Brain API

3. PERFORMANCE TESTS:
   Verify 94ms TTFT for classification
   Verify no regression in other categories

4. COMPARISON TESTS:
   Old vs new routing for same queries
   Document any differences

5. PRODUCTION MONITORING:
   Track model selection decisions
   Monitor response times
   Watch for routing errors
""")

def main():
    """Main execution."""
    show_migration_examples()
    show_priority_files()
    show_testing_strategy()

    print("\n🎯 NEXT STEPS")
    print("=" * 20)
    print("1. Start with HIGH priority files")
    print("2. Test each migration thoroughly")
    print("3. Update imports gradually")
    print("4. Monitor performance impact")
    print("5. Remove old routing systems once stable")

if __name__ == "__main__":
    main()
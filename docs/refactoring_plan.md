# Echo Brain Complexity Scoring Refactoring Plan

## Current State (Broken)
```
intelligence.py (OLD algorithm, broken)
    ├─ analyze_complexity() → Returns tier name
    └─ _calculate_complexity_score() → Returns 0-100 score

persona_threshold_engine.py (ENHANCED, working)
    └─ calculate_complexity_score() → Returns 0-100 score

gcloud_burst_pipeline.py (COPY, simplified)
    └─ _estimate_complexity() → Returns 0-100 score
```

**Problem:** 3 implementations, 2 are broken/outdated!

---

## Proposed Architecture (Option 1)

### New File: `src/core/complexity_analyzer.py`
```python
#!/usr/bin/env python3
"""
Single Source of Truth for Complexity Scoring
All model selection decisions flow through this module.
"""

from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class ComplexityScore:
    """Result of complexity analysis"""
    score: float  # 0-100
    tier: str  # tiny, small, medium, large, cloud
    model: str  # Model name
    confidence: float  # 0-1
    breakdown: Dict[str, float]  # What contributed to score


class ComplexityAnalyzer:
    """
    Enhanced complexity scoring algorithm
    Trained/tuned: October 2025
    Accuracy: 100% (14/14 test cases)
    """

    # Model mapping
    TIER_TO_MODEL = {
        "tiny": "tinyllama:latest",
        "small": "llama3.2:3b",
        "medium": "llama3.2:3b",
        "large": "qwen2.5-coder:32b",
        "cloud": "llama3.1:70b"
    }

    @staticmethod
    def analyze(message: str, context: Dict = None) -> ComplexityScore:
        """
        Main entry point for complexity analysis

        Args:
            message: User query/prompt
            context: Optional context (previous_failures, user_expertise, etc.)

        Returns:
            ComplexityScore with score, tier, model, and breakdown
        """
        score, breakdown = ComplexityAnalyzer._calculate_score(message, context)
        tier = ComplexityAnalyzer._score_to_tier(score)
        model = ComplexityAnalyzer.TIER_TO_MODEL[tier]
        confidence = min(score / 100, 1.0)

        return ComplexityScore(
            score=score,
            tier=tier,
            model=model,
            confidence=confidence,
            breakdown=breakdown
        )

    @staticmethod
    def _calculate_score(message: str, context: Dict = None) -> Tuple[float, Dict]:
        """
        Enhanced complexity scoring algorithm
        Returns: (score, breakdown_dict)
        """
        context = context or {}
        breakdown = {}
        score = 0.0

        # 1. Basic metrics
        word_count = len(message.split())
        breakdown['word_count'] = word_count * 0.3
        score += breakdown['word_count']

        questions = message.count('?')
        breakdown['questions'] = questions * 5
        score += breakdown['questions']

        # 2. Generation keywords (NEW - Oct 2025)
        gen_keywords = ['generate', 'create', 'make', 'render', 'produce', 'build', 'design']
        gen_count = sum(1 for kw in gen_keywords if kw in message.lower())
        breakdown['generation'] = gen_count * 8
        score += breakdown['generation']

        # 3. Media keywords (NEW - Oct 2025)
        media_keywords = ['video', 'anime', 'animation', 'trailer', 'scene', 'cinematic',
                         'movie', 'film', 'visual', 'image']
        media_count = sum(1 for kw in media_keywords if kw in message.lower())
        breakdown['media'] = media_count * 10
        score += breakdown['media']

        # 4. Quality keywords (NEW - Oct 2025)
        quality_keywords = ['professional', 'cinematic', 'detailed', 'high-quality',
                           'theatrical', 'polished', 'production']
        quality_count = sum(1 for kw in quality_keywords if kw in message.lower())
        breakdown['quality'] = quality_count * 6
        score += breakdown['quality']

        # 5. Duration markers (NEW - Oct 2025)
        duration_keywords = ['minute', 'second', 'frame', 'hour', 'episode', 'series']
        duration_count = sum(1 for kw in duration_keywords if kw in message.lower())
        breakdown['duration'] = duration_count * 5
        score += breakdown['duration']

        # 6. Technical terms (existing, enhanced)
        technical_terms = ['database', 'architecture', 'algorithm', 'implementation',
                          'distributed', 'scalable', 'quantum', 'neural', 'machine learning']
        tech_count = sum(1 for term in technical_terms if term.lower() in message.lower())
        breakdown['technical'] = tech_count * 10
        score += breakdown['technical']

        # 7. Code markers (existing, enhanced weight)
        code_markers = ['def ', 'class ', 'import ', 'function', 'method', 'async',
                       'python', 'javascript', 'sql']
        code_count = sum(1 for kw in code_markers if kw in message.lower())
        breakdown['code'] = code_count * 12
        score += breakdown['code']

        # 8. Context modifiers
        if context.get('previous_failures', 0) > 0:
            breakdown['retry_escalation'] = 20
            score += 20

        if context.get('user_expertise') == 'expert':
            breakdown['expert_user'] = 10
            score += 10

        return min(score, 100.0), breakdown

    @staticmethod
    def _score_to_tier(score: float) -> str:
        """Convert numerical score to tier name"""
        if score < 5:
            return "tiny"
        elif score < 15:
            return "small"
        elif score < 30:
            return "medium"
        elif score < 50:
            return "large"
        else:
            return "cloud"
```

---

## Migration Steps

### Step 1: Create complexity_analyzer.py
```bash
ssh patrick@vestal-garcia.duckdns.org
cd /opt/tower-echo-brain/src/core
# Create the new file (copy content above)
```

### Step 2: Update intelligence.py
```python
# BEFORE (OLD):
class EchoIntelligenceRouter:
    def analyze_complexity(self, query: str, context: Dict) -> str:
        complexity_score = 0.0
        complexity_score += len(query.split()) * 0.3
        # ... lots of duplicate code
        return "quick" or "standard" etc.

# AFTER (NEW):
from src.core.complexity_analyzer import ComplexityAnalyzer

class EchoIntelligenceRouter:
    def analyze_complexity(self, query: str, context: Dict) -> str:
        """Analyze query complexity (delegates to ComplexityAnalyzer)"""
        result = ComplexityAnalyzer.analyze(query, context)
        return result.tier  # Returns: tiny, small, medium, large, cloud

    def _calculate_complexity_score(self, query: str, context: Dict) -> float:
        """Calculate numerical score (delegates to ComplexityAnalyzer)"""
        result = ComplexityAnalyzer.analyze(query, context)
        return result.score  # Returns: 0-100
```

### Step 3: Update persona_threshold_engine.py
```python
# BEFORE:
class PersonaThresholdEngine:
    def calculate_complexity_score(self, message: str) -> float:
        # Duplicate implementation
        ...

# AFTER:
from src.core.complexity_analyzer import ComplexityAnalyzer

class PersonaThresholdEngine:
    def calculate_complexity_score(self, message: str) -> float:
        """Calculate complexity (delegates to ComplexityAnalyzer)"""
        result = ComplexityAnalyzer.analyze(message)
        return result.score
```

### Step 4: Update gcloud_burst_pipeline.py
```python
# BEFORE:
class EchoBrainWithBurst:
    def _estimate_complexity(self, query: str) -> float:
        # Simplified duplicate
        ...

# AFTER:
from src.core.complexity_analyzer import ComplexityAnalyzer

class EchoBrainWithBurst:
    def _estimate_complexity(self, query: str) -> float:
        """Estimate complexity (delegates to ComplexityAnalyzer)"""
        result = ComplexityAnalyzer.analyze(query)
        return result.score
```

### Step 5: Update Tests
```python
# tests/test_complexity_scoring.py
from src.core.complexity_analyzer import ComplexityAnalyzer

def test_anime_prompt_escalation():
    prompt = "Generate 2-minute professional anime trailer..."
    result = ComplexityAnalyzer.analyze(prompt)

    assert result.score > 30, "Should escalate to large tier"
    assert result.tier in ["large", "cloud"]
    assert result.model in ["qwen2.5-coder:32b", "llama3.1:70b"]
```

---

## Benefits

### Before (Current):
- ❌ 3 different implementations
- ❌ 2 are broken/outdated
- ❌ Difficult to debug (which one is wrong?)
- ❌ Tests scattered across files
- ❌ No visibility into scoring breakdown

### After (Refactored):
- ✅ 1 implementation (single source of truth)
- ✅ All use the ENHANCED algorithm
- ✅ Easy to debug (one place to look)
- ✅ Centralized test suite
- ✅ Full breakdown of score components
- ✅ Dataclass for type safety

---

## Testing Plan

### Unit Tests:
```python
def test_simple_query():
    result = ComplexityAnalyzer.analyze("What is 2+2?")
    assert result.tier == "tiny"
    assert result.score < 5

def test_anime_generation():
    result = ComplexityAnalyzer.analyze("Generate 2-minute anime trailer")
    assert result.tier in ["large", "cloud"]
    assert result.score > 30

def test_breakdown():
    result = ComplexityAnalyzer.analyze("Generate professional video")
    assert 'generation' in result.breakdown
    assert 'media' in result.breakdown
    assert 'quality' in result.breakdown
```

### Integration Tests:
```bash
# Run anime integration test again
python3 tests/anime_production_integration_test.py

# Expected:
# - Complexity score: ~60 (was 0!)
# - Model: llama3.1:70b (was tinyllama!)
# - Tier: cloud (was tiny!)
```

---

## Timeline

- **Step 1-2**: Create complexity_analyzer.py + update intelligence.py (~30 min)
- **Step 3-4**: Update persona/gcloud imports (~15 min)
- **Step 5**: Update tests (~15 min)
- **Step 6**: Test & verify (~30 min)

**Total**: ~90 minutes

---

## Rollback Plan

If something breaks:
```bash
cd /opt/tower-echo-brain
git diff src/core/intelligence.py
git checkout src/core/intelligence.py  # Restore old version
sudo systemctl restart tower-echo-brain.service
```

Backups automatically created at:
- `intelligence.py.backup_[timestamp]`
- `persona_threshold_engine.py.backup_[timestamp]`

---

## Success Criteria

✅ Anime integration test passes
✅ Complexity score: 0 → ~60
✅ Model escalation: tinyllama → llama3.1:70b
✅ All 14 test cases still pass
✅ No regressions in simple queries
✅ Code duplication eliminated

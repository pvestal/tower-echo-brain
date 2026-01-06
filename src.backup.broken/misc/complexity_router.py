"""
Fixed model selection for AI Assist using ComplexityAnalyzer
This module provides WORKING model selection with enhanced complexity scoring
"""
import logging
from typing import Dict, Optional, Tuple
import time

logger = logging.getLogger(__name__)

class ModelSelector:
    """Model selection using ComplexityAnalyzer for consistent scoring"""

    def __init__(self):
        # Import ComplexityAnalyzer
        try:
            from src.core.complexity_analyzer import ComplexityAnalyzer
            self.analyzer = ComplexityAnalyzer()
            self.use_analyzer = True
            logger.info("✅ ComplexityAnalyzer loaded for model selection")
        except ImportError as e:
            logger.warning(f"⚠️ ComplexityAnalyzer not available: {e}, using fallback")
            self.analyzer = None
            self.use_analyzer = False

        # Models that EXIST and WORK
        self.model_hierarchy = {
            "quick": "tinyllama:latest",
            "standard": "llama3.2:3b",
            "professional": "mistral:7b-instruct",
            "expert": "qwen2.5-coder:32b",
            "genius": "llama3.1:70b",
            # Map legacy names
            "tiny": "tinyllama:latest",
            "small": "llama3.2:3b",
            "medium": "mistral:7b-instruct",
            "large": "qwen2.5-coder:32b",
            "xlarge": "llama3.1:70b",
            "cloud": "llama3.1:70b"
        }

        # Map complexity tiers to intelligence levels
        self.tier_to_level = {
            "tiny": "quick",
            "small": "standard",
            "medium": "professional",
            "large": "expert",
            "cloud": "genius"
        }

        # Performance expectations
        self.expected_times = {
            "tinyllama:latest": 1.0,
            "llama3.2:3b": 2.0,
            "mistral:7b-instruct": 4.0,
            "deepseek-coder-v2:16b": 5.0,
            "qwen2.5-coder:32b": 8.0,
            "llama3.1:70b": 15.0
        }

    def select_model(self, query: str, requested_level: Optional[str] = None) -> Tuple[str, str, str, Optional[float], Optional[str]]:
        """
        Select optimal model for query using ComplexityAnalyzer
        Returns: (model_name, level, reasoning, complexity_score, tier)
        """
        query_lower = query.lower()
        
        # PRIORITY: Detect "think harder" requests - ALWAYS escalate to highest model
        if "think harder" in query_lower:
            return (self.model_hierarchy["genius"], "genius", 
                   "Think harder mode - escalated to 70B model", 100.0, "cloud")
        
        # PRIORITY: Deep analysis requests should use high-end models
        if any(phrase in query_lower for phrase in [
            "analyze this", "review this", "refactor this", "optimize this",
            "improve this", "what.s wrong with", "better way to", "performance issue"
        ]):
            return (self.model_hierarchy["genius"], "genius", 
                   "Deep analysis request - using 70B model", 100.0, "cloud")
        
        # If user explicitly requests a level, RESPECT IT
        if requested_level and requested_level != "auto":
            if requested_level in self.model_hierarchy:
                model = self.model_hierarchy[requested_level]
                logger.info(f"User requested {requested_level}: using {model}")
                # Still calculate complexity for metadata
                complexity_score = None
                tier = None
                if self.use_analyzer:
                    try:
                        result = self.analyzer.analyze(query)
                        complexity_score = result.score
                        tier = result.tier
                    except Exception as e:
                        logger.error(f"Error calculating complexity: {e}")
                return (model, requested_level, f"User requested {requested_level}", 
                       complexity_score, tier)
            else:
                logger.warning(f"Unknown level {requested_level}, using standard")
                return (self.model_hierarchy["standard"], "standard", 
                       "Invalid level, using standard", None, None)

        # Use ComplexityAnalyzer if available
        if self.use_analyzer:
            try:
                # Prepare context
                context = {}
                
                # Analyze complexity
                result = self.analyzer.analyze(query, context)
                
                # Get model for this tier
                model = result.model
                level = self.tier_to_level.get(result.tier, "standard")
                
                # Build reasoning from breakdown
                breakdown_str = ", ".join([f"{k}={v:.1f}" for k, v in result.breakdown.items() if v > 0])
                reasoning = f"Complexity score {result.score:.1f} (tier: {result.tier}) - {breakdown_str}"
                
                logger.info(f"🎯 ComplexityAnalyzer: score={result.score:.1f}, tier={result.tier}, model={model}")
                
                return (model, level, reasoning, result.score, result.tier)
                
            except Exception as e:
                logger.error(f"❌ ComplexityAnalyzer failed: {e}", exc_info=True)
                # Fall through to fallback logic

        # FALLBACK: Simple heuristic-based selection if ComplexityAnalyzer not available
        query_len = len(query)
        
        # Quick responses for simple queries
        if query_len < 25 and any(w in query_lower for w in ["hi", "hello", "hey", "thanks", "bye"]):
            return (self.model_hierarchy["quick"], "quick", 
                   "Simple greeting/acknowledgment", None, "tiny")

        # Default based on length
        if query_len < 50:
            return (self.model_hierarchy["standard"], "standard", 
                   "Short general query (fallback)", None, "small")
        elif query_len < 150:
            return (self.model_hierarchy["professional"], "professional", 
                   "Medium complexity query (fallback)", None, "medium")
        else:
            return (self.model_hierarchy["genius"], "genius", 
                   "Long complex query (fallback)", None, "large")

    def get_expected_time(self, model: str) -> float:
        """Get expected response time for model"""
        return self.expected_times.get(model, 5.0)

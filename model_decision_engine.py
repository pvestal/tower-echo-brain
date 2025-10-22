#!/usr/bin/env python3
"""
Model Decision Engine for AI Assist
Intelligent routing between local models and API with performance tracking
"""

import time
import json
import psycopg2
import psycopg2.pool
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
from contextlib import contextmanager
from code_quality_validator import CodeQualityValidator, ModelReloadManager, ValidationResult

logger = logging.getLogger(__name__)

class ModelTier(str, Enum):
    TINY = "tiny"          # 1B params - basic queries
    SMALL = "small"        # 3-7B params - standard coding
    MEDIUM = "medium"      # 13-16B params - complex coding
    LARGE = "large"        # 32-70B params - architecture
    CLOUD = "cloud"        # API only - cutting edge

class QueryComplexity:
    """Analyze query complexity with learnable weights"""

    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.load_weights()

    def load_weights(self):
        """Load learned weights from database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT feature, weight FROM model_decision_weights
                WHERE active = true
            """)

            self.weights = dict(cursor.fetchall()) if cursor.rowcount > 0 else self.default_weights()
            conn.close()

        except Exception as e:
            logger.warning(f"Using default weights: {e}")
            self.weights = self.default_weights()

    def default_weights(self) -> Dict[str, float]:
        return {
            "code_complexity": 1.0,
            "context_lines": 0.5,
            "technical_depth": 2.0,
            "multi_file": 3.0,
            "architecture": 4.0,
            "debugging": 1.5,
            "optimization": 2.5,
            "security": 2.0,
            "api_design": 2.5,
            "refactoring": 3.0
        }

    def analyze(self, query: str, context: Dict) -> Tuple[float, Dict[str, float]]:
        """
        Analyze query and return complexity score with feature breakdown
        """
        features = {}

        # Code complexity indicators
        code_indicators = ['implement', 'write', 'create', 'build', 'develop']
        features['code_complexity'] = sum(1 for ind in code_indicators if ind in query.lower())

        # Context size
        context_size = len(str(context))
        features['context_lines'] = min(context_size / 1000, 10)  # Scale to 0-10

        # Technical depth
        tech_terms = ['algorithm', 'optimize', 'performance', 'scalability', 'distributed',
                     'microservice', 'architecture', 'design pattern', 'async', 'concurrent']
        features['technical_depth'] = sum(1 for term in tech_terms if term in query.lower())

        # Multi-file operations
        multi_file = ['refactor', 'migrate', 'reorganize', 'across', 'multiple files']
        features['multi_file'] = sum(2 for term in multi_file if term in query.lower())

        # Architecture decisions
        arch_terms = ['design', 'architect', 'structure', 'pattern', 'framework choice']
        features['architecture'] = sum(2 for term in arch_terms if term in query.lower())

        # Debugging complexity
        debug_terms = ['debug', 'fix', 'error', 'bug', 'issue', 'problem', 'failing']
        features['debugging'] = sum(1 for term in debug_terms if term in query.lower())

        # Optimization requests
        opt_terms = ['optimize', 'improve', 'faster', 'performance', 'efficient']
        features['optimization'] = sum(1.5 for term in opt_terms if term in query.lower())

        # Security concerns
        sec_terms = ['security', 'vulnerability', 'authentication', 'authorization', 'encryption']
        features['security'] = sum(2 for term in sec_terms if term in query.lower())

        # API design
        api_terms = ['api', 'endpoint', 'rest', 'graphql', 'websocket', 'interface']
        features['api_design'] = sum(1.5 for term in api_terms if term in query.lower())

        # Large refactoring
        refactor_terms = ['refactor', 'rewrite', 'restructure', 'modernize', 'upgrade']
        features['refactoring'] = sum(2 for term in refactor_terms if term in query.lower())

        # Calculate weighted score
        score = sum(features.get(k, 0) * self.weights.get(k, 1.0) for k in self.weights.keys())

        return score, features

class ModelDecisionEngine:
    """
    Intelligent model selection with performance tracking and learning
    """

    def __init__(self, db_config: Dict):
        self.db_config = db_config

        # Initialize connection pool
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=20,
                host=db_config.get('host', 'localhost'),
                database=db_config.get('database', 'echo_brain'),
                user=db_config.get('user', 'patrick'),
                password=db_config.get('password', 'patrick123')
            )
            logger.info("✅ Database connection pool initialized (20 connections)")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            # Fallback to regular connections
            self.connection_pool = None

        self.complexity_analyzer = QueryComplexity(db_config)
        self.init_database()
        self.performance_cache = {}

        # Initialize code quality validator
        self.code_validator = CodeQualityValidator()
        self.reload_manager = None  # Will be set when model_manager is available

        # Model configurations with actual installed models
        self.local_models = {
            ModelTier.TINY: [
                {"name": "tinyllama:latest", "params": 1, "specialization": None}
            ],
            ModelTier.SMALL: [
                {"name": "deepseek-coder:latest", "params": 1.3, "specialization": "coding"},
                {"name": "llama3.2:3b", "params": 3, "specialization": None},
                {"name": "mistral:7b", "params": 7, "specialization": None},
                {"name": "codellama:7b", "params": 7, "specialization": "coding"}
            ],
            ModelTier.MEDIUM: [
                {"name": "codellama:13b", "params": 13, "specialization": "coding"},
                {"name": "deepseek-coder-v2:16b", "params": 16, "specialization": "coding"}
            ],
            ModelTier.LARGE: [
                {"name": "qwen2.5-coder:32b", "params": 32, "specialization": "coding"},
                {"name": "codellama:70b", "params": 70, "specialization": "analysis"},
                {"name": "llama3.1:70b", "params": 70, "specialization": None}
            ]
        }

        # Thresholds (learnable)
        self.load_thresholds()

    @contextmanager
    def get_db_connection(self):
        """Get connection from pool with proper cleanup"""
        if self.connection_pool:
            conn = None
            try:
                conn = self.connection_pool.getconn()
                yield conn
                conn.commit()
            except Exception as e:
                if conn:
                    conn.rollback()
                raise e
            finally:
                if conn:
                    self.connection_pool.putconn(conn)
        else:
            # Fallback to regular connection
            conn = psycopg2.connect(**self.db_config)
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

    def init_database(self):
        """Initialize performance tracking tables"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Model decision history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_decisions (
                    decision_id SERIAL PRIMARY KEY,
                    query_hash VARCHAR(64),
                    query_text TEXT,
                    complexity_score FLOAT,
                    features JSONB,
                    selected_model VARCHAR(100),
                    model_tier VARCHAR(20),
                    response_time FLOAT,
                    token_count INTEGER,
                    success BOOLEAN,
                    user_feedback INTEGER,
                    api_fallback BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_model_decisions_model
                ON model_decisions(selected_model);

                CREATE INDEX IF NOT EXISTS idx_model_decisions_score
                ON model_decisions(complexity_score);
            """)

            # Model performance metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    model_name VARCHAR(100) PRIMARY KEY,
                    total_queries INTEGER DEFAULT 0,
                    successful_queries INTEGER DEFAULT 0,
                    avg_response_time FLOAT,
                    avg_tokens_per_second FLOAT,
                    satisfaction_score FLOAT,
                    complexity_sweet_spot FLOAT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Decision weights (learnable parameters)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_decision_weights (
                    feature VARCHAR(100) PRIMARY KEY,
                    weight FLOAT DEFAULT 1.0,
                    update_count INTEGER DEFAULT 0,
                    active BOOLEAN DEFAULT TRUE
                );
            """)

            # Complexity thresholds
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS complexity_thresholds (
                    tier VARCHAR(20) PRIMARY KEY,
                    min_score FLOAT,
                    max_score FLOAT,
                    auto_escalate BOOLEAN DEFAULT TRUE,
                    last_adjusted TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Insert default thresholds if not exist
                INSERT INTO complexity_thresholds (tier, min_score, max_score)
                VALUES
                    ('tiny', 0, 5),
                    ('small', 5, 15),
                    ('medium', 15, 30),
                    ('large', 30, 50),
                    ('cloud', 50, 999)
                ON CONFLICT DO NOTHING;
            """)

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to initialize decision database: {e}")

    def load_thresholds(self):
        """Load current complexity thresholds"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT tier, min_score, max_score, auto_escalate
                FROM complexity_thresholds
            """)

            self.thresholds = {}
            for tier, min_score, max_score, auto_escalate in cursor.fetchall():
                self.thresholds[tier] = {
                    "min": min_score,
                    "max": max_score,
                    "auto_escalate": auto_escalate
                }

            conn.close()

        except Exception as e:
            logger.warning(f"Using default thresholds: {e}")
            self.thresholds = {
                "tiny": {"min": 0, "max": 5, "auto_escalate": True},
                "small": {"min": 5, "max": 15, "auto_escalate": True},
                "medium": {"min": 15, "max": 30, "auto_escalate": True},
                "large": {"min": 30, "max": 50, "auto_escalate": True},
                "cloud": {"min": 50, "max": 999, "auto_escalate": False}
            }

    async def decide_model(self, query: str, context: Dict) -> Dict:
        """
        Make intelligent model decision based on query analysis
        """
        # Analyze complexity
        complexity_score, features = self.complexity_analyzer.analyze(query, context)

        # Check for user preference override
        if context.get("preferred_model"):
            return {
                "model": context["preferred_model"],
                "tier": self._get_model_tier(context["preferred_model"]),
                "complexity_score": complexity_score,
                "features": features,
                "reason": "User preference override"
            }

        # Determine tier based on complexity
        selected_tier = self._select_tier(complexity_score)

        # Check if we need API fallback
        if selected_tier == ModelTier.CLOUD or context.get("force_api"):
            return {
                "model": "deepseek-api",
                "tier": ModelTier.CLOUD,
                "complexity_score": complexity_score,
                "features": features,
                "reason": "Complexity exceeds local capacity",
                "use_api": True
            }

        # Select best model from tier
        model = await self._select_best_model(selected_tier, features, context)

        # Record decision
        self._record_decision(query, complexity_score, features, model, selected_tier)

        return {
            "model": model["name"],
            "tier": selected_tier,
            "complexity_score": complexity_score,
            "features": features,
            "reason": f"Optimal for complexity score {complexity_score:.1f}",
            "use_api": False
        }

    def _select_tier(self, complexity_score: float) -> ModelTier:
        """Select appropriate tier based on complexity score"""
        for tier_name, threshold in self.thresholds.items():
            if threshold["min"] <= complexity_score < threshold["max"]:
                return ModelTier(tier_name)

        # Default to large for high complexity
        return ModelTier.LARGE if complexity_score < 50 else ModelTier.CLOUD

    async def _select_best_model(self, tier: ModelTier, features: Dict, context: Dict) -> Dict:
        """Select the best model within a tier based on specialization and performance"""

        candidates = self.local_models.get(tier, [])

        if not candidates:
            # Escalate to next tier
            next_tier = self._escalate_tier(tier)
            return await self._select_best_model(next_tier, features, context)

        # Check for specialization needs
        if features.get('code_complexity', 0) > 0 or features.get('debugging', 0) > 0:
            # Prefer coding specialists
            coding_models = [m for m in candidates if m.get('specialization') == 'coding']
            if coding_models:
                return self._pick_by_performance(coding_models)

        # Default to best performing model in tier
        return self._pick_by_performance(candidates)

    def _pick_by_performance(self, models: List[Dict]) -> Dict:
        """Pick model based on historical performance"""

        if not models:
            return {"name": "mistral:7b", "params": 7}  # Fallback

        # If we have performance data, use it
        if self.performance_cache:
            scored_models = []
            for model in models:
                perf = self.performance_cache.get(model['name'], {})
                score = perf.get('satisfaction_score', 0.5) * 100 + \
                       (100 - perf.get('avg_response_time', 50))
                scored_models.append((score, model))

            scored_models.sort(reverse=True)
            return scored_models[0][1]

        # Default to first model
        return models[0]

    def _escalate_tier(self, current_tier: ModelTier) -> ModelTier:
        """Escalate to next tier"""
        tier_order = [ModelTier.TINY, ModelTier.SMALL, ModelTier.MEDIUM,
                     ModelTier.LARGE, ModelTier.CLOUD]

        current_index = tier_order.index(current_tier)
        next_index = min(current_index + 1, len(tier_order) - 1)

        return tier_order[next_index]

    def _get_model_tier(self, model_name: str) -> ModelTier:
        """Get tier of a specific model"""
        for tier, models in self.local_models.items():
            if any(m['name'] == model_name for m in models):
                return tier
        return ModelTier.CLOUD

    def _record_decision(self, query: str, score: float, features: Dict,
                        model: Dict, tier: ModelTier):
        """Record decision for learning"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            import hashlib
            query_hash = hashlib.sha256(query.encode()).hexdigest()

            cursor.execute("""
                INSERT INTO model_decisions
                (query_hash, query_text, complexity_score, features,
                 selected_model, model_tier, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING decision_id
            """, (
                query_hash, query[:1000], score, json.dumps(features),
                model.get('name', 'unknown'), tier.value, datetime.now()
            ))

            decision_id = cursor.fetchone()[0]
            conn.commit()
            conn.close()

            return decision_id

        except Exception as e:
            logger.error(f"Failed to record decision: {e}")
            return None

    async def record_performance(self, decision_id: int, response_time: float,
                                token_count: int, success: bool):
        """Record actual performance for learning"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE model_decisions
                SET response_time = %s,
                    token_count = %s,
                    success = %s
                WHERE decision_id = %s
            """, (response_time, token_count, success, decision_id))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to record performance: {e}")

    async def record_feedback(self, decision_id: int, feedback: int):
        """Record user feedback (1-5 scale)"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE model_decisions
                SET user_feedback = %s
                WHERE decision_id = %s
            """, (feedback, decision_id))

            # Trigger learning update if feedback is poor
            if feedback <= 2:
                await self._adjust_weights(decision_id)

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")

    async def _adjust_weights(self, decision_id: int):
        """Adjust weights based on poor feedback"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Get decision details
            cursor.execute("""
                SELECT complexity_score, features, model_tier, user_feedback
                FROM model_decisions
                WHERE decision_id = %s
            """, (decision_id,))

            score, features, tier, feedback = cursor.fetchone()

            # If model was too weak, increase weights for active features
            # If model was too strong, decrease weights
            adjustment = 0.1 if feedback <= 2 else -0.05

            for feature, value in features.items():
                if value > 0:
                    cursor.execute("""
                        INSERT INTO model_decision_weights (feature, weight)
                        VALUES (%s, %s)
                        ON CONFLICT (feature) DO UPDATE
                        SET weight = model_decision_weights.weight + %s,
                            update_count = model_decision_weights.update_count + 1
                    """, (feature, 1.0 + adjustment, adjustment))

            conn.commit()
            conn.close()

            # Reload weights
            self.complexity_analyzer.load_weights()

        except Exception as e:
            logger.error(f"Failed to adjust weights: {e}")

    async def validate_and_reload_if_needed(
        self,
        output: str,
        query: str,
        model_used: str,
        expected_type: str = "general"
    ) -> Tuple[str, str, ValidationResult]:
        """
        Validate output quality and reload with better model if needed
        Returns: (final_output, model_used, validation_result)
        """

        # For code generation tasks, validate the output
        if expected_type == "code" or self._is_code_generation_task(query):
            validation = self.code_validator.validate_code(output)

            if validation.is_gibberish or validation.quality_score < 0.3:
                logger.warning(
                    f"🔄 Model {model_used} produced invalid output. "
                    f"Quality: {validation.quality_score:.2f}, "
                    f"Gibberish: {validation.is_gibberish}, "
                    f"Issues: {validation.issues}"
                )

                # Try with a better model
                better_output, better_model = await self._retry_with_better_model(
                    query, model_used, validation
                )

                if better_output:
                    # Re-validate the new output
                    new_validation = self.code_validator.validate_code(better_output)
                    return better_output, better_model, new_validation

            return output, model_used, validation

        # For non-code tasks, return original output
        return output, model_used, ValidationResult(
            is_valid=True,
            confidence=1.0,
            issues=[],
            language=None,
            quality_score=1.0,
            is_gibberish=False,
            requires_reload=False
        )

    def _is_code_generation_task(self, query: str) -> bool:
        """Detect if the query is asking for code generation"""
        code_indicators = [
            "write code", "generate code", "create a function", "implement",
            "write a script", "create a class", "code for", "program",
            "python", "javascript", "sql", "bash", "typescript"
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in code_indicators)

    async def _retry_with_better_model(
        self,
        query: str,
        failed_model: str,
        validation: ValidationResult
    ) -> Tuple[str, str]:
        """Retry with a progressively better model"""

        # Determine current tier
        current_tier = self._get_model_tier(failed_model)

        # Try escalating tiers
        escalation_order = [ModelTier.SMALL, ModelTier.MEDIUM, ModelTier.LARGE]

        for tier in escalation_order:
            if tier.value <= current_tier.value:
                continue

            # Get models in this tier specialized for coding
            tier_models = self.local_models.get(tier, [])
            coding_models = [m for m in tier_models if m.get("specialization") == "coding"]

            if not coding_models:
                coding_models = tier_models  # Fall back to any model in tier

            for model_config in coding_models:
                model_name = model_config["name"]

                if await self._is_model_available(model_name):
                    logger.info(f"🔄 Retrying with {model_name} (tier: {tier})")

                    try:
                        # Enhanced prompt for better results
                        enhanced_query = self._enhance_query_for_code(query, validation.issues)

                        # Call the model (this would integrate with your model manager)
                        result = await self._call_model(model_name, enhanced_query)

                        if result:
                            return result, model_name

                    except Exception as e:
                        logger.error(f"Failed with {model_name}: {e}")
                        continue

        # If all retries fail, return empty
        logger.error("❌ All model retries failed")
        return "", failed_model

    def _enhance_query_for_code(self, query: str, issues: List[str]) -> str:
        """Enhance the query to avoid previous issues"""

        enhancements = []

        if any("gibberish" in str(issue).lower() for issue in issues):
            enhancements.append("Generate syntactically correct code only.")

        if any("syntax" in str(issue).lower() for issue in issues):
            enhancements.append("Ensure proper syntax with balanced brackets.")

        if any("no language" in str(issue).lower() for issue in issues):
            enhancements.append("Clearly use proper programming language syntax.")

        enhancement_text = " ".join(enhancements)

        return f"{query}\n\nIMPORTANT: {enhancement_text} Output ONLY valid code without explanation."

    async def _is_model_available(self, model_name: str) -> bool:
        """Check if a model is available in Ollama"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m['name'] for m in data.get('models', [])]
                        return model_name in models
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
        return False

    async def _call_model(self, model_name: str, query: str) -> Optional[str]:
        """Call specific model via Ollama API with proper error handling"""
        try:
            timeout = aiohttp.ClientTimeout(total=30.0, connect=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                payload = {
                    "model": model_name,
                    "prompt": query,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 2000
                    }
                }

                async with session.post(
                    "http://localhost:11434/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        logger.error(f"Ollama API error: HTTP {response.status}")
                        return None

        except asyncio.TimeoutError:
            logger.error(f"Timeout calling model {model_name}")
            return None
        except Exception as e:
            logger.error(f"Failed to call model {model_name}: {e}")
            return None

    def get_performance_summary(self) -> Dict:
        """Get performance summary for monitoring"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Overall stats
            cursor.execute("""
                SELECT
                    COUNT(*) as total_decisions,
                    AVG(complexity_score) as avg_complexity,
                    AVG(response_time) as avg_response_time,
                    AVG(user_feedback) as avg_feedback,
                    SUM(CASE WHEN api_fallback THEN 1 ELSE 0 END) as api_fallbacks
                FROM model_decisions
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)

            stats = cursor.fetchone()

            # Per-model performance
            cursor.execute("""
                SELECT
                    selected_model,
                    COUNT(*) as queries,
                    AVG(response_time) as avg_time,
                    AVG(user_feedback) as avg_feedback
                FROM model_decisions
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY selected_model
            """)

            model_stats = cursor.fetchall()

            conn.close()

            return {
                "summary": {
                    "total_decisions": stats[0] or 0,
                    "avg_complexity": stats[1] or 0,
                    "avg_response_time": stats[2] or 0,
                    "avg_feedback": stats[3] or 0,
                    "api_fallbacks": stats[4] or 0
                },
                "models": {
                    model: {
                        "queries": queries,
                        "avg_time": avg_time,
                        "avg_feedback": avg_feedback
                    }
                    for model, queries, avg_time, avg_feedback in model_stats
                }
            }

        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}

# Singleton instance
_engine_instance: Optional[ModelDecisionEngine] = None

def get_decision_engine(db_config: Dict) -> ModelDecisionEngine:
    """Get or create decision engine instance"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ModelDecisionEngine(db_config)
    return _engine_instance
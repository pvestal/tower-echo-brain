#!/usr/bin/env python3
"""
Echo Brain KB Enhanced Search - Pattern Learning System
Intelligent KB search with auto-optimization from successful generations
"""

import asyncio
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import aiohttp
from dataclasses import dataclass, asdict
import sqlite3
from pathlib import Path

@dataclass
class SearchPattern:
    """Pattern learned from successful searches/generations"""
    pattern_id: str
    query_terms: List[str]
    successful_params: Dict[str, Any]
    quality_score: float
    usage_count: int
    created_at: str
    last_used: str
    context_type: str  # anime, video, music, etc.

@dataclass
class GenerationMetrics:
    """Metrics from a generation to learn from"""
    duration: float
    resolution: str
    fps: int
    quality_score: float
    user_satisfaction: Optional[float]
    parameters_used: Dict[str, Any]
    timestamp: str

class PatternDatabase:
    """SQLite database for storing learned patterns"""

    def __init__(self, db_path: str = "/opt/tower-echo-brain/data/patterns.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_patterns (
                pattern_id TEXT PRIMARY KEY,
                query_terms TEXT,
                successful_params TEXT,
                quality_score REAL,
                usage_count INTEGER,
                created_at TEXT,
                last_used TEXT,
                context_type TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generation_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                duration REAL,
                resolution TEXT,
                fps INTEGER,
                quality_score REAL,
                user_satisfaction REAL,
                parameters_used TEXT,
                timestamp TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kb_search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                results_found INTEGER,
                relevant_articles TEXT,
                timestamp TEXT
            )
        """)

        conn.commit()
        conn.close()

    def save_pattern(self, pattern: SearchPattern):
        """Save a learned pattern"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO search_patterns
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pattern.pattern_id,
            json.dumps(pattern.query_terms),
            json.dumps(pattern.successful_params),
            pattern.quality_score,
            pattern.usage_count,
            pattern.created_at,
            pattern.last_used,
            pattern.context_type
        ))

        conn.commit()
        conn.close()

    def get_similar_patterns(self, query_terms: List[str], context_type: str = None) -> List[SearchPattern]:
        """Find similar successful patterns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query
        if context_type:
            cursor.execute("""
                SELECT * FROM search_patterns
                WHERE context_type = ? AND quality_score > 0.7
                ORDER BY quality_score DESC, usage_count DESC
                LIMIT 5
            """, (context_type,))
        else:
            cursor.execute("""
                SELECT * FROM search_patterns
                WHERE quality_score > 0.7
                ORDER BY quality_score DESC, usage_count DESC
                LIMIT 5
            """)

        patterns = []
        for row in cursor.fetchall():
            patterns.append(SearchPattern(
                pattern_id=row[0],
                query_terms=json.loads(row[1]),
                successful_params=json.loads(row[2]),
                quality_score=row[3],
                usage_count=row[4],
                created_at=row[5],
                last_used=row[6],
                context_type=row[7]
            ))

        conn.close()

        # Filter by term similarity
        relevant_patterns = []
        for pattern in patterns:
            similarity = len(set(query_terms) & set(pattern.query_terms))
            if similarity > 0:
                relevant_patterns.append((similarity, pattern))

        # Sort by similarity then quality
        relevant_patterns.sort(key=lambda x: (x[0], x[1].quality_score), reverse=True)

        return [p[1] for p in relevant_patterns[:3]]

    def save_metrics(self, metrics: GenerationMetrics):
        """Save generation metrics for learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO generation_metrics
            (duration, resolution, fps, quality_score, user_satisfaction, parameters_used, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.duration,
            metrics.resolution,
            metrics.fps,
            metrics.quality_score,
            metrics.user_satisfaction,
            json.dumps(metrics.parameters_used),
            metrics.timestamp
        ))

        conn.commit()
        conn.close()

    def get_best_parameters(self, context_type: str) -> Dict[str, Any]:
        """Get best performing parameters for a context type"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get top performing generations
        cursor.execute("""
            SELECT parameters_used, quality_score
            FROM generation_metrics
            WHERE quality_score > 0.8
            ORDER BY quality_score DESC
            LIMIT 10
        """)

        best_params = {}
        for row in cursor.fetchall():
            params = json.loads(row[0])
            if params.get("context_type") == context_type:
                best_params = params
                break

        conn.close()

        # Return defaults if nothing found
        if not best_params:
            best_params = {
                "video": {"duration": 30, "fps": 24, "resolution": "1920x1080"},
                "image": {"resolution": "1024x1024", "quality": 95},
                "anime": {"style": "anime", "quality": "masterpiece"}
            }.get(context_type, {})

        return best_params

class EnhancedKBSearch:
    """Enhanced KB search with pattern learning and auto-optimization"""

    def __init__(self):
        self.kb_endpoint = "https://192.168.50.135/api/kb"
        self.pattern_db = PatternDatabase()
        self.search_cache = {}  # Simple in-memory cache

    async def intelligent_search(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform intelligent KB search with pattern learning"""
        # Extract query terms
        query_terms = self._extract_terms(query)
        context_type = context.get("type", "general") if context else "general"

        # Check cache
        cache_key = f"{query}:{context_type}"
        if cache_key in self.search_cache:
            cached = self.search_cache[cache_key]
            if datetime.fromisoformat(cached["timestamp"]) > datetime.now() - timedelta(minutes=5):
                return cached["results"]

        # Find similar successful patterns
        similar_patterns = self.pattern_db.get_similar_patterns(query_terms, context_type)

        # Enhance search with learned patterns
        enhanced_query = self._enhance_query(query, similar_patterns)

        # Perform KB search
        search_results = await self._search_kb(enhanced_query)

        # Apply learned optimizations
        optimized_results = self._apply_optimizations(search_results, similar_patterns)

        # Cache results
        self.search_cache[cache_key] = {
            "results": optimized_results,
            "timestamp": datetime.now().isoformat()
        }

        return optimized_results

    def _extract_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from query"""
        # Remove common words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "for", "of", "to"}

        # Extract words
        words = re.findall(r'\w+', query.lower())

        # Filter stop words and short words
        terms = [w for w in words if w not in stop_words and len(w) > 2]

        return terms

    def _enhance_query(self, query: str, patterns: List[SearchPattern]) -> str:
        """Enhance query based on successful patterns"""
        enhanced = query

        if patterns:
            # Add successful terms from similar patterns
            additional_terms = set()
            for pattern in patterns[:2]:  # Top 2 patterns
                for term in pattern.query_terms:
                    if term not in query.lower():
                        additional_terms.add(term)

            if additional_terms:
                enhanced = f"{query} {' '.join(additional_terms)}"

        return enhanced

    async def _search_kb(self, query: str) -> List[Dict[str, Any]]:
        """Search KB with enhanced query"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {"q": query, "limit": 10}

                async with session.get(
                    f"{self.kb_endpoint}/search",
                    params=params,
                    ssl=False
                ) as response:
                    if response.status == 200:
                        return await response.json()

        except Exception as e:
            print(f"KB search error: {e}")

        # Return default results
        return [
            {
                "id": 71,
                "title": "Video Generation Standards",
                "content": "Video standards: minimum 30 seconds, 1920x1080, 24fps",
                "relevance": 0.8
            }
        ]

    def _apply_optimizations(self, results: List[Dict], patterns: List[SearchPattern]) -> Dict[str, Any]:
        """Apply learned optimizations to search results"""
        optimized = {
            "search_results": results,
            "suggested_parameters": {},
            "learned_insights": []
        }

        if patterns:
            # Get most successful parameters
            best_pattern = patterns[0]
            optimized["suggested_parameters"] = best_pattern.successful_params

            # Add insights
            for pattern in patterns[:3]:
                optimized["learned_insights"].append({
                    "pattern": pattern.pattern_id,
                    "quality_score": pattern.quality_score,
                    "usage_count": pattern.usage_count,
                    "params": pattern.successful_params
                })

        return optimized

    async def learn_from_generation(self, query: str, metrics: GenerationMetrics):
        """Learn from a successful generation"""
        # Extract terms
        query_terms = self._extract_terms(query)

        # Create pattern if quality is good
        if metrics.quality_score >= 0.8:
            pattern = SearchPattern(
                pattern_id=f"pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                query_terms=query_terms,
                successful_params=metrics.parameters_used,
                quality_score=metrics.quality_score,
                usage_count=1,
                created_at=datetime.now().isoformat(),
                last_used=datetime.now().isoformat(),
                context_type=metrics.parameters_used.get("type", "general")
            )

            # Save pattern
            self.pattern_db.save_pattern(pattern)

        # Save metrics
        self.pattern_db.save_metrics(metrics)

    def get_auto_parameters(self, context_type: str) -> Dict[str, Any]:
        """Get auto-optimized parameters based on learning"""
        return self.pattern_db.get_best_parameters(context_type)

class SmartSearchExamples:
    """Examples of enhanced KB search capabilities"""

    @staticmethod
    async def demonstrate_search():
        """Show enhanced search capabilities"""
        search = EnhancedKBSearch()

        print("=== Enhanced KB Search Demonstrations ===\n")

        # Example 1: Search for video standards
        print("1. Searching for 'video standards':")
        results = await search.intelligent_search(
            "video standards trailer",
            {"type": "video"}
        )
        print(f"   Found {len(results['search_results'])} articles")
        if results['suggested_parameters']:
            print(f"   Suggested params: {results['suggested_parameters']}")
        print()

        # Example 2: Learn from successful generation
        print("2. Learning from successful generation:")
        metrics = GenerationMetrics(
            duration=35.0,
            resolution="1920x1080",
            fps=24,
            quality_score=0.92,
            user_satisfaction=0.95,
            parameters_used={
                "type": "video",
                "style": "anime",
                "frames": 30,
                "quality": "high"
            },
            timestamp=datetime.now().isoformat()
        )
        await search.learn_from_generation("generate anime trailer", metrics)
        print("   Pattern learned and saved!")
        print()

        # Example 3: Get auto-optimized parameters
        print("3. Getting auto-optimized parameters for video:")
        auto_params = search.get_auto_parameters("video")
        print(f"   Auto parameters: {auto_params}")

# Integration with Echo Brain
class EchoKBIntegration:
    """Integrates enhanced KB search with Echo Brain"""

    def __init__(self):
        self.kb_search = EnhancedKBSearch()

    async def search_and_optimize(self, query: str, task_type: str = None) -> Dict[str, Any]:
        """Search KB and get optimized parameters"""
        context = {"type": task_type} if task_type else {}

        # Perform intelligent search
        results = await self.kb_search.intelligent_search(query, context)

        # Get auto-optimized parameters
        if task_type:
            auto_params = self.kb_search.get_auto_parameters(task_type)
            results["auto_parameters"] = auto_params

        return results

    async def feedback_loop(self, query: str, generation_result: Dict[str, Any]):
        """Provide feedback for learning"""
        # Create metrics from result
        metrics = GenerationMetrics(
            duration=generation_result.get("duration", 0),
            resolution=generation_result.get("resolution", "1920x1080"),
            fps=generation_result.get("fps", 24),
            quality_score=generation_result.get("quality_score", 0),
            user_satisfaction=generation_result.get("user_satisfaction"),
            parameters_used=generation_result.get("parameters", {}),
            timestamp=datetime.now().isoformat()
        )

        # Learn from this generation
        await self.kb_search.learn_from_generation(query, metrics)

if __name__ == "__main__":
    print("Echo KB Enhanced Search Module")
    print("Features:")
    print("  - Pattern learning from successful generations")
    print("  - Auto-optimization of parameters")
    print("  - Intelligent query enhancement")
    print("  - Quality-based parameter suggestions")
    print("\nRunning demonstrations...\n")

    asyncio.run(SmartSearchExamples.demonstrate_search())
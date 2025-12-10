# Echo Brain Developer Guide

## Overview

Echo Brain is a sophisticated AI orchestration system with an advanced learning architecture that continuously improves through comprehensive data ingestion and pattern recognition. This guide covers the new learning system components and how to work with them effectively.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Separation of Concerns Pattern](#separation-of-concerns-pattern)
3. [Comprehensive Learning Pipeline](#comprehensive-learning-pipeline)
4. [Business Logic Components](#business-logic-components)
5. [Development Best Practices](#development-best-practices)
6. [Testing Framework](#testing-framework)
7. [Performance Monitoring](#performance-monitoring)
8. [Code Quality Standards](#code-quality-standards)
9. [Pattern Development Guide](#pattern-development-guide)
10. [API Documentation](#api-documentation)

## Architecture Overview

Echo Brain's learning architecture is built on three core principles:

### 1. **Separation of Concerns**
Each component has a single responsibility:
- `BusinessLogicPatternMatcher` - Pattern retrieval only
- `BusinessLogicApplicator` - Response modification only
- `BusinessLogicMiddleware` - Centralized application
- `ConversationManager` - Conversation logic only

### 2. **Comprehensive Learning Pipeline**
Multi-source data ingestion from:
- Codebase analysis (Tower services, Echo Brain)
- Database content mining (conversations, learnings, domain data)
- System configuration analysis (nginx, systemd, etc.)
- Documentation processing (KB articles, README files)
- Git history patterns

### 3. **Real-time Feedback Integration**
- Continuous pattern validation
- Effectiveness testing
- User feedback processing
- Automatic pattern refinement

## Separation of Concerns Pattern

The new architecture strictly separates pattern matching from pattern application:

### BusinessLogicPatternMatcher
**Location**: `/opt/tower-echo-brain/src/services/business_logic_matcher.py`
**Responsibility**: Pattern retrieval only

```python
from services.business_logic_matcher import BusinessLogicPatternMatcher

matcher = BusinessLogicPatternMatcher()
patterns = matcher.get_relevant_patterns(query)
transformed_patterns = matcher.transform_patterns_for_application(patterns)
```

**Key Features**:
- 5-minute pattern caching for performance
- Confidence-based pattern ranking
- Multiple fact type support (technical_stack_preferences, quality_standards, communication_patterns, project_priorities, naming_standards)
- Query relevance scoring

### BusinessLogicApplicator
**Location**: `/opt/tower-echo-brain/src/services/business_logic_applicator.py`
**Responsibility**: Response modification only

```python
from services.business_logic_applicator import BusinessLogicApplicator

applicator = BusinessLogicApplicator()
modified_response = applicator.apply_patterns_to_response(query, base_response, patterns)
```

**Pattern Application Types**:
- **Preference Patterns**: Technology choices (PostgreSQL, Vue.js)
- **Requirement Patterns**: Quality standards (proof requirements, verification)
- **Anti-pattern Filters**: Remove unwanted terms/approaches
- **Context Patterns**: Project status awareness

### BusinessLogicMiddleware
**Location**: `/opt/tower-echo-brain/src/api/business_logic_middleware.py`
**Responsibility**: Centralized pattern application

```python
from api.business_logic_middleware import apply_business_logic_to_response

# Apply patterns to any API response
enhanced_response = apply_business_logic_to_response(query, response, "query")
```

**Benefits**:
- Consistent pattern application across all endpoints
- Performance tracking and statistics
- Error handling with graceful fallbacks
- Configurable per endpoint type

## Comprehensive Learning Pipeline

The learning system ingests data from multiple sources to build comprehensive knowledge:

### ComprehensiveDataIngestion
**Location**: `/opt/tower-echo-brain/src/learning/comprehensive_data_ingestion.py`

```python
from learning.comprehensive_data_ingestion import comprehensive_ingestion

# Trigger comprehensive learning
results = await comprehensive_ingestion.ingest_all_sources()
```

### Data Sources Analyzed

#### 1. Codebase Analysis
```python
# Extract patterns from Tower services
codebase_patterns = await analyze_codebase_patterns()
```
- Framework preferences (Vue.js, FastAPI, PostgreSQL)
- File organization patterns (src/, api/, services/)
- Architecture patterns (microservices, reverse proxy)

#### 2. Database Mining
```python
# Mine conversation and learning history
database_patterns = await mine_database_knowledge()
```
- Recent conversation patterns (30 days)
- Existing learning history (confidence > 0.7)
- Communication style analysis
- Technical preference tracking

#### 3. Documentation Processing
```python
# Process KB articles and README files
documentation_patterns = await process_documentation()
```
- Knowledge base articles via API
- README files across /opt/ services
- System knowledge extraction

#### 4. System Configuration Analysis
```python
# Analyze nginx and systemd configurations
system_patterns = await analyze_system_configs()
```
- nginx reverse proxy patterns
- SSL/security configuration patterns
- systemd service deployment patterns

#### 5. Git History Analysis
```python
# Extract development patterns from commits
git_patterns = await analyze_git_history()
```
- Commit message style analysis
- Refactoring frequency patterns
- Development workflow insights

### Triggering Learning Pipeline

**API Endpoint**: `POST /api/echo/learning/comprehensive-ingest`

```bash
curl -X POST http://localhost:8309/api/echo/learning/comprehensive-ingest \
  -H "Content-Type: application/json" \
  -d '{"include_sources": ["codebase", "database", "documentation", "system", "git"]}'
```

**Programmatic Usage**:
```python
from learning.comprehensive_data_ingestion import ComprehensiveDataIngestion

ingestion = ComprehensiveDataIngestion()
results = await ingestion.ingest_all_sources()
print(f"Extracted {results['patterns_extracted']} patterns")
```

## Business Logic Components

### Integration Pattern

The components work together in a coordinated flow:

```python
class ConversationManager:
    def __init__(self):
        # Initialize business logic components
        self.pattern_matcher = BusinessLogicPatternMatcher()
        self.logic_applicator = BusinessLogicApplicator()

    def apply_business_logic(self, query: str, base_response: str) -> str:
        # 1. Find relevant patterns
        raw_patterns = self.pattern_matcher.get_relevant_patterns(query)

        # 2. Transform for application
        patterns = self.pattern_matcher.transform_patterns_for_application(raw_patterns)

        # 3. Apply to response
        return self.logic_applicator.apply_patterns_to_response(query, base_response, patterns)
```

### Pattern Storage Format

Patterns are stored in the `learning_history` table:

```sql
CREATE TABLE learning_history (
    id SERIAL PRIMARY KEY,
    fact_type TEXT NOT NULL,              -- 'technical_stack_preferences', 'quality_standards', etc.
    learned_fact TEXT NOT NULL,           -- The actual pattern/preference
    confidence FLOAT DEFAULT 0.5,        -- 0.0 - 1.0 confidence score
    metadata JSONB,                       -- Source info, extraction details
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(fact_type, learned_fact)
);
```

### Example Pattern Entry

```json
{
  "fact_type": "technical_stack_preferences",
  "learned_fact": "Patrick prefers PostgreSQL over MySQL for database projects",
  "confidence": 0.9,
  "metadata": {
    "source": "/opt/tower-auth/requirements.txt",
    "extraction_source": "codebase_analysis",
    "extraction_timestamp": "2025-12-09T10:30:00"
  }
}
```

## Development Best Practices

### 1. Component Initialization

Always initialize business logic components properly:

```python
class YourService:
    def __init__(self):
        # Initialize pattern matcher
        self.pattern_matcher = BusinessLogicPatternMatcher()
        self.logic_applicator = BusinessLogicApplicator()

        # Check initialization
        stats = self.pattern_matcher.get_pattern_stats()
        logger.info(f"Loaded {stats['total_patterns']} patterns")
```

### 2. Error Handling

Implement graceful fallbacks for pattern application:

```python
def apply_patterns_safely(query: str, response: str) -> str:
    try:
        return apply_business_logic_to_response(query, response)
    except Exception as e:
        logger.error(f"Pattern application failed: {e}")
        return response  # Return original response on error
```

### 3. Performance Monitoring

Track pattern application performance:

```python
# Get application statistics
applicator_stats = self.logic_applicator.get_application_stats()
matcher_stats = self.pattern_matcher.get_pattern_stats()

# Log performance metrics
logger.info(f"Success rate: {applicator_stats['success_rate']:.1f}%")
logger.info(f"Patterns loaded: {matcher_stats['total_patterns']}")
```

### 4. Async Operations

Use async/await for learning pipeline operations:

```python
async def trigger_learning():
    # Use async for heavy operations
    results = await comprehensive_ingestion.ingest_all_sources()

    # Process results
    for source, patterns in results.items():
        logger.info(f"{source}: {len(patterns)} patterns")
```

## Testing Framework

### Unit Testing Business Logic Components

**Location**: `/opt/tower-echo-brain/tests/services/test_business_logic.py`

```python
import pytest
from services.business_logic_matcher import BusinessLogicPatternMatcher
from services.business_logic_applicator import BusinessLogicApplicator

class TestBusinessLogicMatcher:
    def test_pattern_matching(self):
        matcher = BusinessLogicPatternMatcher()

        # Test PostgreSQL preference detection
        query = "Should I use a database for this project?"
        patterns = matcher.get_relevant_patterns(query)

        assert len(patterns) > 0
        assert any('postgresql' in p.get('learned_fact', '').lower()
                  for p in patterns)

    def test_pattern_transformation(self):
        matcher = BusinessLogicPatternMatcher()

        raw_patterns = [
            {
                'fact_type': 'technical_stack_preferences',
                'learned_fact': 'Patrick prefers PostgreSQL',
                'confidence': 0.9,
                'metadata': {}
            }
        ]

        transformed = matcher.transform_patterns_for_application(raw_patterns)

        assert transformed[0]['type'] == 'preference'
        assert transformed[0]['trigger'] == 'database'

class TestBusinessLogicApplicator:
    def test_preference_application(self):
        applicator = BusinessLogicApplicator()

        query = "What database should I use?"
        base_response = "You could use any database."
        patterns = [
            {
                'type': 'preference',
                'trigger': 'database',
                'business_logic': 'Patrick prefers PostgreSQL for reliability',
                'confidence': 0.9
            }
        ]

        result = applicator.apply_patterns_to_response(query, base_response, patterns)

        assert 'PostgreSQL' in result
        assert result != base_response
```

### Integration Testing

Test the complete pattern application flow:

```python
class TestBusinessLogicIntegration:
    def test_end_to_end_pattern_application(self):
        # Setup components
        matcher = BusinessLogicPatternMatcher()
        applicator = BusinessLogicApplicator()

        # Test complete flow
        query = "What frontend framework should I use?"
        base_response = "Any frontend framework works."

        # Get patterns
        raw_patterns = matcher.get_relevant_patterns(query)
        transformed_patterns = matcher.transform_patterns_for_application(raw_patterns)

        # Apply patterns
        enhanced_response = applicator.apply_patterns_to_response(
            query, base_response, transformed_patterns
        )

        # Verify enhancement
        assert enhanced_response != base_response
```

### Learning Pipeline Testing

```python
class TestLearningPipeline:
    @pytest.mark.asyncio
    async def test_comprehensive_ingestion(self):
        from learning.comprehensive_data_ingestion import ComprehensiveDataIngestion

        ingestion = ComprehensiveDataIngestion()

        # Test individual components
        codebase_patterns = await ingestion.analyze_codebase_patterns()
        assert len(codebase_patterns) > 0

        database_patterns = await ingestion.mine_database_knowledge()
        assert len(database_patterns) >= 0  # May be empty in test env

    def test_pattern_consolidation(self):
        ingestion = ComprehensiveDataIngestion()

        # Test duplicate removal
        results = {
            'source1': [
                {'pattern': 'Patrick uses PostgreSQL', 'type': 'tech', 'confidence': 0.8}
            ],
            'source2': [
                {'pattern': 'Patrick uses PostgreSQL for databases', 'type': 'tech', 'confidence': 0.9}
            ]
        }

        consolidated = ingestion.consolidate_patterns(results)

        # Should remove near-duplicates
        assert len(consolidated) == 1
```

### Running Tests

```bash
# Run all business logic tests
cd /opt/tower-echo-brain
source venv/bin/activate
pytest tests/services/test_business_logic.py -v

# Run learning pipeline tests
pytest tests/learning/test_comprehensive_ingestion.py -v

# Run integration tests
pytest tests/integration/test_business_logic_integration.py -v
```

## Performance Monitoring

### Metrics Collection

Monitor pattern application performance:

```python
from api.business_logic_middleware import business_logic_middleware

# Get middleware performance stats
stats = business_logic_middleware.get_middleware_stats()

metrics = {
    'total_requests': stats['total_requests'],
    'patterns_applied': stats['patterns_applied'],
    'application_rate': stats['application_rate'],
    'error_rate': stats['error_rate'],
    'success_rate': stats['success_rate']
}
```

### Pattern Effectiveness Tracking

Track which patterns are most effective:

```python
class PatternEffectivenessTracker:
    def track_pattern_usage(self, pattern_id: str, query: str,
                          user_satisfaction: float):
        """Track how effective patterns are in real usage"""
        # Store effectiveness metrics
        cursor.execute("""
            INSERT INTO pattern_effectiveness
            (pattern_id, query_type, satisfaction_score, timestamp)
            VALUES (%s, %s, %s, %s)
        """, (pattern_id, self.classify_query(query),
              user_satisfaction, datetime.now()))

    def get_pattern_effectiveness(self, pattern_id: str) -> float:
        """Get effectiveness score for a pattern"""
        cursor.execute("""
            SELECT AVG(satisfaction_score)
            FROM pattern_effectiveness
            WHERE pattern_id = %s
        """, (pattern_id,))
        return cursor.fetchone()[0] or 0.5
```

### Learning Pipeline Metrics

Monitor comprehensive learning performance:

```python
async def monitor_learning_pipeline():
    """Monitor learning pipeline performance"""
    start_time = datetime.now()

    results = await comprehensive_ingestion.ingest_all_sources()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    metrics = {
        'duration_seconds': duration,
        'patterns_extracted': results['patterns_extracted'],
        'sources_processed': results['sources_processed'],
        'extraction_rate': results['patterns_extracted'] / duration,
        'pattern_breakdown': results['pattern_breakdown']
    }

    # Log performance metrics
    logger.info(f"Learning pipeline completed in {duration:.1f}s")
    logger.info(f"Extracted {metrics['patterns_extracted']} patterns")
    logger.info(f"Rate: {metrics['extraction_rate']:.1f} patterns/second")

    return metrics
```

### Real-time Monitoring Dashboard

**API Endpoint**: `GET /api/echo/monitoring/business-logic`

```json
{
  "pattern_matcher_stats": {
    "total_patterns": 45,
    "high_confidence": 12,
    "by_type": {
      "technical_stack_preferences": 15,
      "quality_standards": 8,
      "communication_patterns": 12,
      "project_priorities": 6,
      "naming_standards": 4
    }
  },
  "applicator_stats": {
    "total_applications": 234,
    "successful_applications": 221,
    "failed_applications": 13,
    "success_rate": 94.4
  },
  "middleware_stats": {
    "total_requests": 1567,
    "patterns_applied": 432,
    "application_rate": 27.6,
    "error_rate": 0.8,
    "success_rate": 99.2
  }
}
```

## Code Quality Standards

### TypeScript Pattern Compliance

While Echo Brain is primarily Python, follow TypeScript-inspired patterns:

#### 1. Interface-like Typing

```python
from typing import Protocol, Dict, List, Optional

class PatternMatcher(Protocol):
    """Interface definition for pattern matchers"""
    def get_relevant_patterns(self, query: str) -> List[Dict]: ...
    def transform_patterns_for_application(self, patterns: List[Dict]) -> List[Dict]: ...

class BusinessLogicPatternMatcher(PatternMatcher):
    """Concrete implementation of PatternMatcher protocol"""
    pass
```

#### 2. Strict Type Annotations

```python
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

@dataclass
class Pattern:
    type: str
    trigger: str
    business_logic: str
    confidence: float
    metadata: Dict[str, Any]

class BusinessLogicApplicator:
    def apply_patterns_to_response(
        self,
        query: str,
        base_response: str,
        patterns: List[Pattern]
    ) -> str:
        """Apply patterns with strict typing"""
        # Implementation with type safety
```

#### 3. Error Handling Patterns

```python
from typing import Union, Optional
from dataclasses import dataclass

@dataclass
class PatternApplicationResult:
    success: bool
    enhanced_response: Optional[str]
    error_message: Optional[str]
    patterns_applied: int

class BusinessLogicApplicator:
    def apply_patterns_safely(
        self,
        query: str,
        base_response: str,
        patterns: List[Dict]
    ) -> PatternApplicationResult:
        """Safe pattern application with error handling"""
        try:
            enhanced = self.apply_patterns_to_response(query, base_response, patterns)
            return PatternApplicationResult(
                success=True,
                enhanced_response=enhanced,
                error_message=None,
                patterns_applied=len(patterns)
            )
        except Exception as e:
            return PatternApplicationResult(
                success=False,
                enhanced_response=None,
                error_message=str(e),
                patterns_applied=0
            )
```

### Async/Await Best Practices

```python
import asyncio
from typing import List, Dict
import logging

class AsyncBusinessLogicManager:
    async def process_patterns_concurrently(
        self,
        queries: List[str]
    ) -> List[Dict]:
        """Process multiple queries concurrently"""
        tasks = [
            self.get_patterns_for_query(query)
            for query in queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions properly
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Query {queries[i]} failed: {result}")
                processed_results.append([])
            else:
                processed_results.append(result)

        return processed_results
```

### Database Interaction Patterns

```python
import asyncpg
from contextlib import asynccontextmanager
from typing import List, Dict, AsyncContextManager

class DatabaseManager:
    @asynccontextmanager
    async def get_connection(self) -> AsyncContextManager[asyncpg.Connection]:
        """Proper async database connection management"""
        conn = await asyncpg.connect(**self.db_config)
        try:
            yield conn
        finally:
            await conn.close()

    async def fetch_patterns_safely(
        self,
        query: str
    ) -> List[Dict]:
        """Safe async database queries with proper error handling"""
        async with self.get_connection() as conn:
            try:
                rows = await conn.fetch("""
                    SELECT fact_type, learned_fact, confidence, metadata
                    FROM learning_history
                    WHERE metadata->>'source' = 'patrick_conversation_analysis'
                    AND confidence > 0.5
                    ORDER BY confidence DESC
                """)

                return [dict(row) for row in rows]
            except Exception as e:
                logging.error(f"Database query failed: {e}")
                return []
```

## Pattern Development Guide

### Creating New Pattern Types

1. **Define Pattern Schema**:

```python
@dataclass
class NewPatternType:
    type: str = "custom_pattern_type"
    triggers: List[str] = field(default_factory=list)
    application_rules: Dict[str, Any] = field(default_factory=dict)
    confidence_threshold: float = 0.7
```

2. **Add to Pattern Matcher**:

```python
class BusinessLogicPatternMatcher:
    def _pattern_matches_query(self, pattern: Dict, query_lower: str) -> bool:
        fact_type = pattern.get('fact_type', '')

        # Add new pattern type
        if fact_type == 'custom_pattern_type':
            custom_keywords = ['keyword1', 'keyword2', 'keyword3']
            if any(keyword in query_lower for keyword in custom_keywords):
                return True

        return False
```

3. **Add Application Logic**:

```python
class BusinessLogicApplicator:
    def _apply_custom_patterns(
        self,
        query: str,
        response: str,
        patterns: List[Dict]
    ) -> str:
        """Apply custom pattern logic"""
        for pattern in patterns:
            custom_logic = pattern.get('business_logic', '')

            # Apply custom transformation
            if self._should_apply_custom_pattern(query, custom_logic):
                response = self._transform_response_with_custom_logic(response, custom_logic)

        return response
```

### Testing New Patterns

```python
def test_new_pattern_type():
    # Setup
    matcher = BusinessLogicPatternMatcher()
    applicator = BusinessLogicApplicator()

    # Create test pattern
    test_pattern = {
        'fact_type': 'custom_pattern_type',
        'learned_fact': 'Custom pattern behavior description',
        'confidence': 0.8
    }

    # Test matching
    query = "query that should trigger custom pattern"
    assert matcher._pattern_matches_query(test_pattern, query.lower())

    # Test application
    base_response = "base response"
    enhanced = applicator._apply_custom_patterns(query, base_response, [test_pattern])

    assert enhanced != base_response
```

### Feedback Integration

```python
class PatternFeedbackIntegrator:
    async def process_user_feedback(
        self,
        query: str,
        response: str,
        patterns_applied: List[Dict],
        user_satisfaction: float
    ):
        """Process user feedback to improve patterns"""

        # Update pattern effectiveness scores
        for pattern in patterns_applied:
            await self._update_pattern_effectiveness(
                pattern['id'],
                user_satisfaction
            )

        # If satisfaction is low, create improvement task
        if user_satisfaction < 0.6:
            await self._create_pattern_improvement_task(
                query, response, patterns_applied
            )

    async def _create_pattern_improvement_task(
        self,
        query: str,
        response: str,
        patterns: List[Dict]
    ):
        """Create task to improve poorly performing patterns"""
        task = {
            'type': 'PATTERN_IMPROVEMENT',
            'query': query,
            'response': response,
            'patterns': patterns,
            'priority': 'NORMAL',
            'created_at': datetime.now()
        }

        # Add to task queue for processing
        await self.task_queue.add_task(task)
```

## API Documentation

### Business Logic Endpoints

#### Get Pattern Statistics
```
GET /api/echo/business-logic/patterns/stats

Response:
{
  "total_patterns": 45,
  "by_type": {
    "technical_stack_preferences": 15,
    "quality_standards": 8
  },
  "high_confidence": 12,
  "cache_status": "fresh"
}
```

#### Apply Business Logic to Text
```
POST /api/echo/business-logic/apply

Request:
{
  "query": "What database should I use?",
  "response": "Any database works.",
  "endpoint_type": "query"
}

Response:
{
  "enhanced_response": "Patrick prefers PostgreSQL for reliability. Any database works.",
  "patterns_applied": 1,
  "processing_time_ms": 45
}
```

### Learning Pipeline Endpoints

#### Trigger Comprehensive Learning
```
POST /api/echo/learning/comprehensive-ingest

Request:
{
  "include_sources": ["codebase", "database", "documentation"],
  "max_patterns": 1000
}

Response:
{
  "status": "completed",
  "patterns_extracted": 67,
  "sources_processed": 3,
  "pattern_breakdown": {
    "codebase_patterns": 25,
    "database_patterns": 30,
    "documentation_patterns": 12
  },
  "processing_time_seconds": 45.6
}
```

#### Get Learning Status
```
GET /api/echo/learning/status

Response:
{
  "last_learning_run": "2025-12-09T10:30:00Z",
  "patterns_in_database": 156,
  "learning_pipeline_status": "idle",
  "next_scheduled_run": "2025-12-09T22:00:00Z"
}
```

### Monitoring Endpoints

#### Get Business Logic Performance
```
GET /api/echo/monitoring/business-logic

Response:
{
  "pattern_matcher_stats": {...},
  "applicator_stats": {...},
  "middleware_stats": {...},
  "uptime_seconds": 345600
}
```

---

## Conclusion

Echo Brain's learning architecture provides a robust foundation for continuous improvement through comprehensive data ingestion and intelligent pattern application. The separation of concerns design ensures maintainability while the extensive monitoring capabilities provide visibility into system performance.

For additional support or questions, refer to the API documentation or contact the development team.

**Last Updated**: 2025-12-09
**Version**: 2.0.0
**Reviewer**: Claude Code Expert System
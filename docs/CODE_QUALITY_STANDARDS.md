# Echo Brain Code Quality Standards

## Overview

This document establishes code quality standards for Echo Brain development, with special focus on TypeScript-inspired patterns applied to Python development. These standards ensure maintainability, type safety, and architectural consistency.

## Table of Contents

1. [TypeScript-Inspired Python Patterns](#typescript-inspired-python-patterns)
2. [Type Safety Standards](#type-safety-standards)
3. [Async/Await Best Practices](#asyncawait-best-practices)
4. [Error Handling Patterns](#error-handling-patterns)
5. [Database Interaction Standards](#database-interaction-standards)
6. [Interface and Protocol Usage](#interface-and-protocol-usage)
7. [Testing Standards](#testing-standards)
8. [Code Review Checklist](#code-review-checklist)

## TypeScript-Inspired Python Patterns

### 1. Interface-like Protocols

Replace abstract base classes with typing.Protocol for interface definitions:

**✅ Good - Protocol-based Interface**:
```python
from typing import Protocol, Dict, List, Optional, runtime_checkable

@runtime_checkable
class BusinessLogicProcessor(Protocol):
    """Interface for business logic processing components"""

    def get_relevant_patterns(self, query: str) -> List[Dict[str, Any]]: ...

    def apply_patterns(
        self,
        query: str,
        response: str,
        patterns: List[Dict[str, Any]]
    ) -> str: ...

    def get_processing_stats(self) -> Dict[str, int]: ...

# Implementation
class BusinessLogicPatternMatcher(BusinessLogicProcessor):
    def get_relevant_patterns(self, query: str) -> List[Dict[str, Any]]:
        # Implementation
        return []

    def apply_patterns(
        self,
        query: str,
        response: str,
        patterns: List[Dict[str, Any]]
    ) -> str:
        # Implementation
        return response

    def get_processing_stats(self) -> Dict[str, int]:
        return {"patterns_processed": 0}
```

**❌ Bad - Abstract Base Class**:
```python
from abc import ABC, abstractmethod

class BusinessLogicProcessor(ABC):  # Avoid heavy inheritance
    @abstractmethod
    def process(self, data):  # No type hints
        pass
```

### 2. Discriminated Unions with Literal Types

Use Literal types and Union for discriminated unions:

**✅ Good - Discriminated Union**:
```python
from typing import Union, Literal, Dict, Any
from dataclasses import dataclass

@dataclass
class SuccessResult:
    status: Literal["success"]
    data: Dict[str, Any]
    patterns_applied: int

@dataclass
class ErrorResult:
    status: Literal["error"]
    error_message: str
    error_code: str

ProcessingResult = Union[SuccessResult, ErrorResult]

def process_business_logic(query: str) -> ProcessingResult:
    try:
        # Processing logic
        return SuccessResult(
            status="success",
            data={"response": "Enhanced response"},
            patterns_applied=3
        )
    except Exception as e:
        return ErrorResult(
            status="error",
            error_message=str(e),
            error_code="PROCESSING_FAILED"
        )

# Type-safe usage
result = process_business_logic("query")
if result.status == "success":
    # TypeScript-like narrowing - result is now SuccessResult
    print(f"Applied {result.patterns_applied} patterns")
elif result.status == "error":
    # result is now ErrorResult
    print(f"Error: {result.error_message}")
```

### 3. Generic Types and Constraints

Use TypeVar for generic programming:

**✅ Good - Generic Types**:
```python
from typing import TypeVar, Generic, Protocol, List, Optional

T = TypeVar('T')
ProcessorType = TypeVar('ProcessorType', bound='DataProcessor')

class DataProcessor(Protocol):
    def process(self, data: T) -> T: ...

class Repository(Generic[T]):
    """Generic repository pattern"""

    def __init__(self, model_class: type[T]):
        self.model_class = model_class
        self._items: List[T] = []

    def add(self, item: T) -> T:
        self._items.append(item)
        return item

    def find_by_id(self, item_id: str) -> Optional[T]:
        # Implementation
        return None

    def get_all(self) -> List[T]:
        return self._items.copy()

# Usage with specific types
@dataclass
class Pattern:
    id: str
    type: str
    content: str

pattern_repo = Repository[Pattern](Pattern)
pattern = pattern_repo.add(Pattern("1", "preference", "PostgreSQL"))
```

## Type Safety Standards

### 1. Comprehensive Type Annotations

All public functions must have complete type annotations:

**✅ Good - Complete Annotations**:
```python
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

class PatternEffectivenessTracker:
    def __init__(self, db_config: Dict[str, str]) -> None:
        self.db_config = db_config
        self.effectiveness_cache: Dict[str, float] = {}

    async def track_pattern_usage(
        self,
        pattern_id: str,
        query: str,
        user_satisfaction: float,
        timestamp: Optional[datetime] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Track pattern effectiveness with type safety.

        Args:
            pattern_id: Unique identifier for the pattern
            query: User query that triggered the pattern
            user_satisfaction: Score from 0.0 to 1.0
            timestamp: Optional timestamp, defaults to now

        Returns:
            Tuple of (success, error_message)
        """
        if not 0.0 <= user_satisfaction <= 1.0:
            return False, "Satisfaction score must be between 0.0 and 1.0"

        if timestamp is None:
            timestamp = datetime.now()

        try:
            # Database operation
            await self._store_effectiveness_data(
                pattern_id, query, user_satisfaction, timestamp
            )
            return True, None
        except Exception as e:
            return False, str(e)

    async def _store_effectiveness_data(
        self,
        pattern_id: str,
        query: str,
        satisfaction: float,
        timestamp: datetime
    ) -> None:
        # Implementation
        pass
```

**❌ Bad - Missing Annotations**:
```python
def track_pattern_usage(self, pattern_id, query, satisfaction):  # No types
    # Implementation without type safety
    pass
```

### 2. Dataclass Usage for Structured Data

Use dataclasses instead of dictionaries for structured data:

**✅ Good - Dataclass Structure**:
```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class PatternMetadata:
    """Structured pattern metadata"""
    source: str
    extraction_source: str
    extraction_timestamp: datetime
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    validation_history: List[str] = field(default_factory=list)

@dataclass
class BusinessPattern:
    """Business logic pattern with type safety"""
    id: str
    fact_type: str
    learned_fact: str
    confidence: float
    metadata: PatternMetadata
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validation after initialization"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

        if not self.fact_type:
            raise ValueError("Fact type cannot be empty")

# Type-safe usage
pattern = BusinessPattern(
    id="pattern_1",
    fact_type="technical_stack_preferences",
    learned_fact="Patrick prefers PostgreSQL",
    confidence=0.9,
    metadata=PatternMetadata(
        source="/opt/tower-auth/requirements.txt",
        extraction_source="codebase_analysis",
        extraction_timestamp=datetime.now()
    )
)
```

**❌ Bad - Dictionary Usage**:
```python
# Untyped dictionary - prone to errors
pattern = {
    "id": "pattern_1",
    "type": "tech_pref",  # Inconsistent key naming
    "fact": "PostgreSQL",  # Missing context
    "conf": 0.9,  # Abbreviated key
    # Missing required fields
}
```

### 3. Optional vs Required Fields

Be explicit about optional fields:

**✅ Good - Clear Optional/Required**:
```python
from typing import Optional

@dataclass
class LearningConfiguration:
    """Learning pipeline configuration with clear optionals"""
    # Required fields
    max_patterns: int
    confidence_threshold: float
    enabled_sources: List[str]

    # Optional fields with defaults
    cache_duration_minutes: int = 5
    pattern_validation_enabled: bool = True
    debug_mode: bool = False

    # Optional fields that can be None
    custom_extractor: Optional[str] = None
    notification_email: Optional[str] = None
```

## Async/Await Best Practices

### 1. Proper Async Context Management

Use async context managers for resource management:

**✅ Good - Async Context Management**:
```python
import asyncpg
from contextlib import asynccontextmanager
from typing import AsyncContextManager, Dict, Any

class AsyncDatabaseManager:
    def __init__(self, db_config: Dict[str, str]) -> None:
        self.db_config = db_config

    @asynccontextmanager
    async def get_connection(self) -> AsyncContextManager[asyncpg.Connection]:
        """Async context manager for database connections"""
        conn: Optional[asyncpg.Connection] = None
        try:
            conn = await asyncpg.connect(**self.db_config)
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                await conn.close()

    async def execute_with_connection(
        self,
        query: str,
        *args: Any
    ) -> List[Dict[str, Any]]:
        """Execute query with proper connection management"""
        async with self.get_connection() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
```

**❌ Bad - Manual Connection Management**:
```python
async def fetch_data(self):
    conn = await asyncpg.connect(**self.db_config)  # Manual management
    try:
        rows = await conn.fetch("SELECT * FROM table")
        return rows
    except:
        pass  # Swallowing exceptions
    # Missing conn.close() - resource leak
```

### 2. Concurrent Processing with Error Handling

Handle concurrent operations properly:

**✅ Good - Safe Concurrent Processing**:
```python
import asyncio
from typing import List, Union, Dict, Any

class ConcurrentPatternProcessor:
    async def process_queries_concurrently(
        self,
        queries: List[str],
        max_concurrent: int = 5
    ) -> List[Union[Dict[str, Any], Exception]]:
        """Process multiple queries with controlled concurrency"""

        async def process_single_query(query: str) -> Dict[str, Any]:
            """Process a single query with error handling"""
            try:
                patterns = await self.get_patterns_for_query(query)
                return {
                    "query": query,
                    "patterns": patterns,
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"Failed to process query '{query}': {e}")
                raise

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(query: str) -> Union[Dict[str, Any], Exception]:
            async with semaphore:
                try:
                    return await process_single_query(query)
                except Exception as e:
                    return e

        # Process all queries concurrently
        tasks = [process_with_semaphore(query) for query in queries]
        results = await asyncio.gather(*tasks)

        return results
```

## Error Handling Patterns

### 1. Result Type Pattern

Implement Result type for explicit error handling:

**✅ Good - Result Type Pattern**:
```python
from typing import Union, TypeVar, Generic, Callable, Optional
from dataclasses import dataclass

T = TypeVar('T')
E = TypeVar('E', bound=Exception)

@dataclass
class Ok(Generic[T]):
    value: T

@dataclass
class Err(Generic[E]):
    error: E

Result = Union[Ok[T], Err[E]]

class PatternProcessor:
    def process_pattern(
        self,
        pattern_data: str
    ) -> Result[Dict[str, Any], ValueError]:
        """Process pattern with explicit error handling"""
        try:
            if not pattern_data.strip():
                return Err(ValueError("Pattern data cannot be empty"))

            processed_data = self._parse_pattern(pattern_data)
            return Ok(processed_data)

        except ValueError as e:
            return Err(e)
        except Exception as e:
            return Err(ValueError(f"Unexpected error: {str(e)}"))

    def handle_pattern_result(
        self,
        result: Result[Dict[str, Any], ValueError]
    ) -> Optional[Dict[str, Any]]:
        """Handle result with pattern matching"""
        match result:
            case Ok(value):
                logger.info("Pattern processed successfully")
                return value
            case Err(error):
                logger.error(f"Pattern processing failed: {error}")
                return None
```

### 2. Exception Hierarchy

Define clear exception hierarchies:

**✅ Good - Exception Hierarchy**:
```python
class EchoBrainError(Exception):
    """Base exception for Echo Brain system"""
    def __init__(self, message: str, error_code: str = "UNKNOWN"):
        super().__init__(message)
        self.error_code = error_code
        self.message = message

class PatternProcessingError(EchoBrainError):
    """Exceptions related to pattern processing"""
    pass

class PatternNotFoundError(PatternProcessingError):
    """Specific pattern not found"""
    def __init__(self, pattern_id: str):
        super().__init__(
            f"Pattern with ID '{pattern_id}' not found",
            "PATTERN_NOT_FOUND"
        )
        self.pattern_id = pattern_id

class PatternValidationError(PatternProcessingError):
    """Pattern validation failed"""
    def __init__(self, validation_errors: List[str]):
        error_msg = f"Pattern validation failed: {', '.join(validation_errors)}"
        super().__init__(error_msg, "PATTERN_VALIDATION_FAILED")
        self.validation_errors = validation_errors

# Usage with specific error handling
try:
    pattern = await pattern_processor.get_pattern("invalid_id")
except PatternNotFoundError as e:
    logger.warning(f"Pattern not found: {e.pattern_id}")
except PatternValidationError as e:
    logger.error(f"Validation errors: {e.validation_errors}")
except EchoBrainError as e:
    logger.error(f"Echo Brain error {e.error_code}: {e.message}")
```

## Database Interaction Standards

### 1. Repository Pattern Implementation

Implement repository pattern with type safety:

**✅ Good - Typed Repository Pattern**:
```python
from typing import Protocol, List, Optional, TypeVar, Generic
import asyncpg
from dataclasses import dataclass

Entity = TypeVar('Entity')

class Repository(Protocol, Generic[Entity]):
    """Repository interface for data access"""

    async def find_by_id(self, entity_id: str) -> Optional[Entity]: ...
    async def find_all(self) -> List[Entity]: ...
    async def create(self, entity: Entity) -> Entity: ...
    async def update(self, entity: Entity) -> Entity: ...
    async def delete(self, entity_id: str) -> bool: ...

@dataclass
class Pattern:
    id: str
    fact_type: str
    learned_fact: str
    confidence: float

class PatternRepository(Repository[Pattern]):
    """Pattern repository implementation"""

    def __init__(self, db_manager: AsyncDatabaseManager):
        self.db_manager = db_manager

    async def find_by_id(self, pattern_id: str) -> Optional[Pattern]:
        """Find pattern by ID with type safety"""
        query = """
            SELECT id, fact_type, learned_fact, confidence
            FROM learning_history
            WHERE id = $1
        """

        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(query, pattern_id)
            if row:
                return Pattern(
                    id=row['id'],
                    fact_type=row['fact_type'],
                    learned_fact=row['learned_fact'],
                    confidence=float(row['confidence'])
                )
            return None

    async def find_by_type(self, fact_type: str) -> List[Pattern]:
        """Find patterns by type"""
        query = """
            SELECT id, fact_type, learned_fact, confidence
            FROM learning_history
            WHERE fact_type = $1
            ORDER BY confidence DESC
        """

        async with self.db_manager.get_connection() as conn:
            rows = await conn.fetch(query, fact_type)
            return [
                Pattern(
                    id=row['id'],
                    fact_type=row['fact_type'],
                    learned_fact=row['learned_fact'],
                    confidence=float(row['confidence'])
                )
                for row in rows
            ]
```

### 2. Query Builder Pattern

Use query builders for complex queries:

**✅ Good - Type-Safe Query Builder**:
```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class QueryFilter:
    field: str
    operator: str
    value: Any

@dataclass
class QueryBuilder:
    table: str
    select_fields: List[str] = field(default_factory=lambda: ["*"])
    where_filters: List[QueryFilter] = field(default_factory=list)
    order_by: Optional[str] = None
    limit: Optional[int] = None
    offset: Optional[int] = None

    def select(self, *fields: str) -> 'QueryBuilder':
        """Select specific fields"""
        self.select_fields = list(fields)
        return self

    def where(self, field: str, operator: str, value: Any) -> 'QueryBuilder':
        """Add where condition"""
        self.where_filters.append(QueryFilter(field, operator, value))
        return self

    def order(self, field: str, direction: str = "ASC") -> 'QueryBuilder':
        """Add order by clause"""
        self.order_by = f"{field} {direction}"
        return self

    def build(self) -> tuple[str, List[Any]]:
        """Build parameterized query"""
        query_parts = [f"SELECT {', '.join(self.select_fields)}"]
        query_parts.append(f"FROM {self.table}")

        params = []
        if self.where_filters:
            where_conditions = []
            for i, filter_condition in enumerate(self.where_filters, 1):
                where_conditions.append(f"{filter_condition.field} {filter_condition.operator} ${i}")
                params.append(filter_condition.value)

            query_parts.append(f"WHERE {' AND '.join(where_conditions)}")

        if self.order_by:
            query_parts.append(f"ORDER BY {self.order_by}")

        if self.limit:
            query_parts.append(f"LIMIT {self.limit}")

        if self.offset:
            query_parts.append(f"OFFSET {self.offset}")

        return " ".join(query_parts), params

# Usage
builder = QueryBuilder("learning_history")
query, params = (
    builder
    .select("fact_type", "learned_fact", "confidence")
    .where("confidence", ">", 0.8)
    .where("fact_type", "=", "technical_stack_preferences")
    .order("confidence", "DESC")
    .build()
)
```

## Interface and Protocol Usage

### 1. Service Interface Definition

Define clear service interfaces:

**✅ Good - Service Interfaces**:
```python
from typing import Protocol, List, Dict, Any, Optional

class LearningPipelineService(Protocol):
    """Interface for learning pipeline services"""

    async def ingest_data_source(
        self,
        source_type: str,
        configuration: Dict[str, Any]
    ) -> Dict[str, Any]: ...

    async def extract_patterns(
        self,
        raw_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]: ...

    async def validate_patterns(
        self,
        patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]: ...

    async def store_patterns(
        self,
        validated_patterns: List[Dict[str, Any]]
    ) -> bool: ...

class PatternApplicationService(Protocol):
    """Interface for pattern application services"""

    def get_relevant_patterns(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]: ...

    def apply_patterns(
        self,
        query: str,
        base_response: str,
        patterns: List[Dict[str, Any]]
    ) -> str: ...

    def track_effectiveness(
        self,
        pattern_id: str,
        application_result: Dict[str, Any]
    ) -> None: ...
```

### 2. Dependency Injection Container

Implement dependency injection with protocols:

**✅ Good - DI Container with Protocols**:
```python
from typing import Dict, Type, TypeVar, Protocol, Any, cast

ServiceProtocol = TypeVar('ServiceProtocol', bound=Protocol)

class DIContainer:
    """Dependency injection container with protocol support"""

    def __init__(self) -> None:
        self._services: Dict[Type[Protocol], Any] = {}
        self._singletons: Dict[Type[Protocol], Any] = {}

    def register(
        self,
        protocol: Type[ServiceProtocol],
        implementation: ServiceProtocol,
        singleton: bool = False
    ) -> None:
        """Register service implementation for protocol"""
        if singleton:
            self._singletons[protocol] = implementation
        else:
            self._services[protocol] = implementation

    def get(self, protocol: Type[ServiceProtocol]) -> ServiceProtocol:
        """Get service instance by protocol"""
        # Check singletons first
        if protocol in self._singletons:
            return cast(ServiceProtocol, self._singletons[protocol])

        # Check regular services
        if protocol in self._services:
            service_class = self._services[protocol]
            # Create new instance
            return cast(ServiceProtocol, service_class())

        raise ValueError(f"Service not registered for protocol: {protocol}")

# Usage
container = DIContainer()
container.register(
    LearningPipelineService,
    ComprehensiveDataIngestion(),
    singleton=True
)

# Type-safe retrieval
learning_service = container.get(LearningPipelineService)
```

## Testing Standards

### 1. Type-Safe Test Fixtures

Create reusable type-safe test fixtures:

**✅ Good - Type-Safe Fixtures**:
```python
import pytest
from typing import Generator, Dict, Any, List
from dataclasses import dataclass

@dataclass
class TestPattern:
    """Test pattern fixture"""
    id: str
    fact_type: str
    learned_fact: str
    confidence: float

@pytest.fixture
def sample_patterns() -> List[TestPattern]:
    """Provide sample patterns for testing"""
    return [
        TestPattern(
            id="pattern_1",
            fact_type="technical_stack_preferences",
            learned_fact="Patrick prefers PostgreSQL",
            confidence=0.9
        ),
        TestPattern(
            id="pattern_2",
            fact_type="quality_standards",
            learned_fact="Require proof before claiming working",
            confidence=0.8
        )
    ]

@pytest.fixture
def pattern_matcher() -> Generator[BusinessLogicPatternMatcher, None, None]:
    """Provide configured pattern matcher"""
    matcher = BusinessLogicPatternMatcher()
    yield matcher
    # Cleanup if needed

class TestBusinessLogicMatcher:
    def test_pattern_retrieval_with_types(
        self,
        pattern_matcher: BusinessLogicPatternMatcher,
        sample_patterns: List[TestPattern]
    ) -> None:
        """Test pattern retrieval with type safety"""
        # Setup test data
        query = "What database should I use?"

        # Test with type safety
        patterns = pattern_matcher.get_relevant_patterns(query)

        # Type-safe assertions
        assert isinstance(patterns, list)
        for pattern in patterns:
            assert isinstance(pattern, dict)
            assert "fact_type" in pattern
            assert "learned_fact" in pattern
            assert "confidence" in pattern
            assert isinstance(pattern["confidence"], (int, float))
```

### 2. Mock Services with Protocols

Create type-safe mocks:

**✅ Good - Protocol-Based Mocks**:
```python
from unittest.mock import Mock
from typing import List, Dict, Any

class MockPatternService:
    """Mock implementation of PatternApplicationService protocol"""

    def __init__(self) -> None:
        self.patterns_retrieved: List[str] = []
        self.patterns_applied: List[str] = []

    def get_relevant_patterns(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Mock pattern retrieval"""
        self.patterns_retrieved.append(query)
        return [
            {
                "id": "mock_pattern_1",
                "fact_type": "technical_stack_preferences",
                "learned_fact": "Mock preference",
                "confidence": 0.8
            }
        ]

    def apply_patterns(
        self,
        query: str,
        base_response: str,
        patterns: List[Dict[str, Any]]
    ) -> str:
        """Mock pattern application"""
        self.patterns_applied.append(query)
        return f"Enhanced: {base_response}"

    def track_effectiveness(
        self,
        pattern_id: str,
        application_result: Dict[str, Any]
    ) -> None:
        """Mock effectiveness tracking"""
        pass

# Usage in tests
def test_with_mock_service() -> None:
    mock_service = MockPatternService()

    # Test interaction
    patterns = mock_service.get_relevant_patterns("test query")
    enhanced = mock_service.apply_patterns("query", "response", patterns)

    # Verify mock interactions
    assert len(mock_service.patterns_retrieved) == 1
    assert len(mock_service.patterns_applied) == 1
    assert enhanced.startswith("Enhanced:")
```

## Code Review Checklist

### Type Safety Review
- [ ] All public functions have complete type annotations
- [ ] Return types are explicitly specified
- [ ] Optional vs required parameters are clearly marked
- [ ] Generic types are properly constrained
- [ ] Protocols are used instead of abstract base classes where appropriate

### Error Handling Review
- [ ] Exceptions are properly typed and documented
- [ ] Error cases are handled explicitly (Result type or proper exception handling)
- [ ] Exception hierarchy is logical and specific
- [ ] Resource cleanup is guaranteed (context managers, try/finally)

### Async Code Review
- [ ] Async context managers are used for resource management
- [ ] Concurrent operations have proper error handling
- [ ] Semaphores or other controls are used to limit concurrency
- [ ] Database connections are properly managed

### Database Code Review
- [ ] Repository pattern is used for data access
- [ ] Parameterized queries are used (no SQL injection)
- [ ] Transactions are properly handled
- [ ] Connection pooling is considered

### Architecture Review
- [ ] Single responsibility principle is followed
- [ ] Dependencies are injected, not hardcoded
- [ ] Interfaces/protocols define contracts clearly
- [ ] Business logic is separated from infrastructure concerns

### Testing Review
- [ ] Type-safe test fixtures are provided
- [ ] Mocks implement proper protocols
- [ ] Edge cases and error conditions are tested
- [ ] Tests are deterministic and isolated

---

## Conclusion

These code quality standards ensure Echo Brain maintains high code quality while leveraging TypeScript-inspired patterns for better maintainability and type safety. All new code should follow these patterns, and existing code should be gradually refactored to comply.

**Last Updated**: 2025-12-09
**Version**: 1.0.0
**Applies To**: Echo Brain v2.0.0+
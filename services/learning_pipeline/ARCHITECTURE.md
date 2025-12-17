# Echo Brain Learning Pipeline Architecture

## Executive Summary

This document outlines a production-ready learning pipeline architecture for Echo Brain that solves current critical issues:

- **Fixed Issues**: Non-existent cron job script, stale vector database (1,780 vectors), incorrect database connections
- **New Capabilities**: Real-time learning from Claude conversations, KB article processing, circuit breaker resilience
- **Architecture**: TDD-first modular design with comprehensive monitoring

## Current Issues Analysis

### Critical Problems Identified
1. **Broken Cron Job**: Points to `/home/patrick/Tower/echo_learning_pipeline.py` (non-existent)
2. **Stale Vector Database**: Qdrant has 1,780 stale vectors with no active updates
3. **Database Connection Issues**: Services trying to connect to ***REMOVED*** instead of localhost
4. **Missing Semantic Memory Service**: tower-semantic-memory (port 8320) not accessible
5. **Table Conflicts**: Two overlapping tables (`conversations` and `echo_unified_interactions`)

### Infrastructure Status
- ✅ **Qdrant Vector DB**: Running on port 6333 with collections `claude_conversations`, `echo_memories`
- ❌ **Echo Brain Database**: PostgreSQL connection timeout to `echo_brain` database
- ❌ **Semantic Memory Service**: Port 8320 not accessible
- ✅ **Claude Conversations**: ~50,000 conversation files in `~/.claude/conversations/`

## Architecture Overview

### 1. File Structure

```
/opt/tower-echo-brain/services/learning_pipeline/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pipeline.py              # Main pipeline orchestrator
│   │   ├── processor.py             # Content processing engine
│   │   ├── embedder.py              # Embedding generation
│   │   └── circuit_breaker.py       # Resilience patterns
│   ├── connectors/
│   │   ├── __init__.py
│   │   ├── database_connector.py    # PostgreSQL interface
│   │   ├── vector_connector.py      # Qdrant interface
│   │   ├── claude_connector.py      # Claude conversation reader
│   │   └── kb_connector.py          # Knowledge base interface
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── conversation_processor.py  # Claude conversation processing
│   │   ├── kb_processor.py           # Knowledge base article processing
│   │   ├── content_extractor.py     # Text extraction utilities
│   │   └── metadata_enricher.py     # Content metadata enhancement
│   ├── models/
│   │   ├── __init__.py
│   │   ├── conversation.py          # Conversation data models
│   │   ├── learning_item.py         # Learning item abstractions
│   │   └── pipeline_state.py       # Pipeline state management
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py              # Configuration management
│   │   └── logging_config.py       # Logging setup
│   └── utils/
│       ├── __init__.py
│       ├── text_processing.py       # Text utilities
│       ├── metrics.py               # Performance monitoring
│       └── health_check.py          # Health monitoring
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_pipeline.py
│   │   ├── test_processors.py
│   │   ├── test_connectors.py
│   │   └── test_models.py
│   ├── integration/
│   │   ├── test_database_integration.py
│   │   ├── test_vector_integration.py
│   │   └── test_end_to_end.py
│   └── fixtures/
│       ├── sample_conversations.json
│       ├── sample_kb_articles.json
│       └── test_config.yaml
├── config/
│   ├── development.yaml             # Dev environment config
│   ├── production.yaml              # Production environment config
│   └── test.yaml                    # Test environment config
├── scripts/
│   ├── run_pipeline.py              # Main execution script
│   ├── health_check.py              # Health monitoring script
│   ├── migrate_database.py          # Database migration utility
│   └── setup_vector_db.py           # Vector database initialization
├── logs/
│   └── .gitkeep
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project configuration
├── docker-compose.yml              # Container orchestration
├── Dockerfile                       # Container definition
└── README.md                        # Quick start guide
```

### 2. Core Architecture Components

#### 2.1 Pipeline Orchestrator (`core/pipeline.py`)

```python
class LearningPipeline:
    """Main orchestrator for the learning pipeline with circuit breaker pattern."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.circuit_breaker = CircuitBreaker()
        self.processors = self._init_processors()
        self.connectors = self._init_connectors()

    async def run_learning_cycle(self) -> PipelineResult:
        """Execute a complete learning cycle with error handling."""

    async def process_new_conversations(self) -> int:
        """Process new Claude conversations since last run."""

    async def process_kb_updates(self) -> int:
        """Process updated knowledge base articles."""

    async def update_vector_database(self, items: List[LearningItem]) -> bool:
        """Update Qdrant with new embeddings."""
```

#### 2.2 Circuit Breaker Pattern (`core/circuit_breaker.py`)

```python
class CircuitBreaker:
    """Implements circuit breaker pattern for external service resilience."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.state = CircuitState.CLOSED

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
```

#### 2.3 Content Processors

**Conversation Processor** (`processors/conversation_processor.py`):
```python
class ConversationProcessor:
    """Processes Claude conversation files into learning items."""

    async def extract_learnings(self, conversation_file: Path) -> List[LearningItem]:
        """Extract meaningful learning content from conversations."""

    def _identify_key_insights(self, content: str) -> List[str]:
        """Use NLP to identify key insights and learnings."""

    def _extract_code_examples(self, content: str) -> List[CodeExample]:
        """Extract and categorize code examples."""
```

**KB Article Processor** (`processors/kb_processor.py`):
```python
class KnowledgeBaseProcessor:
    """Processes knowledge base articles into learning items."""

    async def process_article(self, article_id: str) -> LearningItem:
        """Convert KB article into learning item with embeddings."""

    def _enrich_with_metadata(self, article: Article) -> EnrichedArticle:
        """Add metadata like categories, tags, relationships."""
```

### 3. Data Flow Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Claude          │    │ Knowledge Base   │    │ Echo Brain      │
│ Conversations   │────│ Articles         │────│ Database        │
│ ~/.claude/      │    │ (SQLite/PG)      │    │ (PostgreSQL)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Learning Pipeline                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Content     │  │ Embedding   │  │ Vector Database         │  │
│  │ Processors  │──│ Generator   │──│ (Qdrant)                │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Processed       │    │ Learning         │    │ Performance     │
│ Conversations   │    │ Metrics          │    │ Monitoring      │
│ (Database)      │    │ (Prometheus)     │    │ (Grafana)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 4. Database Schema Design

#### 4.1 Unified Learning Schema

```sql
-- Replace conflicting tables with unified schema
CREATE TABLE learning_conversations (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(255) UNIQUE NOT NULL,
    source_type VARCHAR(50) NOT NULL, -- 'claude', 'kb_article', 'user_input'
    file_path TEXT,
    processed_at TIMESTAMP DEFAULT NOW(),
    last_modified TIMESTAMP NOT NULL,
    content_hash VARCHAR(64) UNIQUE, -- SHA-256 for duplicate detection
    metadata JSONB,
    processing_status VARCHAR(20) DEFAULT 'pending', -- pending, processed, failed
    error_message TEXT,
    vector_id VARCHAR(255), -- Reference to Qdrant vector
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE learning_items (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(255) REFERENCES learning_conversations(conversation_id),
    item_type VARCHAR(50) NOT NULL, -- 'insight', 'code_example', 'solution', 'error_fix'
    title VARCHAR(500),
    content TEXT NOT NULL,
    metadata JSONB,
    importance_score FLOAT DEFAULT 0.0, -- ML-generated importance
    categories TEXT[], -- Categorization tags
    extracted_at TIMESTAMP DEFAULT NOW(),
    vector_embedding_id VARCHAR(255) -- Qdrant vector ID
);

CREATE TABLE pipeline_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) UNIQUE NOT NULL,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'running', -- running, completed, failed
    conversations_processed INTEGER DEFAULT 0,
    articles_processed INTEGER DEFAULT 0,
    vectors_updated INTEGER DEFAULT 0,
    errors_encountered INTEGER DEFAULT 0,
    performance_metrics JSONB
);

-- Indexes for performance
CREATE INDEX idx_learning_conversations_processed_at ON learning_conversations(processed_at);
CREATE INDEX idx_learning_conversations_status ON learning_conversations(processing_status);
CREATE INDEX idx_learning_items_conversation ON learning_items(conversation_id);
CREATE INDEX idx_learning_items_type ON learning_items(item_type);
CREATE INDEX idx_pipeline_runs_status ON pipeline_runs(status);
```

#### 4.2 Vector Database Collections

**Qdrant Collection Structure**:
```python
# claude_conversations collection
{
    "name": "claude_conversations",
    "vectors": {
        "size": 384,  # sentence-transformers/all-MiniLM-L6-v2
        "distance": "Cosine"
    },
    "payload_schema": {
        "conversation_id": "keyword",
        "source_type": "keyword",
        "categories": "keyword[]",
        "importance_score": "float",
        "created_at": "datetime",
        "content_preview": "text"
    }
}
```

### 5. Configuration Management

#### 5.1 Production Configuration (`config/production.yaml`)

```yaml
database:
  host: "localhost"  # Fixed: was ***REMOVED***
  port: 5432
  name: "echo_brain"
  user: "patrick"
  password_env: "ECHO_BRAIN_DB_PASSWORD"
  pool_size: 10
  max_overflow: 20

vector_database:
  host: "localhost"
  port: 6333
  collection_name: "claude_conversations"
  embedding_dimension: 384

semantic_memory:
  host: "localhost"
  port: 8320
  timeout: 30
  fallback_enabled: true
  fallback_model: "sentence-transformers/all-MiniLM-L6-v2"

pipeline:
  batch_size: 100
  max_concurrent_processors: 5
  processing_timeout: 300

circuit_breaker:
  failure_threshold: 5
  reset_timeout: 60
  half_open_max_calls: 3

monitoring:
  prometheus_port: 9090
  metrics_enabled: true
  health_check_interval: 30

logging:
  level: "INFO"
  file: "/opt/tower-echo-brain/logs/learning_pipeline.log"
  max_size: "100MB"
  backup_count: 5

sources:
  claude_conversations:
    path: "/home/patrick/.claude/conversations"
    file_pattern: "*.md"
    watch_for_changes: true

  knowledge_base:
    enabled: false  # Until KB location is determined
    database_path: ""
    table_name: "articles"
```

### 6. Test Strategy (TDD Approach)

#### 6.1 Unit Test Coverage

```python
# tests/unit/test_pipeline.py
class TestLearningPipeline:
    @pytest.fixture
    def mock_config(self):
        return create_test_config()

    @pytest.fixture
    def pipeline(self, mock_config):
        return LearningPipeline(mock_config)

    async def test_pipeline_initialization(self, pipeline):
        """Test pipeline initializes all components correctly."""

    async def test_conversation_processing(self, pipeline):
        """Test processing of Claude conversation files."""

    async def test_vector_database_update(self, pipeline):
        """Test vector database updates with new embeddings."""

    async def test_circuit_breaker_activation(self, pipeline):
        """Test circuit breaker protects against service failures."""
```

#### 6.2 Integration Test Strategy

```python
# tests/integration/test_end_to_end.py
class TestEndToEndPipeline:
    @pytest.mark.integration
    async def test_full_learning_cycle(self):
        """Test complete learning cycle from file to vector database."""

    @pytest.mark.integration
    async def test_database_persistence(self):
        """Test data persistence across pipeline runs."""

    @pytest.mark.integration
    async def test_error_recovery(self):
        """Test pipeline recovery from various failure scenarios."""
```

### 7. Deployment Plan

#### 7.1 Phase 1: Infrastructure Setup (Week 1)
1. **Database Migration**: Create unified schema, migrate existing data
2. **Vector Database**: Initialize Qdrant collections with proper schema
3. **Configuration**: Deploy production configurations
4. **Monitoring**: Set up Prometheus metrics and Grafana dashboards

#### 7.2 Phase 2: Core Pipeline (Week 2)
1. **Core Components**: Implement pipeline orchestrator and circuit breaker
2. **Connectors**: Database and vector database connectors
3. **Basic Processing**: Simple conversation file processing
4. **Unit Tests**: Comprehensive unit test coverage

#### 7.3 Phase 3: Advanced Processing (Week 3)
1. **Content Analysis**: Advanced NLP for insight extraction
2. **Metadata Enrichment**: Automatic categorization and tagging
3. **Performance Optimization**: Batch processing and concurrent execution
4. **Integration Tests**: End-to-end testing

#### 7.4 Phase 4: Production Deployment (Week 4)
1. **Cron Job Replacement**: Replace broken cron with systemd timer
2. **Service Integration**: Connect to existing Echo Brain ecosystem
3. **Monitoring**: Full observability with alerts
4. **Documentation**: Complete API and operational documentation

#### 7.5 Phase 5: Optimization & Scaling (Week 5)
1. **Performance Tuning**: Optimize based on production metrics
2. **Advanced Features**: Real-time processing, webhook triggers
3. **ML Enhancement**: Improve content analysis with custom models
4. **Knowledge Graph**: Build relationships between learning items

### 8. Monitoring & Observability

#### 8.1 Metrics Collection
- **Processing Metrics**: Items processed per minute, error rates
- **Performance Metrics**: Processing latency, database query times
- **Health Metrics**: Service availability, circuit breaker state
- **Business Metrics**: Learning items extracted, vector database growth

#### 8.2 Alerting Strategy
- **Critical**: Pipeline failures, database connectivity issues
- **Warning**: High error rates, slow processing times
- **Info**: Successful runs, performance improvements

### 9. Security Considerations

#### 9.1 Data Protection
- **Encryption**: Encrypt sensitive conversation content at rest
- **Access Control**: Role-based access to learning pipeline data
- **Audit Logging**: Track all data access and modifications

#### 9.2 Service Security
- **API Authentication**: Secure all service-to-service communication
- **Input Validation**: Sanitize all file inputs and database queries
- **Rate Limiting**: Prevent resource exhaustion from automated processing

### 10. Success Metrics

#### 10.1 Performance Targets
- **Processing Speed**: >1000 conversations/hour
- **Accuracy**: >95% successful processing rate
- **Availability**: >99.9% uptime
- **Latency**: <30 seconds end-to-end processing

#### 10.2 Quality Metrics
- **Content Quality**: >90% relevant insights extracted
- **Deduplication**: <5% duplicate content in vector database
- **Categorization**: >85% accurate automatic categorization

## Implementation Priority

1. **IMMEDIATE** (Days 1-3): Fix broken cron job, database connections
2. **CRITICAL** (Week 1): Core pipeline with basic processing
3. **HIGH** (Weeks 2-3): Advanced features and optimization
4. **MEDIUM** (Weeks 4-5): ML enhancement and scaling

This architecture provides a robust, scalable foundation for Echo Brain's learning capabilities while addressing all current critical issues.
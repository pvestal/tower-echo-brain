# AI Assist Board of Directors Framework

The Board of Directors system provides a comprehensive framework for domain expertise evaluation within AI Assist. Each director acts as a specialized expert that can evaluate tasks, provide recommendations, and build consensus across different areas of knowledge.

## Architecture Overview

```
AI Assist Directors/
├── base_director.py        # Abstract base class for all directors
├── director_registry.py    # Central registry and coordination system
├── example_director.py     # Example implementation and template
├── __init__.py            # Module exports and metadata
└── README.md              # This documentation
```

## Core Components

### DirectorBase (Abstract Base Class)

The foundation class that all directors inherit from. Provides:

- **Evaluation Framework**: Abstract `evaluate()` method for task assessment
- **Knowledge Management**: `load_knowledge()` for domain-specific expertise
- **Reasoning Engine**: `generate_reasoning()` for detailed explanations
- **Confidence Scoring**: `calculate_confidence()` for reliability metrics
- **Improvement Suggestions**: `suggest_improvements()` for optimization recommendations
- **Performance Tracking**: Metrics and success rate monitoring

### DirectorRegistry (Coordination System)

Central management system that:

- **Director Management**: Register/unregister specialized directors
- **Task Routing**: Find relevant directors based on expertise matching
- **Consensus Building**: Combine multiple director evaluations
- **Performance Analytics**: Track board-wide metrics and success rates
- **Caching & Optimization**: Cache evaluations and prune history

### Example Implementation

A complete working example (`ExampleDirector`) that demonstrates:

- Proper inheritance from `DirectorBase`
- Domain-specific knowledge loading (code quality expertise)
- Task analysis and factor-based evaluation
- Recommendation generation based on analysis
- Risk identification and effort estimation

## Quick Start

### 1. Basic Usage

```python
from directors import DirectorBase, DirectorRegistry
from directors.example_director import ExampleDirector

# Create registry and directors
registry = DirectorRegistry()
code_director = ExampleDirector()

# Register director
registry.register_director(code_director)

# Evaluate a task
task = {
    "type": "code_review",
    "description": "Review authentication system for security issues"
}

context = {
    "priority": "high",
    "resources": {"time": "moderate", "team_size": 3}
}

# Get evaluation
result = registry.evaluate_task(task, context)
print(f"Assessment: {result['assessment']}")
print(f"Confidence: {result['confidence']}")
```

### 2. Creating a Custom Director

```python
class SecurityDirector(DirectorBase):
    def __init__(self):
        super().__init__(
            name="SecurityDirector",
            expertise="Application security, vulnerability assessment, secure coding",
            version="1.0.0"
        )

    def evaluate(self, task, context):
        # Your evaluation logic here
        return {
            "assessment": "Security evaluation completed",
            "confidence": 0.85,
            "reasoning": "Based on security best practices...",
            "recommendations": ["Enable 2FA", "Use HTTPS"],
            "risk_factors": ["Weak password policy"],
            "estimated_effort": "Medium (5-7 days)"
        }

    def load_knowledge(self):
        return {
            "best_practices": [
                "Implement defense in depth",
                "Use principle of least privilege",
                "Validate all inputs"
            ],
            "anti_patterns": [
                "Storing passwords in plain text",
                "Trusting user input without validation"
            ],
            "risk_factors": [
                "SQL injection vulnerabilities",
                "Cross-site scripting (XSS)"
            ],
            "optimization_strategies": [
                "Use prepared statements",
                "Implement proper session management"
            ]
        }
```

### 3. Registry Management

```python
# Multiple directors
registry = DirectorRegistry(consensus_threshold=0.7)

security_director = SecurityDirector()
performance_director = PerformanceDirector()
architecture_director = ArchitectureDirector()

# Register all directors
registry.register_director(security_director)
registry.register_director(performance_director)
registry.register_director(architecture_director)

# Get board status
board_info = registry.get_available_directors()
performance = registry.get_board_performance()

print(f"Board has {len(board_info)} directors")
print(f"Consensus rate: {performance['board_summary']['consensus_rate']:.1%}")
```

## Key Features

### 1. Expertise Matching
- Automatic director selection based on task keywords
- Configurable maximum directors per task
- Performance-based director scoring

### 2. Consensus Building
- Configurable consensus thresholds
- Confidence variance analysis
- Weighted recommendation consolidation
- Director agreement tracking

### 3. Performance Analytics
- Individual director metrics
- Board-wide success rates
- Response time tracking
- Agreement matrix between directors

### 4. Knowledge Management
- Domain-specific knowledge bases
- Best practices and anti-patterns
- Risk factors and optimization strategies
- Knowledge export and sharing

### 5. Caching & Optimization
- Task evaluation caching
- History pruning
- Response time optimization
- Memory-efficient operations

## Configuration Options

### DirectorRegistry Parameters
```python
registry = DirectorRegistry(
    consensus_threshold=0.6,      # Minimum agreement for consensus (0.0-1.0)
    max_directors_per_task=5      # Maximum directors to consult per task
)
```

### Performance Tuning
```python
# Cache management
cleared_count = registry.clear_cache()

# History management
pruned_count = registry.prune_task_history(days_to_keep=30)

# Board metrics
metrics = registry.get_board_performance()
```

## Evaluation Result Structure

Every evaluation returns a standardized structure:

```python
{
    "status": "success",                    # success/error
    "consensus_achieved": true,             # Whether consensus was reached
    "assessment": "Overall assessment...",   # Main evaluation result
    "confidence": 0.75,                     # Confidence score (0.0-1.0)
    "confidence_variance": 0.12,            # Variance in confidence scores
    "recommendations": [...],               # List of recommendations
    "risk_factors": [...],                  # List of identified risks
    "reasoning": "Detailed reasoning...",    # Explanation of evaluation
    "individual_evaluations": {...},        # Each director's evaluation
    "director_count": 3,                    # Number of directors consulted
    "evaluation_metadata": {
        "directors_consulted": [...],       # List of director names
        "response_time_seconds": 0.245,     # Evaluation response time
        "task_hash": "abc123...",           # Cache key for task
        "timestamp": "2025-09-16T..."       # Evaluation timestamp
    }
}
```

## Best Practices

### 1. Director Design
- **Single Responsibility**: Each director should focus on one domain
- **Clear Expertise**: Define specific areas of knowledge
- **Consistent Evaluation**: Use standardized scoring methods
- **Comprehensive Knowledge**: Include best practices, anti-patterns, and risks

### 2. Knowledge Base Structure
```python
def load_knowledge(self):
    return {
        "best_practices": [        # Recommended approaches
            "Clear, actionable practices"
        ],
        "anti_patterns": [         # Patterns to avoid
            "Common mistakes and pitfalls"
        ],
        "risk_factors": [          # Potential problems
            "Specific risks to watch for"
        ],
        "optimization_strategies": [ # Performance improvements
            "Ways to optimize and improve"
        ]
    }
```

### 3. Evaluation Implementation
- **Factor-based Analysis**: Break down evaluation into measurable factors
- **Confidence Calculation**: Use objective criteria for confidence scoring
- **Detailed Reasoning**: Provide clear explanations for decisions
- **Actionable Recommendations**: Give specific, implementable suggestions

### 4. Error Handling
```python
def evaluate(self, task, context):
    try:
        # Evaluation logic
        return evaluation_result
    except Exception as e:
        logger.error(f"Evaluation error in {self.name}: {str(e)}")
        return {
            "assessment": "Evaluation failed",
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}",
            "recommendations": [],
            "risk_factors": ["System error occurred"]
        }
```

## Integration with AI Assist

The Directors framework integrates with AI Assist's unified service through:

### 1. Task Evaluation Endpoint
```python
# In echo_unified_service.py
from directors import DirectorRegistry
from directors.specialized import SecurityDirector, PerformanceDirector

# Initialize board
board = DirectorRegistry()
board.register_director(SecurityDirector())
board.register_director(PerformanceDirector())

# Evaluation endpoint
@app.post("/api/directors/evaluate")
async def evaluate_with_directors(task_data: dict):
    result = board.evaluate_task(task_data["task"], task_data.get("context", {}))
    return result
```

### 2. Director Management Endpoints
```python
@app.get("/api/directors")
async def get_directors():
    return board.get_available_directors()

@app.get("/api/directors/performance")
async def get_board_performance():
    return board.get_board_performance()
```

## Testing

### Run Example Director
```bash
cd /opt/tower-echo-brain
python3 directors/example_director.py
```

### Test Framework Integration
```python
from directors import DirectorRegistry
from directors.example_director import ExampleDirector

# Test basic functionality
registry = DirectorRegistry()
director = ExampleDirector()
registry.register_director(director)

task = {"type": "test", "description": "Test evaluation"}
result = registry.evaluate_task(task)
assert result["status"] == "success"
```

## Extending the Framework

### 1. Create Specialized Directors

Create new directors for specific domains:
- `SecurityDirector` - Security and vulnerability assessment
- `PerformanceDirector` - Performance optimization and scalability
- `ArchitectureDirector` - System architecture and design patterns
- `DatabaseDirector` - Database design and optimization
- `UIDirector` - User interface and experience design

### 2. Enhance Consensus Building

Implement advanced consensus algorithms:
- Weighted voting based on director expertise
- Machine learning for director reliability scoring
- Domain-specific confidence adjustments

### 3. Add Learning Capabilities

Implement feedback loops:
- Track recommendation success rates
- Learn from past evaluations
- Adjust confidence scoring based on outcomes

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're in the correct directory
2. **No Directors Found**: Check director registration
3. **Low Consensus**: Adjust consensus threshold or add more directors
4. **Performance Issues**: Clear cache and prune history

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging for debugging
```

## Future Enhancements

1. **Specialized Directors**: Implement domain-specific directors
2. **ML Integration**: Add machine learning for pattern recognition
3. **Web Interface**: Create management dashboard
4. **API Integration**: Connect with external expertise services
5. **Collaborative Learning**: Share knowledge between directors

## License

Proprietary - Tower System - AI Assist Board of Directors Framework
Copyright 2025 Tower System

---

*This framework provides the foundation for building a comprehensive board of directors system that can evaluate any task with domain expertise, build consensus, and provide actionable recommendations.*
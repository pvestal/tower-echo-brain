"""
Configuration management for the learning pipeline.
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = "localhost"
    port: int = 5432
    name: str = "echo_brain"
    user: str = "patrick"
    password_env: str = "ECHO_BRAIN_DB_PASSWORD"
    pool_size: int = 10
    max_overflow: int = 20
    connection_timeout: int = 30
    command_timeout: int = 300

    @property
    def password(self) -> str:
        """Get password from environment variable."""
        return os.getenv(self.password_env, "")

    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def async_connection_string(self) -> str:
        """Get async PostgreSQL connection string."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class VectorDatabaseConfig:
    """Vector database (Qdrant) configuration."""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "claude_conversations"
    embedding_dimension: int = 384
    batch_size: int = 100
    timeout: int = 60


@dataclass
class SemanticMemoryConfig:
    """Semantic memory service configuration."""
    host: str = "localhost"
    port: int = 8320
    timeout: int = 30
    fallback_enabled: bool = True
    fallback_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class PipelineProcessingConfig:
    """Pipeline processing configuration."""
    batch_size: int = 100
    max_concurrent_processors: int = 5
    processing_timeout: int = 300
    min_content_length: int = 50
    max_content_length: int = 50000


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    reset_timeout: int = 60
    half_open_max_calls: int = 3
    timeout: float = 30.0


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration."""
    prometheus_port: int = 9090
    metrics_enabled: bool = True
    health_check_interval: int = 30
    enable_detailed_logging: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: str = "/opt/tower-echo-brain/logs/learning_pipeline.log"
    max_size: str = "100MB"
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class SourceConfig:
    """Data source configuration."""
    path: str
    file_pattern: str = "*.md"
    watch_for_changes: bool = True
    exclude_patterns: List[str] = None
    max_file_age_days: int = 365

    def __post_init__(self):
        if self.exclude_patterns is None:
            self.exclude_patterns = ["**/test_*", "**/.tmp_*"]


@dataclass
class ContentProcessingConfig:
    """Content processing configuration."""
    extract_code_blocks: bool = True
    extract_commands: bool = True
    extract_insights: bool = True
    extract_solutions: bool = True
    min_insight_length: int = 100
    code_block_languages: List[str] = None

    def __post_init__(self):
        if self.code_block_languages is None:
            self.code_block_languages = [
                "python", "bash", "sql", "javascript", "typescript", "yaml", "json"
            ]


@dataclass
class NLPConfig:
    """NLP processing configuration."""
    enable_advanced_processing: bool = True
    category_confidence_threshold: float = 0.7
    importance_threshold: float = 0.5
    duplicate_similarity_threshold: float = 0.95


@dataclass
class SecurityConfig:
    """Security configuration."""
    sanitize_content: bool = True
    exclude_sensitive_patterns: List[str] = None
    max_personal_info_score: float = 0.3

    def __post_init__(self):
        if self.exclude_sensitive_patterns is None:
            self.exclude_sensitive_patterns = [
                "password", "secret", "token", "key", "credential"
            ]


class PipelineConfig:
    """Complete pipeline configuration."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.raw_config = config_dict

        # Initialize all configuration sections
        self.database = DatabaseConfig(**config_dict.get('database', {}))
        self.vector_database = VectorDatabaseConfig(**config_dict.get('vector_database', {}))
        self.semantic_memory = SemanticMemoryConfig(**config_dict.get('semantic_memory', {}))
        self.pipeline = PipelineProcessingConfig(**config_dict.get('pipeline', {}))
        self.circuit_breaker = CircuitBreakerConfig(**config_dict.get('circuit_breaker', {}))
        self.monitoring = MonitoringConfig(**config_dict.get('monitoring', {}))
        self.logging = LoggingConfig(**config_dict.get('logging', {}))
        self.content_processing = ContentProcessingConfig(**config_dict.get('content_processing', {}))
        self.nlp = NLPConfig(**config_dict.get('nlp', {}))
        self.security = SecurityConfig(**config_dict.get('security', {}))

        # Handle sources section
        sources = config_dict.get('sources', {})
        self.claude_conversations = SourceConfig(**sources.get('claude_conversations', {}))

        # Knowledge base configuration (optional)
        kb_config = sources.get('knowledge_base', {})
        self.knowledge_base_enabled = kb_config.get('enabled', False)
        self.knowledge_base_config = kb_config if self.knowledge_base_enabled else {}

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            List of validation error messages
        """
        errors = []

        # Check required database password
        if not self.database.password:
            errors.append(f"Database password not found in environment variable: {self.database.password_env}")

        # Check Claude conversations path
        if not Path(self.claude_conversations.path).exists():
            errors.append(f"Claude conversations directory not found: {self.claude_conversations.path}")

        # Check log file directory
        log_dir = Path(self.logging.file).parent
        if not log_dir.exists():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create log directory {log_dir}: {e}")

        # Validate numeric ranges
        if self.pipeline.batch_size <= 0:
            errors.append("Pipeline batch_size must be greater than 0")

        if self.pipeline.max_concurrent_processors <= 0:
            errors.append("Pipeline max_concurrent_processors must be greater than 0")

        if self.vector_database.embedding_dimension <= 0:
            errors.append("Vector database embedding_dimension must be greater than 0")

        # Validate thresholds
        if not (0.0 <= self.nlp.category_confidence_threshold <= 1.0):
            errors.append("NLP category_confidence_threshold must be between 0.0 and 1.0")

        if not (0.0 <= self.nlp.importance_threshold <= 1.0):
            errors.append("NLP importance_threshold must be between 0.0 and 1.0")

        return errors

    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about the current environment."""
        return {
            'database_password_set': bool(self.database.password),
            'claude_conversations_exists': Path(self.claude_conversations.path).exists(),
            'log_directory_writable': self._check_log_directory_writable(),
            'required_env_vars': {
                'ECHO_BRAIN_DB_PASSWORD': bool(os.getenv('ECHO_BRAIN_DB_PASSWORD')),
                'PYTHONPATH': bool(os.getenv('PYTHONPATH'))
            }
        }

    def _check_log_directory_writable(self) -> bool:
        """Check if log directory is writable."""
        try:
            log_dir = Path(self.logging.file).parent
            test_file = log_dir / '.write_test'
            test_file.touch()
            test_file.unlink()
            return True
        except Exception:
            return False


def load_config(config_path: str) -> PipelineConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Loaded and validated configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
        ValueError: If configuration is invalid
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        if not isinstance(config_dict, dict):
            raise ValueError("Configuration file must contain a YAML dictionary")

        # Create configuration object
        config = PipelineConfig(config_dict)

        # Validate configuration
        validation_errors = config.validate()
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors)
            raise ValueError(error_msg)

        return config

    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}")


def load_config_with_overrides(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> PipelineConfig:
    """
    Load configuration with optional overrides.

    Args:
        config_path: Path to base configuration file
        overrides: Dictionary of configuration overrides

    Returns:
        Configuration with overrides applied
    """
    config = load_config(config_path)

    if overrides:
        # Apply overrides to raw config and recreate
        raw_config = config.raw_config.copy()
        _deep_update(raw_config, overrides)
        config = PipelineConfig(raw_config)

    return config


def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
    """Recursively update nested dictionary."""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
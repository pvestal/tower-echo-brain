"""
Echo Brain Configuration - Limited Autonomy Mode
Strict resource limits and guardrails for safe production deployment
"""

from datetime import datetime

# Deployment mode
DEPLOYMENT_MODE = "LIMITED_AUTONOMY"
DEPLOYMENT_START = datetime.now().isoformat()

# Autonomy Configuration with Strict Limits
AUTONOMY_CONFIG = {
    'mode': 'LIMITED_AUTONOMY',
    'version': '1.0.0',
    'anime_generation': {
        'enabled': True,
        'max_gpu_percent': 50,
        'max_concurrent_jobs': 2,
        'max_daily_generations': 20,
        'output_path': '/opt/tower-echo-brain/data/outputs/validation/',
        'allowed_workflows': ['tier2_svd_template.json'],
        'max_resolution': 512
    },
    'code_analysis': {
        'enabled': True,
        'read_only': True,  # Can analyze but not modify
        'max_files_per_scan': 1000,
        'allowed_paths': ['/opt/tower-echo-brain/src/'],
        'report_path': '/opt/tower-echo-brain/data/reports/',
        'exclude_patterns': ['*.pyc', '__pycache__', '.git']
    },
    'financial_monitoring': {
        'enabled': True,
        'read_only': True,  # No transactions, alerts only
        'check_interval_hours': 24,
        'alert_threshold_usd': 500,
        'log_path': '/opt/tower-echo-brain/data/financial/',
        'plaid_sandbox_only': True  # Extra safety
    },
    'self_modification': {
        'enabled': False,  # DISABLED until proven stable
        'require_approval': True,
        'allowed_files': [],  # Empty = no files can be modified
        'git_commits_enabled': False
    },
    'resource_limits': {
        'max_gpu_vram_gb': 6,  # Leave 6GB free for system
        'max_ram_gb': 8,
        'max_disk_gb': 100,
        'max_cpu_percent': 50,
        'max_processes': 10,
        'max_threads_per_process': 4
    },
    'safety_features': {
        'auto_shutdown_on_exceed': True,
        'alert_on_threshold': True,
        'threshold_percent': 80,
        'cooldown_minutes': 5,
        'max_retries': 3,
        'quarantine_failed_tasks': True
    },
    'monitoring': {
        'health_check_interval_seconds': 60,
        'metrics_retention_days': 7,
        'log_level': 'INFO',
        'alert_channels': ['file', 'console'],
        'dashboard_refresh_seconds': 5
    }
}

# Database Configuration
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'patrick',
    'password': 'tower_echo_brain_secret_key_2025',
    'database': 'echo_brain',
    'max_connections': 10,
    'connection_timeout': 5
}

# Service URLs
SERVICES = {
    'echo_brain': 'http://localhost:8309',
    'comfyui': 'http://localhost:8188',
    'redis': 'redis://localhost:6379',
    'qdrant': 'http://localhost:6333'
}

# Fail Conditions (triggers auto-shutdown)
FAIL_CONDITIONS = {
    'gpu_vram_percent': 90,
    'disk_percent': 90,
    'cpu_percent_sustained': 80,
    'cpu_sustained_minutes': 5,
    'service_crash': True,
    'database_connection_lost': True,
    'failed_task_threshold': 100,
    'memory_leak_gb': 2  # If process grows by 2GB
}

# Success Metrics (for 24h validation)
SUCCESS_METRICS = {
    'required_uptime_percent': 100,
    'min_tasks_completed': 50,
    'min_anime_generated': 3,
    'max_crashes_allowed': 0,
    'max_error_logs': 10,
    'required_services': ['echo_brain', 'postgresql', 'redis', 'qdrant']
}

# Validation Tasks Schedule
VALIDATION_TASKS = {
    'anime_generation': {
        'schedule': '0 */6 * * *',  # Every 6 hours
        'priority': 5,
        'timeout_seconds': 300
    },
    'code_analysis': {
        'schedule': '0 */12 * * *',  # Every 12 hours
        'priority': 3,
        'timeout_seconds': 120
    },
    'financial_check': {
        'schedule': '0 9 * * *',  # Daily at 9am
        'priority': 8,
        'timeout_seconds': 60
    },
    'system_health': {
        'schedule': '0 * * * *',  # Every hour
        'priority': 10,
        'timeout_seconds': 30
    }
}

def is_within_limits(resource_type: str, current_value: float) -> bool:
    """Check if resource usage is within configured limits"""
    limits = AUTONOMY_CONFIG['resource_limits']

    if resource_type == 'gpu_vram':
        return current_value <= limits['max_gpu_vram_gb']
    elif resource_type == 'ram':
        return current_value <= limits['max_ram_gb']
    elif resource_type == 'disk':
        return current_value <= limits['max_disk_gb']
    elif resource_type == 'cpu':
        return current_value <= limits['max_cpu_percent']
    else:
        return True

def should_shutdown(metrics: dict) -> tuple[bool, str]:
    """Determine if system should shutdown based on fail conditions"""
    for condition, threshold in FAIL_CONDITIONS.items():
        if condition in metrics:
            if isinstance(threshold, bool) and metrics[condition] == threshold:
                return True, f"Fail condition triggered: {condition}"
            elif isinstance(threshold, (int, float)) and metrics[condition] > threshold:
                return True, f"Fail condition exceeded: {condition}={metrics[condition]} > {threshold}"
    return False, "All systems within limits"
#!/usr/bin/env python3
"""
Echo Brain Task Queue System
AsyncIO-based task queue with Redis backend and PostgreSQL persistence
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import redis.asyncio as redis
import psycopg2
from psycopg2.extras import RealDictCursor
import os

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    URGENT = 1      # System failures, security issues
    HIGH = 2        # Performance issues, user requests  
    NORMAL = 3      # Regular monitoring, optimization
    LOW = 4         # Background learning, cleanup
    SCHEDULED = 5   # Daily reports, backups

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class TaskType(Enum):
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"
    LEARNING = "learning"
    MAINTENANCE = "maintenance"
    USER_REQUEST = "user_request"
    SYSTEM_REPAIR = "system_repair"
    ANALYSIS = "analysis"
    CODE_REFACTOR = "code_refactor"
    BACKUP = "backup"
    LORA_GENERATION = "LORA_GENERATION"
    LORA_TAGGING = "LORA_TAGGING" 
    LORA_TRAINING = "LORA_TRAINING"

@dataclass
class Task:
    """Task representation with all metadata"""
    id: str
    name: str
    task_type: TaskType
    priority: TaskPriority
    status: TaskStatus
    payload: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    scheduled_for: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries: int = 0
    max_retries: int = 3
    timeout: int = 300  # 5 minutes default
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    dependencies: List[str] = None
    creator: str = "echo_autonomous"
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization"""
        data = asdict(self)
        # Convert enums to values
        data['task_type'] = self.task_type.value
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        # Convert datetime to ISO string
        for field in ['created_at', 'updated_at', 'scheduled_for', 'started_at', 'completed_at']:
            if data[field]:
                data[field] = data[field].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary"""
        # Convert string values back to enums
        data['task_type'] = TaskType(data['task_type'])
        data['priority'] = TaskPriority(data['priority'])
        data['status'] = TaskStatus(data['status'])
        # Convert ISO strings back to datetime (if not already datetime)
        for field in ['created_at', 'updated_at', 'scheduled_for', 'started_at', 'completed_at']:
            if data[field] and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        return cls(**data)

class TaskQueue:
    """AsyncIO-based task queue with Redis and PostgreSQL persistence"""
    
    def __init__(self,
                 redis_url: str = "redis://localhost:6379/0",
                 db_config: Optional[Dict[str, str]] = None):
        self.redis_url = redis_url
        self.db_config = db_config or self._get_db_config_from_vault()
        self.redis_client: Optional[redis.Redis] = None
        self.running = False
        self.task_handlers: Dict[TaskType, Callable] = {}

    def _get_db_config_from_vault(self):
        """Get database configuration from HashiCorp Vault or fallback to env"""
        if os.environ.get("USE_VAULT", "false").lower() == "true":
            try:
                import hvac
                vault_addr = os.environ.get("VAULT_ADDR", "http://127.0.0.1:8200")

                # Try to get vault token from file
                token_file = os.environ.get("VAULT_TOKEN_FILE", "/opt/tower-echo-brain/.vault-token")
                if os.path.exists(token_file):
                    with open(token_file, 'r') as f:
                        vault_token = f.read().strip()
                else:
                    vault_token = os.environ.get("VAULT_TOKEN")

                if vault_token:
                    client = hvac.Client(url=vault_addr, token=vault_token)

                    # Get database credentials from vault
                    db_secrets = client.secrets.kv.v2.read_secret_version(path='tower/database')
                    db_data = db_secrets['data']['data']

                    return {
                        "database": db_data.get("database", "echo_brain"),
                        "user": db_data.get("user", "patrick"),
                        "host": db_data.get("host", "localhost"),
                        "password": db_data.get("password"),
                        "port": int(db_data.get("port", 5432))
                    }
            except Exception as e:
                logger.warning(f"Failed to get credentials from Vault: {e}, falling back to env vars")

        # Fallback to environment variables
        return {
            "database": os.environ.get("DB_NAME", "echo_brain"),
            "user": os.environ.get("DB_USER", "patrick"),
            "host": os.environ.get("DB_HOST", "localhost"),
            "password": os.environ.get("DB_PASSWORD", "***REMOVED***"),
            "port": int(os.environ.get("DB_PORT", 5432))
        }
        
    async def initialize(self):
        """Initialize Redis connection and database"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("✅ Connected to Redis for task queue")
        except Exception as e:
            logger.warning(f"Redis connection failed, using in-memory queue: {e}")
            self.redis_client = None
            
        await self._create_database_tables()
        
    async def _create_database_tables(self):
        """Create task tables if they don't exist"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS echo_tasks (
                    id VARCHAR(36) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    task_type VARCHAR(50) NOT NULL,
                    priority INTEGER NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    scheduled_for TIMESTAMP WITH TIME ZONE,
                    started_at TIMESTAMP WITH TIME ZONE,
                    completed_at TIMESTAMP WITH TIME ZONE,
                    retries INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    timeout INTEGER DEFAULT 300,
                    result JSONB,
                    error TEXT,
                    dependencies JSONB DEFAULT '[]',
                    creator VARCHAR(100) DEFAULT 'echo_autonomous'
                );
                
                CREATE INDEX IF NOT EXISTS idx_echo_tasks_status ON echo_tasks(status);
                CREATE INDEX IF NOT EXISTS idx_echo_tasks_priority ON echo_tasks(priority);
                CREATE INDEX IF NOT EXISTS idx_echo_tasks_scheduled ON echo_tasks(scheduled_for);
                CREATE INDEX IF NOT EXISTS idx_echo_tasks_type ON echo_tasks(task_type);
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("✅ Task database tables created/verified")
            
        except Exception as e:
            logger.error(f"Failed to create task database tables: {e}")
            
    async def add_task(self, task: Task) -> bool:
        """Add task to queue"""
        try:
            # Store in database for persistence
            await self._persist_task(task)
            
            # Add to Redis queue for processing
            if self.redis_client:
                queue_key = f"echo_tasks:{task.priority.value}"
                await self.redis_client.lpush(queue_key, task.id)
                logger.info(f"📝 Task {task.id} ({task.name}) added to {task.priority.name} queue")
            else:
                logger.info(f"📝 Task {task.id} ({task.name}) added to database (no Redis)")
                
            return True
        except Exception as e:
            logger.error(f"Failed to add task {task.id}: {e}")
            return False
            
    async def _persist_task(self, task: Task):
        """Persist task to PostgreSQL"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO echo_tasks (
                    name, task_type, priority, status, payload,
                    created_at, updated_at, scheduled_for, started_at, completed_at,
                    retries, max_retries, timeout, result, error, dependencies, creator
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                task.name, task.task_type.value, task.priority.value,
                task.status.value, json.dumps(task.payload),
                task.created_at, task.updated_at, task.scheduled_for,
                task.started_at, task.completed_at, task.retries, task.max_retries,
                task.timeout, json.dumps(task.result) if task.result else None,
                task.error, json.dumps(task.dependencies), task.creator
            ))
            
            # Get the database-generated ID
            db_id = cursor.fetchone()[0]
            logger.debug(f"Task {task.id} (UUID) persisted with DB ID: {db_id}")
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to persist task {task.id}: {e}")
            
    async def get_next_task(self) -> Optional[Task]:
        """Get next task from priority queues"""
        if not self.redis_client:
            return await self._get_next_task_from_db()
            
        # Check queues in priority order
        for priority in TaskPriority:
            queue_key = f"echo_tasks:{priority.value}"
            task_id = await self.redis_client.rpop(queue_key)
            if task_id:
                return await self._load_task(task_id.decode())
                
        # Fall back to database if Redis queues are empty
        return await self._get_next_task_from_db()
        
    async def _get_next_task_from_db(self) -> Optional[Task]:
        """Fallback: get next task directly from database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM echo_tasks 
                WHERE status = 'pending' 
                AND (scheduled_for IS NULL OR scheduled_for <= NOW())
                ORDER BY priority ASC, created_at ASC 
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if row:
                task_data = dict(row)
                # Handle JSONB fields that might already be dicts
                if isinstance(task_data['payload'], str):
                    task_data['payload'] = json.loads(task_data['payload'])
                if isinstance(task_data['dependencies'], str):
                    task_data['dependencies'] = json.loads(task_data['dependencies'] or '[]')
                elif task_data['dependencies'] is None:
                    task_data['dependencies'] = []
                if task_data['result'] and isinstance(task_data['result'], str):
                    task_data['result'] = json.loads(task_data['result'])
                return Task.from_dict(task_data)
                
        except Exception as e:
            logger.error(f"Failed to get next task from database: {e}")
            
        return None
        
    async def _load_task(self, task_id: str) -> Optional[Task]:
        """Load task from database by ID"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("SELECT * FROM echo_tasks WHERE id = %s", (task_id,))
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if row:
                task_data = dict(row)
                # Handle JSONB fields that might already be dicts
                if isinstance(task_data['payload'], str):
                    task_data['payload'] = json.loads(task_data['payload'])
                if isinstance(task_data['dependencies'], str):
                    task_data['dependencies'] = json.loads(task_data['dependencies'] or '[]')
                elif task_data['dependencies'] is None:
                    task_data['dependencies'] = []
                if task_data['result'] and isinstance(task_data['result'], str):
                    task_data['result'] = json.loads(task_data['result'])
                return Task.from_dict(task_data)
                
        except Exception as e:
            logger.error(f"Failed to load task {task_id}: {e}")
            
        return None
        
    async def update_task_status(self, task_id: str, status: TaskStatus, 
                               result: Optional[Dict] = None, error: Optional[str] = None):
        """Update task status in database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            now = datetime.now()
            if status == TaskStatus.RUNNING:
                cursor.execute("""
                    UPDATE echo_tasks 
                    SET status = %s, started_at = %s, updated_at = %s
                    WHERE id = %s
                """, (status.value, now, now, task_id))
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                cursor.execute("""
                    UPDATE echo_tasks 
                    SET status = %s, completed_at = %s, updated_at = %s, result = %s, error = %s
                    WHERE id = %s
                """, (status.value, now, now, 
                       json.dumps(result) if result else None, error, task_id))
            else:
                cursor.execute("""
                    UPDATE echo_tasks 
                    SET status = %s, updated_at = %s, error = %s
                    WHERE id = %s
                """, (status.value, now, error, task_id))
                
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update task {task_id} status: {e}")
            
    async def get_task_stats(self) -> Dict[str, Any]:
        """Get task queue statistics"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    status,
                    task_type,
                    priority,
                    COUNT(*) as count
                FROM echo_tasks 
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY status, task_type, priority
            """)
            
            stats = {
                'total_24h': 0,
                'by_status': {},
                'by_type': {},
                'by_priority': {}
            }
            
            for row in cursor.fetchall():
                status, task_type, priority, count = row
                stats['total_24h'] += count
                stats['by_status'][status] = stats['by_status'].get(status, 0) + count
                stats['by_type'][task_type] = stats['by_type'].get(task_type, 0) + count
                stats['by_priority'][str(priority)] = stats['by_priority'].get(str(priority), 0) + count
                
            cursor.close()
            conn.close()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get task stats: {e}")
            return {'error': str(e)}
            
    async def cleanup_old_tasks(self, days: int = 7):
        """Clean up completed tasks older than specified days"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM echo_tasks 
                WHERE status IN ('completed', 'failed', 'cancelled')
                AND completed_at < NOW() - INTERVAL '%s days'
            """, (days,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"🧹 Cleaned up {deleted_count} old tasks")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old tasks: {e}")
            return 0

# Utility functions for creating common tasks
def create_monitoring_task(name: str, target: str, check_type: str, 
                          priority: TaskPriority = TaskPriority.NORMAL) -> Task:
    """Create a monitoring task"""
    return Task(
        id=str(uuid.uuid4()),
        name=name,
        task_type=TaskType.MONITORING,
        priority=priority,
        status=TaskStatus.PENDING,
        payload={
            'target': target,
            'check_type': check_type,
            'timestamp': datetime.now().isoformat()
        },
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

def create_optimization_task(name: str, system: str, metric: str,
                           priority: TaskPriority = TaskPriority.NORMAL) -> Task:
    """Create an optimization task"""
    return Task(
        id=str(uuid.uuid4()),
        name=name,
        task_type=TaskType.OPTIMIZATION,
        priority=priority,
        status=TaskStatus.PENDING,
        payload={
            'system': system,
            'metric': metric,
            'timestamp': datetime.now().isoformat()
        },
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

def create_learning_task(name: str, data_source: str, pattern: str,
                        priority: TaskPriority = TaskPriority.LOW) -> Task:
    """Create a learning task"""
    return Task(
        id=str(uuid.uuid4()),
        name=name,
        task_type=TaskType.LEARNING,
        priority=priority,
        status=TaskStatus.PENDING,
        payload={
            'data_source': data_source,
            'pattern': pattern,
            'timestamp': datetime.now().isoformat()
        },
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

def create_maintenance_task(name: str, action: str, target: str,
                          priority: TaskPriority = TaskPriority.LOW) -> Task:
    """Create a maintenance task"""
    return Task(
        id=str(uuid.uuid4()),
        name=name,
        task_type=TaskType.MAINTENANCE,
        priority=priority,
        status=TaskStatus.PENDING,
        payload={
            'action': action,
            'target': target,
            'timestamp': datetime.now().isoformat()
        },
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

def create_scheduled_task(name: str, task_type: TaskType, schedule_time: datetime,
                         payload: Dict[str, Any]) -> Task:
    """Create a scheduled task"""
    return Task(
        id=str(uuid.uuid4()),
        name=name,
        task_type=task_type,
        priority=TaskPriority.SCHEDULED,
        status=TaskStatus.PENDING,
        payload=payload,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        scheduled_for=schedule_time
    )

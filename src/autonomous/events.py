"""
Event Watcher for Echo Brain Autonomous Operations

The EventWatcher class monitors for triggering events that should initiate
autonomous operations, including file system changes, database modifications,
and external triggers.
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import asyncpg
from contextlib import asynccontextmanager
import json
import hashlib

logger = logging.getLogger(__name__)

# For file system monitoring
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logger.warning("watchdog not available - file system monitoring disabled")

    # Create stub classes when watchdog is not available
    class FileSystemEventHandler:
        pass

    class FileSystemEvent:
        pass


@dataclass
class EventTrigger:
    """Represents an event that can trigger autonomous operations"""
    id: str
    event_type: str
    description: str
    handler_function: str
    config: Dict[str, Any]
    active: bool = True
    last_triggered: Optional[datetime] = None


@dataclass
class DetectedEvent:
    """Represents a detected event"""
    trigger_id: str
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    source: str


class ConversationFileHandler(FileSystemEventHandler):
    """Handles file system events for conversation files"""

    def __init__(self, event_watcher):
        self.event_watcher = event_watcher
        self.processed_files = set()

    def on_created(self, event: FileSystemEvent):
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Check if it's a conversation file
        if self._is_conversation_file(file_path):
            self.event_watcher.queue_event(DetectedEvent(
                trigger_id="conversation_file_created",
                event_type="file_created",
                timestamp=datetime.now(),
                data={
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size if file_path.exists() else 0
                },
                source="file_system"
            ))

    def on_modified(self, event: FileSystemEvent):
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Check if it's a conversation file and hasn't been processed recently
        if (self._is_conversation_file(file_path) and
            str(file_path) not in self.processed_files):

            # Add to processed files with TTL
            self.processed_files.add(str(file_path))

            # Remove from processed files after delay to avoid spam
            asyncio.create_task(self._remove_from_processed(str(file_path), 30))

            self.event_watcher.queue_event(DetectedEvent(
                trigger_id="conversation_file_modified",
                event_type="file_modified",
                timestamp=datetime.now(),
                data={
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size if file_path.exists() else 0
                },
                source="file_system"
            ))

    def _is_conversation_file(self, file_path: Path) -> bool:
        """Check if a file is a conversation file"""
        # Look for common conversation file patterns
        conversation_patterns = [
            "conversation",
            "chat",
            "dialogue",
            "session",
            ".json",
            ".txt",
            ".md"
        ]

        file_name_lower = file_path.name.lower()
        return any(pattern in file_name_lower for pattern in conversation_patterns)

    async def _remove_from_processed(self, file_path: str, delay: int):
        """Remove file from processed set after delay"""
        await asyncio.sleep(delay)
        self.processed_files.discard(file_path)


class EventWatcher:
    """
    Monitors for events that should trigger autonomous operations.

    Provides capabilities for watching file systems, database changes,
    and external triggers to initiate autonomous tasks.
    """

    def __init__(self):
        """Initialize the EventWatcher with database configuration."""
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'echo_brain',
            'user': 'patrick',
            'password': os.environ.get('ECHO_BRAIN_DB_PASSWORD', 'RP78eIrW7cI2jYvL5akt1yurE')
        }
        self._pool = None

        # Event handling
        self.event_queue = asyncio.Queue()
        self.event_handlers = {}
        self.active_triggers = {}
        self.last_db_check = {}

        # File system monitoring
        self.file_observers = {}
        self.watching_directories = set()

        # Database monitoring
        self.db_monitor_task = None
        self.db_monitor_interval = 30  # seconds

        # Event processing
        self.processing_task = None
        self.running = False

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(**self.db_config, min_size=1, max_size=5)

        async with self._pool.acquire() as connection:
            yield connection

    async def start_watching(self):
        """Start monitoring for events."""
        if self.running:
            logger.warning("EventWatcher is already running")
            return

        self.running = True
        logger.info("Starting EventWatcher")

        try:
            # Load event triggers from database
            await self.load_triggers()

            # Start file system monitoring
            await self.start_file_monitoring()

            # Start database monitoring
            self.db_monitor_task = asyncio.create_task(self.monitor_database())

            # Start event processing
            self.processing_task = asyncio.create_task(self.process_events())

            logger.info("EventWatcher started successfully")

        except Exception as e:
            logger.error(f"Failed to start EventWatcher: {e}")
            await self.stop_watching()
            raise

    async def stop_watching(self):
        """Stop monitoring for events."""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping EventWatcher")

        # Stop file system monitoring
        for observer in self.file_observers.values():
            observer.stop()
            observer.join()
        self.file_observers.clear()

        # Stop database monitoring
        if self.db_monitor_task:
            self.db_monitor_task.cancel()
            try:
                await self.db_monitor_task
            except asyncio.CancelledError:
                pass

        # Stop event processing
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        logger.info("EventWatcher stopped")

    async def load_triggers(self):
        """Load event triggers from database."""
        try:
            async with self.get_connection() as conn:
                # Check if triggers table exists
                table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = 'autonomous_event_triggers'
                    )
                """)

                if not table_exists:
                    # Create the table and default triggers
                    await self.create_triggers_table(conn)

                # Load triggers
                triggers = await conn.fetch("""
                    SELECT id, event_type, description, handler_function, config, active, last_triggered
                    FROM autonomous_event_triggers
                    WHERE active = true
                """)

                self.active_triggers = {}
                for trigger in triggers:
                    self.active_triggers[trigger['id']] = EventTrigger(
                        id=trigger['id'],
                        event_type=trigger['event_type'],
                        description=trigger['description'],
                        handler_function=trigger['handler_function'],
                        config=trigger['config'] or {},
                        active=trigger['active'],
                        last_triggered=trigger['last_triggered']
                    )

                logger.info(f"Loaded {len(self.active_triggers)} active event triggers")

        except Exception as e:
            logger.error(f"Failed to load triggers: {e}")
            # Continue with default triggers
            await self.create_default_triggers()

    async def create_triggers_table(self, conn):
        """Create the event triggers table with default triggers."""
        try:
            # Create table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS autonomous_event_triggers (
                    id VARCHAR(100) PRIMARY KEY,
                    event_type VARCHAR(100) NOT NULL,
                    description TEXT NOT NULL,
                    handler_function VARCHAR(255) NOT NULL,
                    config JSONB DEFAULT '{}',
                    active BOOLEAN DEFAULT true,
                    last_triggered TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)

            # Insert default triggers
            default_triggers = [
                (
                    'conversation_file_created',
                    'file_created',
                    'New conversation file detected',
                    'handle_conversation_file',
                    json.dumps({
                        'auto_process': True,
                        'create_summary_task': True,
                        'extract_insights': True
                    })
                ),
                (
                    'conversation_file_modified',
                    'file_modified',
                    'Conversation file updated',
                    'handle_conversation_update',
                    json.dumps({
                        'auto_process': True,
                        'min_interval_minutes': 10
                    })
                ),
                (
                    'new_goal_created',
                    'database_insert',
                    'New autonomous goal created',
                    'handle_new_goal',
                    json.dumps({
                        'auto_create_initial_tasks': True
                    })
                ),
                (
                    'task_completion',
                    'database_update',
                    'Task completed successfully',
                    'handle_task_completion',
                    json.dumps({
                        'check_goal_progress': True,
                        'create_follow_up_tasks': True
                    })
                )
            ]

            for trigger_id, event_type, description, handler, config in default_triggers:
                await conn.execute("""
                    INSERT INTO autonomous_event_triggers (id, event_type, description, handler_function, config)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (id) DO NOTHING
                """, trigger_id, event_type, description, handler, config)

            logger.info("Created event triggers table with default triggers")

        except Exception as e:
            logger.error(f"Failed to create triggers table: {e}")

    async def create_default_triggers(self):
        """Create default triggers when database is not available."""
        self.active_triggers = {
            'conversation_file_created': EventTrigger(
                id='conversation_file_created',
                event_type='file_created',
                description='New conversation file detected',
                handler_function='handle_conversation_file',
                config={'auto_process': True}
            ),
            'conversation_file_modified': EventTrigger(
                id='conversation_file_modified',
                event_type='file_modified',
                description='Conversation file updated',
                handler_function='handle_conversation_update',
                config={'auto_process': True, 'min_interval_minutes': 10}
            )
        }

    async def start_file_monitoring(self):
        """Start file system monitoring for conversation directories."""
        if not WATCHDOG_AVAILABLE:
            logger.warning("File system monitoring not available (watchdog not installed)")
            return

        # Common conversation directories to monitor
        conversation_dirs = [
            "/home/patrick/.claude/conversations",
            "/home/patrick/Documents/conversations",
            "/opt/tower-echo-brain/conversations",
            "/tmp/claude"
        ]

        for dir_path in conversation_dirs:
            if Path(dir_path).exists():
                await self.watch_directory(dir_path)

    async def watch_directory(self, directory_path: str):
        """Watch a specific directory for file changes."""
        if not WATCHDOG_AVAILABLE or directory_path in self.watching_directories:
            return

        try:
            path = Path(directory_path)
            if not path.exists():
                logger.warning(f"Directory {directory_path} does not exist")
                return

            observer = Observer()
            event_handler = ConversationFileHandler(self)
            observer.schedule(event_handler, str(path), recursive=True)
            observer.start()

            self.file_observers[directory_path] = observer
            self.watching_directories.add(directory_path)

            logger.info(f"Started watching directory: {directory_path}")

        except Exception as e:
            logger.error(f"Failed to watch directory {directory_path}: {e}")

    def queue_event(self, event: DetectedEvent):
        """Queue an event for processing."""
        try:
            self.event_queue.put_nowait(event)
            logger.debug(f"Queued event: {event.event_type} from {event.source}")
        except asyncio.QueueFull:
            logger.warning("Event queue is full, dropping event")

    async def process_events(self):
        """Main event processing loop."""
        logger.info("Started event processing loop")

        while self.running:
            try:
                # Wait for events with timeout
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue

                # Process the event
                await self.handle_event(event)

            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(1)

        logger.info("Event processing loop stopped")

    async def handle_event(self, event: DetectedEvent):
        """Handle a detected event."""
        try:
            trigger = self.active_triggers.get(event.trigger_id)
            if not trigger:
                logger.debug(f"No trigger found for event: {event.trigger_id}")
                return

            # Check rate limiting
            if trigger.last_triggered:
                min_interval = trigger.config.get('min_interval_minutes', 0)
                if min_interval > 0:
                    time_since_last = datetime.now() - trigger.last_triggered
                    if time_since_last < timedelta(minutes=min_interval):
                        logger.debug(f"Event {event.trigger_id} rate limited")
                        return

            # Call appropriate handler
            handler_name = trigger.handler_function
            if hasattr(self, handler_name):
                handler = getattr(self, handler_name)
                await handler(event, trigger)

                # Update last triggered time
                await self.update_trigger_timestamp(trigger.id)
            else:
                logger.warning(f"Handler {handler_name} not found for event {event.trigger_id}")

        except Exception as e:
            logger.error(f"Failed to handle event {event.trigger_id}: {e}")

    async def handle_conversation_file(self, event: DetectedEvent, trigger: EventTrigger):
        """Handle new conversation file events."""
        try:
            file_path = event.data.get('file_path')
            logger.info(f"Processing new conversation file: {file_path}")

            if trigger.config.get('create_summary_task', False):
                # Create a task to summarize the conversation
                # This would integrate with the scheduler to create actual tasks
                logger.info(f"Would create summary task for {file_path}")

            if trigger.config.get('extract_insights', False):
                # Create a task to extract insights
                logger.info(f"Would create insights extraction task for {file_path}")

        except Exception as e:
            logger.error(f"Failed to handle conversation file event: {e}")

    async def handle_conversation_update(self, event: DetectedEvent, trigger: EventTrigger):
        """Handle conversation file update events."""
        try:
            file_path = event.data.get('file_path')
            logger.info(f"Processing conversation file update: {file_path}")

            # Could check file size, modification time, etc.
            # and decide whether to create update processing tasks

        except Exception as e:
            logger.error(f"Failed to handle conversation update event: {e}")

    async def monitor_database(self):
        """Monitor database for changes that should trigger events."""
        logger.info("Started database monitoring")

        while self.running:
            try:
                await self.check_database_changes()
                await asyncio.sleep(self.db_monitor_interval)
            except Exception as e:
                logger.error(f"Error in database monitoring: {e}")
                await asyncio.sleep(10)

        logger.info("Database monitoring stopped")

    async def check_database_changes(self):
        """Check for database changes that should trigger events."""
        try:
            async with self.get_connection() as conn:
                # Check for new goals
                await self.check_new_goals(conn)

                # Check for completed tasks
                await self.check_completed_tasks(conn)

        except Exception as e:
            logger.error(f"Failed to check database changes: {e}")

    async def check_new_goals(self, conn):
        """Check for newly created goals."""
        try:
            last_check = self.last_db_check.get('goals', datetime.now() - timedelta(minutes=5))

            new_goals = await conn.fetch("""
                SELECT id, name, goal_type, created_at
                FROM autonomous_goals
                WHERE created_at > $1
                ORDER BY created_at
            """, last_check)

            for goal in new_goals:
                self.queue_event(DetectedEvent(
                    trigger_id="new_goal_created",
                    event_type="database_insert",
                    timestamp=goal['created_at'],
                    data={
                        "goal_id": goal['id'],
                        "goal_name": goal['name'],
                        "goal_type": goal['goal_type']
                    },
                    source="database"
                ))

            if new_goals:
                self.last_db_check['goals'] = max(goal['created_at'] for goal in new_goals)

        except Exception as e:
            logger.error(f"Failed to check new goals: {e}")

    async def check_completed_tasks(self, conn):
        """Check for recently completed tasks."""
        try:
            last_check = self.last_db_check.get('tasks', datetime.now() - timedelta(minutes=5))

            completed_tasks = await conn.fetch("""
                SELECT id, name, task_type, goal_id, completed_at, status
                FROM autonomous_tasks
                WHERE completed_at > $1 AND status = 'completed'
                ORDER BY completed_at
            """, last_check)

            for task in completed_tasks:
                self.queue_event(DetectedEvent(
                    trigger_id="task_completion",
                    event_type="database_update",
                    timestamp=task['completed_at'],
                    data={
                        "task_id": task['id'],
                        "task_name": task['name'],
                        "task_type": task['task_type'],
                        "goal_id": task['goal_id']
                    },
                    source="database"
                ))

            if completed_tasks:
                self.last_db_check['tasks'] = max(task['completed_at'] for task in completed_tasks)

        except Exception as e:
            logger.error(f"Failed to check completed tasks: {e}")

    async def update_trigger_timestamp(self, trigger_id: str):
        """Update the last triggered timestamp for a trigger."""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    UPDATE autonomous_event_triggers
                    SET last_triggered = NOW()
                    WHERE id = $1
                """, trigger_id)

            # Update in memory
            if trigger_id in self.active_triggers:
                self.active_triggers[trigger_id].last_triggered = datetime.now()

        except Exception as e:
            logger.error(f"Failed to update trigger timestamp for {trigger_id}: {e}")

    async def register_handler(self, trigger_id: str, handler_function: Callable):
        """Register a custom event handler."""
        self.event_handlers[trigger_id] = handler_function
        logger.info(f"Registered custom handler for trigger: {trigger_id}")

    async def check_events(self) -> List[DetectedEvent]:
        """Check for pending events (non-blocking)."""
        events = []
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                events.append(event)
            except asyncio.QueueEmpty:
                break
        return events

    async def cleanup(self):
        """Cleanup resources."""
        await self.stop_watching()

        if self._pool:
            await self._pool.close()

        logger.info("EventWatcher cleaned up")
"""
Celery configuration for distributed task execution
"""
from celery import Celery
from kombu import Queue
import os
import logging

logger = logging.getLogger(__name__)

# Create Celery app
app = Celery('echo_brain_tasks')

# Configuration
app.conf.update(
    broker_url='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/0',
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Task execution settings
    task_time_limit=300,  # 5 minutes hard limit
    task_soft_time_limit=240,  # 4 minutes soft limit
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Worker settings
    worker_prefetch_multiplier=2,
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks to prevent memory leaks

    # Queue configuration
    task_default_queue='default',
    task_queues=(
        Queue('default', routing_key='default'),
        Queue('high_priority', routing_key='high'),
        Queue('low_priority', routing_key='low'),
        Queue('coding', routing_key='coding'),
        Queue('reasoning', routing_key='reasoning'),
    ),

    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour

    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Auto-discover tasks from these modules
app.autodiscover_tasks(['autonomous.celery_tasks'])
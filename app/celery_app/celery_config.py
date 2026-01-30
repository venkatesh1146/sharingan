"""
Celery Configuration - Task queue setup and scheduling.

Configures Celery with Redis broker/backend and periodic task schedules.
"""

from celery import Celery
from celery.schedules import crontab

from app.config import get_settings

settings = get_settings()

# Create Celery application
celery_app = Celery(
    "market_intelligence",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.celery_app.tasks.news_tasks",
        "app.celery_app.tasks.snapshot_tasks",
        "app.celery_app.tasks.indices_tasks",
        "app.celery_app.tasks.cleanup_tasks",
    ],
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    
    # Timezone
    timezone=settings.CELERY_TIMEZONE,
    enable_utc=True,
    
    # Task execution
    task_track_started=True,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    
    # Worker settings
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
    
    # Result backend
    result_expires=3600,  # Results expire after 1 hour
    
    # Task routing
    task_routes={
        "app.celery_app.tasks.news_tasks.*": {"queue": "news"},
        "app.celery_app.tasks.snapshot_tasks.*": {"queue": "snapshots"},
        "app.celery_app.tasks.indices_tasks.*": {"queue": "indices"},
        "app.celery_app.tasks.cleanup_tasks.*": {"queue": "maintenance"},
    },
    
    # Default queue
    task_default_queue="default",
    
    # Define all queues
    task_queues={
        "default": {},
        "news": {},
        "snapshots": {},
        "indices": {},
        "maintenance": {},
    },
)

# Periodic task schedule (Celery Beat)
celery_app.conf.beat_schedule = {
    # Fetch and process news every 15 minutes
    "fetch-news-every-15-min": {
        "task": "app.celery_app.tasks.news_tasks.fetch_and_process_news",
        "schedule": settings.NEWS_FETCH_INTERVAL,
        "options": {"queue": "news"},
    },
    
    # Generate market snapshot every 30 minutes
    "generate-snapshot-every-30-min": {
        "task": "app.celery_app.tasks.snapshot_tasks.generate_market_snapshot",
        "schedule": settings.SNAPSHOT_GENERATION_INTERVAL,
        "options": {"queue": "snapshots"},
    },
    
    # Fetch indices data every 5 minutes (during market hours handled in task)
    "fetch-indices-every-5-min": {
        "task": "app.celery_app.tasks.indices_tasks.fetch_indices_data",
        "schedule": settings.INDICES_FETCH_INTERVAL,
        "options": {"queue": "indices"},
    },
    
    # Cleanup old data daily at 2 AM IST
    "cleanup-old-data-daily": {
        "task": "app.celery_app.tasks.cleanup_tasks.cleanup_old_data",
        "schedule": crontab(hour=2, minute=0),
        "options": {"queue": "maintenance"},
    },
}


def get_celery_app() -> Celery:
    """Get the Celery application instance."""
    return celery_app

"""
Celery tasks for Market Intelligence system.

Exports all task modules for automatic discovery.
"""

from app.celery_app.tasks import news_tasks
from app.celery_app.tasks import snapshot_tasks
from app.celery_app.tasks import indices_tasks
from app.celery_app.tasks import cleanup_tasks

__all__ = [
    "news_tasks",
    "snapshot_tasks",
    "indices_tasks",
    "cleanup_tasks",
]

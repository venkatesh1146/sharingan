"""
Celery application for Market Intelligence system.

Provides background task processing for:
- Periodic news fetching and processing
- Market snapshot generation
- Indices data collection
- Data cleanup
"""

from app.celery_app.celery_config import celery_app

__all__ = ["celery_app"]

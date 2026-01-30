"""
Repository pattern implementation for MongoDB collections.

Provides clean data access layer with CRUD operations for each collection.
"""

from app.db.repositories.news_repository import NewsRepository, get_news_repository
from app.db.repositories.snapshot_repository import SnapshotRepository, get_snapshot_repository
from app.db.repositories.indices_repository import IndicesRepository, get_indices_repository

__all__ = [
    "NewsRepository",
    "SnapshotRepository",
    "IndicesRepository",
    "get_news_repository",
    "get_snapshot_repository",
    "get_indices_repository",
]

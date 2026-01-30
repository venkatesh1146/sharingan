"""
MongoDB document models for Market Intelligence system.

These models define the structure of documents stored in MongoDB collections.
"""

from app.db.models.news_document import NewsArticleDocument
from app.db.models.snapshot_document import MarketSnapshotDocument
from app.db.models.indices_document import IndicesTimeseriesDocument

__all__ = [
    "NewsArticleDocument",
    "MarketSnapshotDocument", 
    "IndicesTimeseriesDocument",
]

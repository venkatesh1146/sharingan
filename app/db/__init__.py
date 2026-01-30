"""
Database layer for Market Intelligence system.

Provides MongoDB connection management and repository pattern implementation.
"""

from app.db.mongodb import MongoDBClient, get_mongodb_client

__all__ = ["MongoDBClient", "get_mongodb_client"]

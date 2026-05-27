"""
Chess database module for storing move classifications.

This module provides database operations for storing and querying
chess game metadata and move classifications.
"""

from .schema import init_database, get_connection, DB_PATH, get_schema_version
from .chess_db import ChessDatabase, ParsedGame, MoveClassification

__all__ = [
    "ChessDatabase",
    "ParsedGame",
    "MoveClassification",
    "init_database",
    "get_connection",
    "DB_PATH",
    "get_schema_version",
]

"""
Chess database operations module.
Provides CRUD operations for games and moves.
"""

import sqlite3
import threading
from datetime import datetime
from typing import Any, Optional, List
from dataclasses import dataclass, asdict

from .schema import get_connection, init_database, DB_PATH


@dataclass
class ParsedGame:
    """Represents a parsed chess game."""
    id: int
    white_player: str
    black_player: str
    fen_start: Optional[str]
    fen_end: Optional[str]
    result: Optional[str]
    classification: str
    created_at: str


@dataclass
class MoveClassification:
    """Represents a move classification."""
    id: int
    game_id: int
    move_number: int
    fen_before: Optional[str]
    fen_after: Optional[str]
    move_san: str
    classification: str
    evaluation: float
    created_at: str


class ChessDatabase:
    """Database wrapper for chess game and move storage."""
    
    _lock = threading.Lock()
    
    def __init__(self, db_path: str = None):
        """
        Initialize the chess database.
        
        Args:
            db_path: Path to the SQLite database file. Defaults to chess_bot.db
        """
        self.db_path = db_path or DB_PATH
        self._connection: Optional[sqlite3.Connection] = None
        self._initialize_connection()
    
    def _initialize_connection(self) -> None:
        """Initialize the database connection and create tables."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            init_database(conn)
        finally:
            conn.close()
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
            self._connection.row_factory = sqlite3.Row
        return self._connection
    
    def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def save_game(self, game: ParsedGame) -> int:
        """
        Save a game to the database.
        
        Args:
            game: The game to save
            
        Returns:
            The game ID (or existing ID if game already exists)
        """
        with self._lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT id FROM games WHERE "
                "(white_player = ? AND black_player = ? AND fen_start = ? AND fen_end = ?)",
                (game.white_player, game.black_player, game.fen_start, game.fen_end)
            )
            
            if cursor.fetchone():
                return game.id
            
            # Insert new game
            cursor.execute("""
                INSERT INTO games (
                    white_player, black_player, fen_start, fen_end,
                    result, classification
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                game.white_player, game.black_player, game.fen_start,
                game.fen_end, game.result, game.classification
            ))
            
            game.id = cursor.lastrowid
            conn.commit()
            return game.id
    
    def save_moves_batch(self, game_id: int, move_classifications: List[Any]) -> None:
        """
        Save a batch of move classifications for a game.
        """
        if not move_classifications:
            return
            
        with self._lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = """
                INSERT INTO moves (
                    game_id, move_number, fen_before, fen_after, move_san, classification, evaluation
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            data = [
                (
                    game_id,
                    move.move_number,
                    move.fen_before,
                    move.fen_after,
                    move.move_san,
                    move.classification,
                    move.evaluation
                )
                for move in move_classifications
            ]
            
            cursor.executemany(query, data)
            conn.commit()
    
    def get_games_by_classification(
        self, 
        classification: str, 
        limit: int = 100
    ) -> List[ParsedGame]:
        """
        Get games filtered by classification type.
        
        Args:
            classification: The classification to filter by
            limit: Maximum number of games to return
            
        Returns:
            List of games matching the classification
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, white_player, black_player, fen_start, fen_end,
                   result, classification, created_at
            FROM games
            WHERE classification = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (classification, limit))
        
        rows = cursor.fetchall()
        return [
            ParsedGame(
                id=row["id"],
                white_player=row["white_player"],
                black_player=row["black_player"],
                fen_start=row["fen_start"],
                fen_end=row["fen_end"],
                result=row["result"],
                classification=row["classification"],
                created_at=row["created_at"]
            )
            for row in rows
        ]
    
    def get_player_statistics(self, player: str) -> dict:
        """
        Get statistics for a player.
        
        Args:
            player: The player name
            
        Returns:
            Dictionary with player statistics
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get total games played
        cursor.execute("""
            SELECT 
                COUNT(*) as total_games,
                COUNT(CASE WHEN classification = 'win' THEN 1 END) as wins,
                COUNT(CASE WHEN classification = 'loss' THEN 1 END) as losses,
                COUNT(CASE WHEN classification = 'draw' THEN 1 END) as draws
            FROM games
            WHERE white_player = ? OR black_player = ?
        """, (player, player))
        
        row = cursor.fetchone()
        stats = {
            "player": player,
            "total_games": row["total_games"] or 0,
            "wins": row["wins"] or 0,
            "losses": row["losses"] or 0,
            "draws": row["draws"] or 0,
            "win_rate": 0.0
        }
        
        if stats["total_games"] > 0:
            stats["win_rate"] = (stats["wins"] / stats["total_games"]) * 100
        
        # Get classification breakdown
        cursor.execute("""
            SELECT classification, COUNT(*) as count
            FROM games
            WHERE white_player = ? OR black_player = ?
            GROUP BY classification
            ORDER BY count DESC
        """, (player, player))
        
        stats["classification_breakdown"] = {
            row["classification"]: row["count"]
            for row in cursor.fetchall()
        }
        
        return stats
    
    def get_all_games(self, limit: int = None) -> List[ParsedGame]:
        """
        Get all games from the database.
        
        Args:
            limit: Maximum number of games to return. If None, returns all games.
            
        Returns:
            List of all games (up to limit if specified)
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if limit is not None:
            cursor.execute("""
                SELECT id, white_player, black_player, fen_start, fen_end,
                       result, classification, created_at
                FROM games
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
        else:
            cursor.execute("""
                SELECT id, white_player, black_player, fen_start, fen_end,
                       result, classification, created_at
                FROM games
                ORDER BY created_at DESC
            """)
        
        rows = cursor.fetchall()
        return [
            ParsedGame(
                id=row["id"],
                white_player=row["white_player"],
                black_player=row["black_player"],
                fen_start=row["fen_start"],
                fen_end=row["fen_end"],
                result=row["result"],
                classification=row["classification"],
                created_at=row["created_at"]
            )
            for row in rows
        ]
    
    def get_game_by_id(self, game_id: int) -> Optional[ParsedGame]:
        """
        Get a game by its ID.
        
        Args:
            game_id: The game ID
            
        Returns:
            The game or None if not found
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, white_player, black_player, fen_start, fen_end,
                   result, classification, created_at
            FROM games
            WHERE id = ?
        """, (game_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return ParsedGame(
            id=row["id"],
            white_player=row["white_player"],
            black_player=row["black_player"],
            fen_start=row["fen_start"],
            fen_end=row["fen_end"],
            result=row["result"],
            classification=row["classification"],
            created_at=row["created_at"]
        )
    
    def get_moves_by_game(self, game_id: int) -> List[MoveClassification]:
        """
        Get all moves for a specific game.
        
        Args:
            game_id: The game ID
            
        Returns:
            List of move classifications for the game
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, game_id, move_number, fen_before, fen_after,
                   move_san, classification, created_at
            FROM moves
            WHERE game_id = ?
            ORDER BY move_number
        """, (game_id,))
        
        rows = cursor.fetchall()
        return [
            MoveClassification(
                id=row["id"],
                game_id=row["game_id"],
                move_number=row["move_number"],
                fen_before=row["fen_before"],
                fen_after=row["fen_after"],
                move_san=row["move_san"],
                classification=row["classification"],
                created_at=row["created_at"]
            )
            for row in rows
        ]
    
    def get_moves_by_classification(
        self, 
        classification: str, 
        limit: int = 100
    ) -> List[MoveClassification]:
        """
        Get moves filtered by classification type.
        
        Args:
            classification: The classification to filter by
            limit: Maximum number of moves to return
            
        Returns:
            List of moves matching the classification
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, game_id, move_number, fen_before, fen_after,
                   move_san, classification, created_at
            FROM moves
            WHERE classification = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (classification, limit))
        
        rows = cursor.fetchall()
        return [
            MoveClassification(
                id=row["id"],
                game_id=row["game_id"],
                move_number=row["move_number"],
                fen_before=row["fen_before"],
                fen_after=row["fen_after"],
                move_san=row["move_san"],
                classification=row["classification"],
                created_at=row["created_at"]
            )
            for row in rows
        ]
    
    def delete_game(self, game_id: int) -> bool:
        """
        Delete a game and its associated moves.
        
        Args:
            game_id: The game ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM games WHERE id = ?", (game_id,))
            
            if cursor.rowcount > 0:
                conn.commit()
                return True
            
            return False
    
    def get_game_count(self) -> int:
        """Get the total number of games in the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM games")
        return cursor.fetchone()[0]
    
    def get_move_count(self) -> int:
        """Get the total number of moves in the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM moves")
        return cursor.fetchone()[0]

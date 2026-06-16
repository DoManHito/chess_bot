"""
SQL schema definitions for chess database.
"""

import sqlite3
from typing import Optional

DB_PATH = "chess_bot.db"


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    """Create the games and moves tables if they don't exist."""
    cursor = conn.cursor()
    
    # Games table - stores game metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            white_player TEXT NOT NULL,
            black_player TEXT NOT NULL,
            fen_start TEXT,
            fen_end TEXT,
            result TEXT,
            classification TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Moves table - stores move classifications
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS moves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER NOT NULL,
            move_number INTEGER NOT NULL,
            fen_before TEXT,
            fen_after TEXT,
            move_san TEXT,
            classification TEXT,
            evaluation REAL,
            evaluation_change REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (game_id) REFERENCES games (id) ON DELETE CASCADE
        )
    """)
    
    # Create indexes for efficient querying
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_games_classification
        ON games (classification)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_moves_game_id
        ON moves (game_id)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_moves_classification
        ON moves (classification)
    """)
    
    # Self-play moves table for RL training
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS self_play_moves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER,
            fen_before TEXT,
            move_uci TEXT,
            mcts_policy TEXT,
            result_value REAL,
            lookahead_depth INTEGER,
            future_moves TEXT,
            final_classification TEXT,
            move_sequence_classes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Indexes for self_play_moves
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_self_play_game_id
        ON self_play_moves (game_id)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_self_play_fen
        ON self_play_moves (fen_before)
    """)
    
    conn.commit()


def init_database(db_path: Optional[str] = None) -> None:
    """Initialize the database and create tables."""
    conn = get_connection()
    try:
        create_tables(conn)
    finally:
        conn.close()


def get_schema_version() -> str:
    """Return the current schema version."""
    return "1.0.0"

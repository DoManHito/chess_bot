"""
Database migration script for unified model with lookahead support.

This script updates the database schema to support:
1. Lookahead data in self_play_moves table
2. New move_sequences table for storing lookahead sequences
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def migrate_database(db_path: str = "chess_bot.db"):
    """
    Apply database migrations for unified model support.

    Args:
        db_path: Path to the SQLite database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("Starting database migration...")

    # Migration 1: Add lookahead columns to self_play_moves table
    print("Migration 1: Adding lookahead columns to self_play_moves...")
    
    try:
        cursor.execute("""
            ALTER TABLE self_play_moves 
            ADD COLUMN lookahead_depth INTEGER
        """)
        print("  ✓ Added lookahead_depth column")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("  - lookahead_depth column already exists")
        else:
            raise

    try:
        cursor.execute("""
            ALTER TABLE self_play_moves 
            ADD COLUMN future_moves TEXT
        """)
        print("  ✓ Added future_moves column")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("  - future_moves column already exists")
        else:
            raise

    try:
        cursor.execute("""
            ALTER TABLE self_play_moves 
            ADD COLUMN final_classification TEXT
        """)
        print("  ✓ Added final_classification column")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("  - final_classification column already exists")
        else:
            raise

    try:
        cursor.execute("""
            ALTER TABLE self_play_moves 
            ADD COLUMN move_sequence_classes TEXT
        """)
        print("  ✓ Added move_sequence_classes column")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("  - move_sequence_classes column already exists")
        else:
            raise

    # Migration 2: Create move_sequences table for lookahead data
    print("\nMigration 2: Creating move_sequences table...")
    
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS move_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                move_number INTEGER NOT NULL,
                fen_before TEXT,
                move_san TEXT,
                lookahead_depth INTEGER,
                future_moves TEXT,
                final_evaluation REAL,
                final_classification TEXT,
                move_sequence_classes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (game_id) REFERENCES games (id) ON DELETE CASCADE
            )
        """)
        print("  ✓ Created move_sequences table")
    except sqlite3.OperationalError as e:
        print(f"  - Table may already exist: {e}")

    # Migration 3: Create indexes for move_sequences
    print("\nMigration 3: Creating indexes...")
    
    try:
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_move_sequences_game_id 
            ON move_sequences (game_id)
        """)
        print("  ✓ Created index on game_id")
    except sqlite3.OperationalError as e:
        print(f"  - Index may already exist: {e}")

    try:
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_move_sequences_classification 
            ON move_sequences (final_classification)
        """)
        print("  ✓ Created index on final_classification")
    except sqlite3.OperationalError as e:
        print(f"  - Index may already exist: {e}")

    # Migration 4: Add evaluation_change column to moves table (if not exists)
    print("\nMigration 4: Checking moves table columns...")
    
    try:
        cursor.execute("PRAGMA table_info(moves)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'evaluation_change' not in columns:
            cursor.execute("""
                ALTER TABLE moves 
                ADD COLUMN evaluation_change REAL
            """)
            print("  ✓ Added evaluation_change column")
        else:
            print("  - evaluation_change column already exists")
    except sqlite3.OperationalError as e:
        print(f"  - Column may already exist: {e}")

    conn.commit()
    conn.close()

    print("\n✓ Database migration completed successfully!")


def main():
    """Run the database migration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate chess bot database for unified model")
    parser.add_argument("--db", default="chess_bot.db", help="Path to database file")
    args = parser.parse_args()
    
    migrate_database(args.db)


if __name__ == "__main__":
    main()

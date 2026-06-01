"""
Database migration script to add evaluation_change column.
This script adds the missing evaluation_change column to the existing moves table.
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = "chess_bot.db"


def add_evaluation_change_column(db_path: str) -> bool:
    """
    Add evaluation_change column to the moves table if it doesn't exist.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        True if column was added, False if it already exists or error occurred
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if column already exists
        cursor.execute("PRAGMA table_info(moves)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if "evaluation_change" in columns:
            print("Column 'evaluation_change' already exists in moves table.")
            conn.close()
            return True
        
        # Add the column
        cursor.execute("""
            ALTER TABLE moves
            ADD COLUMN evaluation_change REAL DEFAULT 0.0
        """)
        
        conn.commit()
        print(f"Successfully added 'evaluation_change' column to moves table in {db_path}")
        conn.close()
        return True
        
    except sqlite3.OperationalError as e:
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def calculate_evaluation_changes(db_path: str) -> int:
    """
    Calculate evaluation_change for each move (evaluation_after - evaluation_before).
    Requires chess module to be installed.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        Number of rows updated
    """
    try:
        import chess
    except ImportError:
        print("chess module not installed. Skipping evaluation change calculation.")
        return 0
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all moves with their evaluations
        cursor.execute("""
            SELECT id, game_id, move_number, fen_before, fen_after, evaluation
            FROM moves
            WHERE evaluation_change IS NULL OR evaluation_change = 0
        """)
        
        moves = cursor.fetchall()
        updated_count = 0
        
        for move in moves:
            move_id, game_id, move_number, fen_before, fen_after, evaluation = move
            
            try:
                board_before = chess.Board(fen_before)
                
                # Calculate evaluation change using score() method
                # score() returns a Score object with white_score and black_score
                # For simplicity, we use white_score as the evaluation
                score = board_before.score()
                eval_before = score.white_score if score.white_score else 0
                
                # Calculate evaluation change
                eval_change = evaluation - eval_before
                
                # Update the row
                cursor.execute("""
                    UPDATE moves
                    SET evaluation_change = ?
                    WHERE id = ?
                """, (eval_change, move_id))
                
                updated_count += 1
                
            except Exception as e:
                # Skip moves that can't be parsed
                continue
        
        conn.commit()
        print(f"Updated evaluation_change for {updated_count} moves")
        conn.close()
        return updated_count
        
    except Exception as e:
        print(f"Error calculating evaluation changes: {e}")
        return 0


if __name__ == "__main__":
    print("=" * 60)
    print("Chess Database Migration Script")
    print("=" * 60)
    
    # Add the missing column
    print("\n[1/2] Adding evaluation_change column...")
    success = add_evaluation_change_column(DB_PATH)
    
    if success:
        # Calculate evaluation changes
        print("\n[2/2] Calculating evaluation_change values...")
        updated = calculate_evaluation_changes(DB_PATH)
        print(f"\nTotal moves updated: {updated}")
    else:
        print("\nMigration failed. Skipping evaluation change calculation.")
    
    print("\nMigration complete!")

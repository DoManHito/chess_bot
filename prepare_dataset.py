import sqlite3
import chess
from tqdm import tqdm

def add_fen_after_column(db_path="chess_bot.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(moves)")
    columns = [col[1] for col in cursor.fetchall()]
    if "fen_after" not in columns:
        print("Adding fen_after column to database...")
        cursor.execute("ALTER TABLE moves ADD COLUMN fen_after TEXT")
        conn.commit()
    
    print("Checking database for uncalculated FENs...")
    cursor.execute("""
        SELECT id, fen_before, move_san
        FROM moves
        WHERE fen_after IS NULL OR fen_after = ''
    """)
    rows = cursor.fetchall()
    
    if not rows:
        print("All FENs calculated or database is empty!")
        conn.close()
        return

    print(f"Starting fen_after calculation for {len(rows)} moves...")
    
    updates = []
    for row_id, fen_before, move_san in tqdm(rows):
        board = chess.Board(fen_before)
        try:
            move = board.parse_san(move_san)
            board.push(move)
            fen_after = board.fen()
        except Exception:
            fen_after = fen_before
            
        updates.append((fen_after, row_id))
        
        if len(updates) >= 50000:
            cursor.executemany("UPDATE moves SET fen_after = ? WHERE id = ?", updates)
            conn.commit()
            updates = []
            
    if updates:
        cursor.executemany("UPDATE moves SET fen_after = ? WHERE id = ?", updates)
        conn.commit()
        
    print("Creating index to speed up queries during training...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_moves_game_id ON moves(game_id, id)")
    conn.commit()
    
    conn.close()
    print("Done! Database successfully optimized.")

if __name__ == "__main__":
    add_fen_after_column()
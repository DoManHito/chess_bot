# check_db.py
import sqlite3

def diagnose_database(db_path="chess_bot.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Check for moves table existence
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='moves'")
    if not cursor.fetchone():
        print("❌ Error: 'moves' table does not exist in database!")
        conn.close()
        return
        
    # 2. Count total moves
    cursor.execute("SELECT COUNT(*) FROM moves")
    total_rows = cursor.fetchone()[0]
    print(f"📊 Total moves in database (rows): {total_rows}")
    
    if total_rows == 0:
        print("⚠️ Database is empty! You likely forgot to run pgn_parser.py to populate it.")
        conn.close()
        return

    # 3. Check fen_after column status
    cursor.execute("PRAGMA table_info(moves)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if "fen_after" not in columns:
        print("ℹ️ Column 'fen_after' does not exist yet.")
    else:
        cursor.execute("SELECT COUNT(*) FROM moves WHERE fen_after IS NULL OR fen_after = ''")
        empty_rows = cursor.fetchone()[0]
        
        filled_rows = total_rows - empty_rows
        print(f"✅ FENs (fen_after) already filled: {filled_rows}")
        print(f"🔍 Remaining to fill: {empty_rows}")
        
    conn.close()

if __name__ == "__main__":
    diagnose_database()
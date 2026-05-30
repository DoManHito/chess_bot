# check_db.py
import sqlite3

def diagnose_database(db_path="chess_bot.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Проверяем наличие таблицы moves
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='moves'")
    if not cursor.fetchone():
        print("❌ Ошибка: Таблицы 'moves' вообще нет в базе данных!")
        conn.close()
        return
        
    # 2. Считаем общее количество ходов
    cursor.execute("SELECT COUNT(*) FROM moves")
    total_rows = cursor.fetchone()[0]
    print(f"📊 Всего ходов в базе данных (строк): {total_rows}")
    
    if total_rows == 0:
        print("⚠️ База данных пуста! Вы, скорее всего, забыли запустить pgn_parser.py для наполнения базы.")
        conn.close()
        return

    # 3. Проверяем состояние колонки fen_after
    cursor.execute("PRAGMA table_info(moves)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if "fen_after" not in columns:
        print("ℹ️ Колонки 'fen_after' еще не существует.")
    else:
        cursor.execute("SELECT COUNT(*) FROM moves WHERE fen_after IS NULL OR fen_after = ''")
        empty_rows = cursor.fetchone()[0]
        
        filled_rows = total_rows - empty_rows
        print(f"✅ Уже заполнено FEN-ов (fen_after): {filled_rows}")
        print(f"🔍 Осталось заполнить: {empty_rows}")
        
    conn.close()

if __name__ == "__main__":
    diagnose_database()
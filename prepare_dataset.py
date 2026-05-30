# prepare_dataset.py
import sqlite3
import chess
from tqdm import tqdm

def add_fen_after_column(db_path="chess_bot.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Проверяем, есть ли колонка fen_after
    cursor.execute("PRAGMA table_info(moves)")
    columns = [col[1] for col in cursor.fetchall()]
    if "fen_after" not in columns:
        print("Добавляем колонку fen_after в базу данных...")
        cursor.execute("ALTER TABLE moves ADD COLUMN fen_after TEXT")
        conn.commit()
    
    # 2. Выбираем ходы (проверяем и NULL, и пустые строки)
    print("Проверяем базу данных на наличие нерассчитанных FEN...")
    cursor.execute("""
        SELECT id, fen_before, move_san 
        FROM moves 
        WHERE fen_after IS NULL OR fen_after = ''
    """)
    rows = cursor.fetchall()
    
    if not rows:
        print("Все FEN уже рассчитаны или база пуста!")
        conn.close()
        return

    print(f"Начинаем расчет fen_after для {len(rows)} ходов...")
    
    updates = []
    # Использование tqdm покажет красивый прогресс-бар в консоли
    for row_id, fen_before, move_san in tqdm(rows):
        board = chess.Board(fen_before)
        try:
            move = board.parse_san(move_san)
            board.push(move)
            fen_after = board.fen()
        except Exception:
            fen_after = fen_before  # Фолбэк на случай битых данных
            
        updates.append((fen_after, row_id))
        
        # Записываем пачками по 50 000
        if len(updates) >= 50000:
            cursor.executemany("UPDATE moves SET fen_after = ? WHERE id = ?", updates)
            conn.commit()
            updates = []
            
    if updates:
        cursor.executemany("UPDATE moves SET fen_after = ? WHERE id = ?", updates)
        conn.commit()
        
    print("Создаем индекс для ускорения выборки при обучении...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_moves_game_id ON moves(game_id, id)")
    conn.commit()
    
    conn.close()
    print("Готово! База данных успешно оптимизирована.")

if __name__ == "__main__":
    add_fen_after_column()
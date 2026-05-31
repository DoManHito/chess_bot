import sqlite3
import chess
import chess.pgn
import json

DB_PATH = "chess_bot.db"

def export_self_play_to_pgn(output_filename="self_play_games.pgn"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Проверяем, есть ли вообще данные
    cursor.execute("SELECT DISTINCT game_id FROM self_play_moves")
    game_ids = [row[0] for row in cursor.fetchall()]
    
    if not game_ids:
        print("В базе данных пока нет сыгранных self-play партий.")
        conn.close()
        return

    pgn_file = open(output_filename, "w", encoding="utf-8")
    
    print(f"Экспорт {len(game_ids)} партий в файл {output_filename}...")
    
    for game_id in game_ids:
        # Вытаскиваем все ходы конкретной игры по порядку их генерации
        cursor.execute("""
            SELECT fen_before, move_uci, result_value 
            FROM self_play_moves 
            WHERE game_id = ? 
            ORDER BY id
        """, (game_id,))
        
        moves_data = cursor.fetchall()
        if not moves_data:
            continue
            
        # Создаем PGN структуру
        game = chess.pgn.Game()
        game.headers["Event"] = f"Self-Play AI Session"
        game.headers["Site"] = "Local Bot Environment"
        game.headers["Round"] = str(game_id)
        game.headers["White"] = f"Bot_v1 (MCTS)"
        game.headers["Black"] = f"Bot_v1 (MCTS)"
        
        # Определяем финальный результат по последнему ходу Белых/Черных
        last_move_reward = moves_data[-1][2]
        # Так как reward привязан к ходившему, определим по FEN, чей был ход в конце
        is_white_last = " w " in moves_data[-1][0]
        
        if last_move_reward == 0:
            result_str = "1/2-1/2"
        elif (last_move_reward > 0 and is_white_last) or (last_move_reward < 0 and not is_white_last):
            result_str = "1-0" # Выиграли Белые
        else:
            result_str = "0-1" # Выиграли Черные
            
        game.headers["Result"] = result_str
        
        # Заполняем дерево ходов
        node = game
        board = chess.Board()
        
        for fen_before, move_uci, _ in moves_data:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                node = node.add_main_variation(move)
                board.push(move)
            else:
                # На случай, если что-то рассинхронизировалось
                break
                
        # Записываем партию в файл
        print(game, file=pgn_file, end="\n\n")
        
    pgn_file.close()
    conn.close()
    print("Готово! Можешь загрузить файл на lichess.org/paste")

if __name__ == "__main__":
    export_self_play_to_pgn()
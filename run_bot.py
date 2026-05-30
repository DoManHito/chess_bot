import chess
from classifiers.move_classifier import MoveClassifier
from engine.mcts import MoveClassifierMCTS

def play_against_bot():
    print("Загрузка мозга бота (Нейросеть + MCTS)...")
    classifier = MoveClassifier()
    # Инициализируем наш MCTS движок
    engine = MoveClassifierMCTS(classifier, cpuct=1.5)
    
    board = chess.Board()
    print("\nИгра началась! Вы играете Белыми. Вводите ходы в формате SAN (например: e4, Nf3, d6).")
    
    while not board.is_game_over():
        print("\n" + "="*30)
        print(board)
        print("="*30)
        
        # Ход человека
        if board.turn == chess.WHITE:
            move_ok = False
            while not move_ok:
                try:
                    human_move_str = input("\nВаш ход: ")
                    move = board.parse_san(human_move_str)
                    if move in board.legal_moves:
                        board.push(move)
                        move_ok = True
                    else:
                        print("Нелегальный ход! Попробуйте еще раз.")
                except ValueError:
                    print("Неверный формат хода. Используйте SAN (e4, Nf3...).")
        # Ход бота
        else:
            print("\nБот думает (MCTS симуляции)...")
            # 80-100 симуляций для Python — это комфортно по скорости (~2-4 секунды на ход)
            bot_move = engine.search(board, num_simulations=80) 
            print(f"Бот сыграл: {board.san(bot_move)}")
            board.push(bot_move)
            
    print("\nИгра окончена!")
    print(f"Результат: {board.result()}")

if __name__ == "__main__":
    play_against_bot()
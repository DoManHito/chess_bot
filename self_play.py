import os
import sqlite3
import json
import chess
import numpy as np
import torch
from tqdm import tqdm

from classifiers.move_classifier import MoveClassifier
from engine.mcts import MoveClassifierMCTS, MCTSNode

DB_PATH = "chess_bot.db"

def init_self_play_db():
    """Создает таблицу для сохранения сыгранных ботом партий, если её нет."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS self_play_moves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER,
            fen_before TEXT,
            move_uci TEXT,
            mcts_policy TEXT, -- Распределение визитов MCTS в формате JSON
            result_value REAL  -- Конечный результат игры для этого хода (-1.0, 0.0, 1.0)
        )
    """)
    conn.commit()
    conn.close()

def get_next_game_id():
    """Возвращает уникальный ID для новой сессии self-play."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(game_id) FROM self_play_moves")
    row = cursor.fetchone()
    conn.close()
    return (row[0] + 1) if row[0] is not None else 1

def run_self_play_session(num_games: int = 10, num_simulations: int = 80, temperature: float = 1.0):
    """
    Запускает генерацию партий бота с самим собой.
    
    :param num_games: Сколько игр сыграть.
    :param num_simulations: Количество симуляций MCTS на каждый ход (больше -> сильнее игра).
    :param temperature: Контролирует случайность (1.0 в начале партии для разнообразия, 
                        0.0 в эндшпиле для выбора строго лучшего хода).
    """
    init_self_play_db()
    
    print("Загрузка классификатора для MCTS...")
    classifier = MoveClassifier(weights_path="models/weights_classifier.pth")
    # Используем немного увеличенный cpuct (например, 2.0) для лучшего исследования в селф-плей
    engine = MoveClassifierMCTS(classifier=classifier, cpuct=2.0)
    
    current_game_id = get_next_game_id()
    
    for game_idx in range(num_games):
        print(f"\n--- Запуск партии Self-Play {game_idx + 1}/{num_games} (Game ID: {current_game_id}) ---")
        board = chess.Board()
        
        # История ходов текущей партии для последующей записи результата
        game_history = []
        
        # Ограничитель на случай бесконечных ничьих
        max_moves = 200
        move_count = 0
        
        with tqdm(total=max_moves, desc="Ходы в партии") as pbar:
            while not board.is_game_over() and move_count < max_moves:
                # 1. Запускаем MCTS поиск из текущего состояния
                # Нам нужен корневой узел, чтобы вытащить распределение посещений (Priors/Visits)
                root = MCTSNode(board=board.copy())
                engine._evaluate_and_expand_node(root)
                
                # Проводим симуляции
                for _ in range(num_simulations):
                    node = root
                    # Спуск
                    while node.is_expanded and node.board.legal_moves:
                        legal_moves = list(node.board.legal_moves)
                        if not node.priors:
                            engine._evaluate_and_expand_node(node)
                        best_move = max(legal_moves, key=lambda m: node.get_ucb_score(m, engine.cpuct))
                        if best_move not in node.children:
                            next_board = node.board.copy()
                            next_board.push(best_move)
                            node.children[best_move] = MCTSNode(board=next_board, parent=node, move=best_move)
                        node = node.children[best_move]
                    
                    # Оценка листа
                    value = engine._evaluate_and_expand_node(node)
                    if node.board.turn != board.turn:
                        value = -value
                    
                    # Бэкпроп
                    while node is not None:
                        node.visit_count += 1
                        node.total_value += value
                        value = -value
                        node = node.parent

                # 2. Собираем статистику визитов ходов
                legal_moves = list(board.legal_moves)
                visits = np.array([root.children[m].visit_count if m in root.children else 0 for m in legal_moves], dtype=np.float32)
                
                if visits.sum() == 0:
                    # Если симуляции не посетили никого (редко), делаем равномерное распределение
                    policy = np.ones(len(legal_moves)) / len(legal_moves)
                else:
                    # Применяем температуру для добавления стохастичности
                    if move_count < 15 and temperature > 0:
                        # В дебюте играем разнообразно
                        policy_counts = visits ** (1.0 / temperature)
                        policy = policy_counts / policy_counts.sum()
                    else:
                        # В миттельшпиле/эндшпиле выбираем строго самый посещаемый ход
                        best_idx = np.argmax(visits)
                        policy = np.zeros_like(visits)
                        policy[best_idx] = 1.0

                # 3. Выбираем ход на основе полученных вероятностей
                chosen_move = np.random.choice(legal_moves, p=policy)
                
                # Создаем словарь распределения ходов: {"e2e4": 0.7, "g1f3": 0.3}
                policy_dict = {move.uci(): float(prob) for move, prob in zip(legal_moves, policy)}
                
                # Сохраняем промежуточные данные хода
                game_history.append({
                    "fen_before": board.fen(),
                    "move_uci": chosen_move.uci(),
                    "mcts_policy": json.dumps(policy_dict),
                    "turn": board.turn # Запоминаем, чей это был ход (True = Белые, False = Черные)
                })
                
                # Делаем ход на доске
                board.push(chosen_move)
                move_count += 1
                pbar.update(1)
        
        # 4. Партия завершена, определяем результат
        result = board.result()
        print(f"Результат партии: {result}")
        
        # Конвертируем результат в очки для Белых
        if result == "1-0":
            game_reward = 1.0
        elif result == "0-1":
            game_reward = -1.0
        else:
            game_reward = 0.0 # Ничья или превышение лимита ходов
            
        # 5. Записываем всю партию в базу данных с правильными знаками наград
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        for h in game_history:
            # Если ходили Белые, награда идет "как есть". Если Черные — инвертируется.
            move_reward = game_reward if h["turn"] == chess.WHITE else -game_reward
            
            cursor.execute("""
                INSERT INTO self_play_moves (game_id, fen_before, move_uci, mcts_policy, result_value)
                VALUES (?, ?, ?, ?, ?)
            """, (current_game_id, h["fen_before"], h["move_uci"], h["mcts_policy"], move_reward))
            
        conn.commit()
        conn.close()
        
        current_game_id += 1

if __name__ == "__main__":
    # Запустим генерацию, например, 5 тестовых партий. 
    # Можешь поставить 100 или 1000 партий, когда будешь готов генерировать большую базу.
    run_self_play_session(num_games=5, num_simulations=60, temperature=1.0)
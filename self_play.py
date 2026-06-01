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
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS self_play_moves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER,
            fen_before TEXT,
            move_uci TEXT,
            mcts_policy TEXT,
            result_value REAL
        )
    """)
    conn.commit()
    conn.close()

def get_next_game_id():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(game_id) FROM self_play_moves")
    row = cursor.fetchone()
    conn.close()
    return (row[0] + 1) if row[0] is not None else 1

def run_self_play_session(num_games: int = 10, num_simulations: int = 80, temperature: float = 1.0):
    init_self_play_db()
    
    print("Загрузка классификатора для MCTS...")
    classifier = MoveClassifier(weights_path="models/weights_classifier.pth")
    engine = MoveClassifierMCTS(classifier=classifier, cpuct=2.0)
    
    current_game_id = get_next_game_id()
    
    for game_idx in range(num_games):
        print(f"\n--- Запуск партии Self-Play {game_idx + 1}/{num_games} (Game ID: {current_game_id}) ---")
        board = chess.Board()
        
        game_history = []
        
        max_moves = 200
        move_count = 0
        
        with tqdm(total=max_moves, desc="Ходы в партии") as pbar:
            while not board.is_game_over() and move_count < max_moves:
                root = MCTSNode(board=board.copy())
                engine._evaluate_and_expand_node(root)
                
                for _ in range(num_simulations):
                    node = root
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
                    
                    value = engine._evaluate_and_expand_node(node)
                    if node.board.turn != board.turn:
                        value = -value
                    
                    while node is not None:
                        node.visit_count += 1
                        node.total_value += value
                        value = -value
                        node = node.parent

                legal_moves = list(board.legal_moves)
                visits = np.array([root.children[m].visit_count if m in root.children else 0 for m in legal_moves], dtype=np.float32)
                
                if visits.sum() == 0:
                    policy = np.ones(len(legal_moves)) / len(legal_moves)
                else:
                    if move_count < 15 and temperature > 0:
                        policy_counts = visits ** (1.0 / temperature)
                        policy = policy_counts / policy_counts.sum()
                    else:
                        best_idx = np.argmax(visits)
                        policy = np.zeros_like(visits)
                        policy[best_idx] = 1.0

                chosen_move = np.random.choice(legal_moves, p=policy)
                
                policy_dict = {move.uci(): float(prob) for move, prob in zip(legal_moves, policy)}
                
                game_history.append({
                    "fen_before": board.fen(),
                    "move_uci": chosen_move.uci(),
                    "mcts_policy": json.dumps(policy_dict),
                    "turn": board.turn
                })
                
                board.push(chosen_move)
                move_count += 1
                pbar.update(1)
        
        result = board.result()
        print(f"Результат партии: {result}")
        
        if result == "1-0":
            game_reward = 1.0
        elif result == "0-1":
            game_reward = -1.0
        else:
            game_reward = 0.0
            
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        for h in game_history:
            move_reward = game_reward if h["turn"] == chess.WHITE else -game_reward
            
            cursor.execute("""
                INSERT INTO self_play_moves (game_id, fen_before, move_uci, mcts_policy, result_value)
                VALUES (?, ?, ?, ?, ?)
            """, (current_game_id, h["fen_before"], h["move_uci"], h["mcts_policy"], move_reward))
            
        conn.commit()
        conn.close()
        
        current_game_id += 1

if __name__ == "__main__":
    run_self_play_session(num_games=5, num_simulations=60, temperature=1.0)
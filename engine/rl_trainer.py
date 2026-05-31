import os
import sqlite3
import json
import chess
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from classifiers.move_classifier import MoveClassifier
from engine.mcts import MoveClassifierMCTS
from models.chess_nets import ChessCoreNet, MoveClassifierNet
from classifiers.classification_config import CLASS_NAMES
from classifiers.self_play_dataset import ChessSelfPlayDataset

def init_self_play_db(db_path: str):
    conn = sqlite3.connect(db_path)
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

def get_next_game_id(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(game_id) FROM self_play_moves")
    row = cursor.fetchone()
    conn.close()
    return (row[0] + 1) if row[0] is not None else 1


# ----- Функция для параллельной игры одной партии -----
def play_one_game_parallel(game_id, bot_weights, num_simulations, temperature, max_moves=200):
    # КРИТИЧЕСКИ ВАЖНО: отключаем внутренний OpenMP/MKL внутри подпроцессов, чтобы убрать микрозависания CPU
    torch.set_num_threads(1)
    
    classifier = MoveClassifier(weights_path=bot_weights, device="cpu")
    engine = MoveClassifierMCTS(classifier=classifier, cpuct=1.5)
    
    board = chess.Board()
    game_history = []
    move_count = 0
    
    while not board.is_game_over() and move_count < max_moves:
        chosen_move, visit_dict = engine.search(board, num_simulations=num_simulations)
        if chosen_move is None:
            break
        
        visits = np.array(list(visit_dict.values()), dtype=np.float32)
        if visits.sum() == 0:
            policy = np.ones(len(visits)) / len(visits)
        else:
            if temperature > 0:
                policy_counts = visits ** (1.0 / max(temperature, 0.1))
                policy = policy_counts / policy_counts.sum()
            else:
                best_idx = np.argmax(visits)
                policy = np.zeros_like(visits)
                policy[best_idx] = 1.0
        
        policy_dict = {move_uci: float(prob) for move_uci, prob in zip(visit_dict.keys(), policy)}
        game_history.append({
            "fen_before": board.fen(),
            "move_uci": chosen_move.uci(),
            "mcts_policy": json.dumps(policy_dict),
            "turn": board.turn
        })
        board.push(chosen_move)
        move_count += 1
    
    result = board.result()
    if result == "1-0":
        game_reward = 1.0
    elif result == "0-1":
        game_reward = -1.0
    else:
        game_reward = 0.0
    
    for h in game_history:
        h["result_value"] = game_reward if h["turn"] == chess.WHITE else -game_reward
    
    return game_id, game_history


# ----- Генерация партий -----
def run_self_play_session(num_games: int, num_simulations: int, temperature: float,
                          db_path: str, num_workers: int = 1):
    print("\n" + "-"*40)
    print(f"🎮 ФАЗА 1: ГЕНЕРАЦИЯ ПАРТИЙ SELF-PLAY ({num_games} игр)")
    print("-"*40)
    
    bot_weights = "models/weights_bot.pth"
    if not os.path.exists(bot_weights):
        os.makedirs("models", exist_ok=True)
        if os.path.exists("models/weights_classifier.pth"):
            print("🌱 Копируем базовый классификатор как стартовую точку бота...")
            shutil.copy("models/weights_classifier.pth", bot_weights)
        else:
            print("⚠️ Веса бота не найдены! Бот начнет со случайных весов.")
    
    current_game_id = get_next_game_id(db_path)
    game_ids = list(range(current_game_id, current_game_id + num_games))
    
    if num_workers <= 1:
        torch.set_num_threads(1)
        classifier = MoveClassifier(weights_path=bot_weights, device="cpu")
        engine = MoveClassifierMCTS(classifier=classifier, cpuct=1.5)
        
        for idx, gid in enumerate(game_ids):
            board = chess.Board()
            game_history = []
            move_count = 0
            max_moves = 200
            
            with tqdm(total=max_moves, desc=f"Партия {idx+1}/{num_games} (ID: {gid})", leave=False) as pbar:
                while not board.is_game_over() and move_count < max_moves:
                    chosen_move, visit_dict = engine.search(board, num_simulations=num_simulations)
                    if chosen_move is None:
                        break
                    
                    visits = np.array(list(visit_dict.values()), dtype=np.float32)
                    if visits.sum() == 0:
                        policy = np.ones(len(visits)) / len(visits)
                    else:
                        if temperature > 0:
                            policy_counts = visits ** (1.0 / max(temperature, 0.1))
                            policy = policy_counts / policy_counts.sum()
                        else:
                            best_idx = np.argmax(visits)
                            policy = np.zeros_like(visits)
                            policy[best_idx] = 1.0
                    
                    policy_dict = {move_uci: float(prob) for move_uci, prob in zip(visit_dict.keys(), policy)}
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
            if result == "1-0": game_reward = 1.0
            elif result == "0-1": game_reward = -1.0
            else: game_reward = 0.0
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            for h in game_history:
                move_reward = game_reward if h["turn"] == chess.WHITE else -game_reward
                cursor.execute("""
                    INSERT INTO self_play_moves (game_id, fen_before, move_uci, mcts_policy, result_value)
                    VALUES (?, ?, ?, ?, ?)
                """, (gid, h["fen_before"], h["move_uci"], h["mcts_policy"], move_reward))
            conn.commit()
            conn.close()
        
        print(f"✅ {num_games} партий сгенерированы последовательно.")
        return
    
    print(f"🚀 Запуск {num_workers} процессов для генерации {num_games} партий...")
    play_func = partial(play_one_game_parallel,
                        bot_weights=bot_weights,
                        num_simulations=num_simulations,
                        temperature=temperature)
    
    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(play_func, [(gid,) for gid in game_ids])
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for game_id, game_history in results:
        for h in game_history:
            cursor.execute("""
                INSERT INTO self_play_moves (game_id, fen_before, move_uci, mcts_policy, result_value)
                VALUES (?, ?, ?, ?, ?)
            """, (game_id, h["fen_before"], h["move_uci"], h["mcts_policy"], h["result_value"]))
    conn.commit()
    conn.close()
    print(f"✅ Все {num_games} партий сгенерированы и сохранены.")


# ----- Очистка старых партий -----
def clean_old_games(db_path: str, keep_last_n: int):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='self_play_moves'")
    if not cursor.fetchone():
        conn.close()
        return
    cursor.execute("SELECT DISTINCT game_id FROM self_play_moves ORDER BY game_id DESC")
    games = [row[0] for row in cursor.fetchall()]
    if len(games) > keep_last_n:
        games_to_delete = games[keep_last_n:]
        print(f"🧹 Sliding Window: Удаляем {len(games_to_delete)} старых партий.")
        placeholders = ",".join("?" for _ in games_to_delete)
        cursor.execute(f"DELETE FROM self_play_moves WHERE game_id IN ({placeholders})", games_to_delete)
        conn.commit()
    conn.close()


# ----- Обучение бота (на GPU) -----
def train_rl_iteration(epochs: int, batch_size: int, lr: float, alpha: float, db_path: str):
    print("\n" + "-"*40)
    print("🧠 ФАЗА 2: ОБУЧЕНИЕ НЕЙРОСЕТИ БОТА (RL Update)")
    print("-"*40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ChessSelfPlayDataset(db_path=db_path)
    print(f"📡 ЦЕЛЕВОЙ ДЕВАЙС ДЛЯ ФАЗЫ ОБУЧЕНИЯ: {device}")
    
    # 2. Передаем девайс в датасет
    dataset = ChessSelfPlayDataset(db_path=db_path, device=str(device))
    
    if len(dataset) < 50:
        return

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    num_loaders = 0 if str(device) == "cuda" else 2
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_loaders,
        pin_memory=(str(device) == "cuda")
    )
    

    core = ChessCoreNet(in_channels=25)
    model = MoveClassifierNet(core_net=core, num_classes=len(CLASS_NAMES))
    
    weights_path = "models/weights_bot.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    
    model.to(device)
    
    criterion_class = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, class_targets, value_targets in train_loader:
            inputs, class_targets, value_targets = inputs.to(device), class_targets.to(device), value_targets.to(device)
            optimizer.zero_grad()
            
            class_logits, value_preds = model(inputs)
            
            loss_class = criterion_class(class_logits, class_targets)
            # Жесткое выравнивание через .view(-1) защищает от багов несовпадения скаляров/векторов
            loss_value = criterion_value(value_preds.view(-1), value_targets.view(-1))
            
            loss = loss_class + alpha * loss_value
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"   Эпоха {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), weights_path)
    print(f"✅ Веса БОТА успешно обновлены и сохранены в {weights_path}")


# ----- Логирование метрик -----
def log_iteration_metrics(db_path: str, iteration_num: int):
    from collections import Counter
    evaluator = MoveClassifier(weights_path="models/weights_classifier.pth", device="cpu")
    evaluator.model.eval()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT fen_before, move_uci FROM self_play_moves")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return

    stats = Counter()
    print(f"\n📊 ВНЕШНИЙ АНАЛИЗ КАЧЕСТВА ИГРЫ БОТА НА ИТЕРАЦИИ {iteration_num}:")
    
    for fen, move_uci in rows:
        try:
            board = chess.Board(fen)
            move_san = board.san(chess.Move.from_uci(move_uci))
            res, _ = evaluator.classify_move(fen, move_san)
            stats[res.classification] += 1
        except Exception:
            continue
            
    total_moves = sum(stats.values())
    if total_moves == 0:
        return
        
    print(f"Всего ходов проанализировано: {total_moves}")
    for class_name in CLASS_NAMES:
        count = stats[class_name]
        percentage = (count / total_moves) * 100
        print(f"  - {class_name}: {count} ({percentage:.2f}%)")
    print("-" * 50)


# ----- Главный цикл обучения -----
def run_continuous_loop(iterations: int, games_per_iter: int, sims: int, epochs: int,
                        keep_last_n: int, db_path: str, temperature: float = 1.2,
                        num_workers: int = 1):
    init_self_play_db(db_path)
    print(f"\n🚀 ЗАПУСК ЦИКЛА ОБУЧЕНИЯ НА {iterations} ИТЕРАЦИЙ (workers={num_workers})")
    for i in range(iterations):
        print(f"\n{'='*50}\n🌟 ГЛОБАЛЬНАЯ ИТЕРАЦИЯ {i+1}/{iterations}\n{'='*50}")
        run_self_play_session(num_games=games_per_iter, num_simulations=sims,
                              temperature=temperature, db_path=db_path,
                              num_workers=num_workers)
        log_iteration_metrics(db_path, i+1)
        train_rl_iteration(epochs=epochs, batch_size=64, lr=0.0005, alpha=0.5, db_path=db_path)
        clean_old_games(db_path=db_path, keep_last_n=keep_last_n)
    print("\n🎉 Цикл обучения успешно завершен!")
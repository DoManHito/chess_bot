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


# ----- Функция для параллельной игры одной партии (С ПРЕД-РАЗМЕТКОЙ КЛАССОВ) -----
def play_one_game_parallel(game_id, bot_weights, num_simulations, temperature, db_path):
    # Каждому процессу — свой чистый коннект к SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Инициализируем локальные утилиты на CPU для генерации ходов
    evaluator = MoveClassifier(weights_path=bot_weights, device="cpu")
    evaluator.model.eval()
    mcts = MoveClassifierMCTS(classifier=evaluator)

    board = chess.Board()
    game_history = []  # Хранит кортежи: (fen, move_uci, mcts_policy_dict)

    while not board.is_game_over() and len(game_history) < 300:
        fen_before = board.fen()
        
        # Запускаем MCTS симуляцию (возвращает visit_dict с количеством посещений)
        _, visit_dict = mcts.search(board, num_simulations=num_simulations)
        if not visit_dict:
            break

        moves_uci_list = list(visit_dict.keys())
        mcts_visits = np.array(list(visit_dict.values()), dtype=np.float32)

        # --- РЕАЛИЗАЦИЯ ТЕМПЕРАТУРЫ (EXPLORATION) ---
        if temperature > 0:
            exp_visits = mcts_visits ** (1.0 / (temperature + 1e-8))
            sum_exp = exp_visits.sum()
            move_probabilities = exp_visits / sum_exp if sum_exp > 0 else np.ones_like(exp_visits) / len(exp_visits)
            
            # Случайно выбираем ход на основе распределения вероятностей
            chosen_move_uci = np.random.choice(moves_uci_list, p=move_probabilities)
            move = chess.Move.from_uci(chosen_move_uci)
        else:
            # Если температура 0, выбираем строго самый посещаемый ход (жадный выбор)
            best_idx = np.argmax(mcts_visits)
            move = chess.Move.from_uci(moves_uci_list[best_idx])

        # --- Размечаем классы ходов ОДИН РАЗ прямо здесь для базы данных ---
        # Для обучения сети нам нужны нормализованные вероятности на основе сырых посещений MCTS
        total_visits = mcts_visits.sum()
        mcts_probs = mcts_visits / total_visits if total_visits > 0 else np.ones_like(mcts_visits) / len(mcts_visits)
        
        moves_san = []
        valid_indices = []
        class_target_distribution = np.zeros(len(CLASS_NAMES), dtype=np.float32)

        for idx, mu in enumerate(moves_uci_list):
            try:
                m_obj = chess.Move.from_uci(mu)
                moves_san.append(board.san(m_obj))
                valid_indices.append(idx)
            except Exception:
                class_target_distribution[0] += mcts_probs[idx]

        if moves_san:
            classes, _, _ = evaluator.classify_moves_batch(board, moves_san)
            for cls, idx in zip(classes, valid_indices):
                c_idx = CLASS_NAMES.index(cls) if cls in CLASS_NAMES else 0
                class_target_distribution[c_idx] += mcts_probs[idx]

        sum_dist = class_target_distribution.sum()
        if sum_dist > 0:
            class_target_distribution /= sum_dist
        else:
            class_target_distribution = np.ones(len(CLASS_NAMES), dtype=np.float32) / len(CLASS_NAMES)

        class_policy_dict = {CLASS_NAMES[i]: float(class_target_distribution[i]) for i in range(len(CLASS_NAMES))}
        
        game_history.append((fen_before, move.uci(), class_policy_dict))
        board.push(move)

    # Определяем результат игры
    result = board.result()
    if result == "1-0":
        value_white = 1.0
    elif result == "0-1":
        value_white = -1.0
    else:
        value_white = 0.0

    # Сохраняем всю партию в базу данных
    for fen_before, move_uci, class_policy_dict in game_history:
        # Для кого ход?
        current_board = chess.Board(fen_before)
        actual_value = value_white if current_board.turn == chess.WHITE else -value_white
        
        cursor.execute("""
            INSERT INTO self_play_moves (game_id, fen_before, move_uci, mcts_policy, result_value)
            VALUES (?, ?, ?, ?, ?)
        """, (game_id, fen_before, move_uci, json.dumps(class_policy_dict), actual_value))

    conn.commit()
    conn.close()


def run_self_play_session(num_games: int, num_simulations: int, temperature: float,
                          db_path: str, bot_weights: str, num_workers: int):
    print(f"\n----------------------------------------")
    print(f"🎮 ФАЗА 1: ГЕНЕРАЦИЯ ПАРТИЙ SELF-PLAY ({num_games} игр)")
    print(f"----------------------------------------")
    print(f"🚀 Запуск {num_workers} процессов для генерации {num_games} партий...")

    start_game_id = get_next_game_id(db_path)
    game_ids = list(range(start_game_id, start_game_id + num_games))

    if num_workers <= 1:
        for gid in tqdm(game_ids, desc="Синхронная генерация"):
            play_one_game_parallel(gid, bot_weights, num_simulations, temperature, db_path)
    else:
        # Пул процессов для CPU-мультипроцессинга
        worker_func = partial(play_one_game_parallel, bot_weights=bot_weights,
                              num_simulations=num_simulations, temperature=temperature, db_path=db_path)
        with mp.Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap_unordered(worker_func, game_ids), total=num_games, desc="Генерация партий"))

    print(f"✅ Все {num_games} партий сгенерированы и сохранены.")


def train_rl_iteration(epochs: int, batch_size: int, lr: float, alpha: float, db_path: str):
    print("\n" + "-"*40)
    print("🧠 ФАЗА 2: ОБУЧЕНИЕ НЕЙРОСЕТИ БОТА (RL Update)")
    print("-"*40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📡 ЦЕЛЕВОЙ ДЕВАЙС ДЛЯ ФАЗЫ ОБУЧЕНИЯ: {device}")
    if device.type == "cuda":
        print(f"🔥 Обучение ускорено на GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    print("-" * 40)

    # Инициализируем наш теперь уже ультра-легкий датасет
    dataset = ChessSelfPlayDataset(db_path=db_path)
    
    if len(dataset) < 50:
        print("⚠️ Слишком мало данных для обучения. Пропускаем...")
        return

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Теперь, когда __getitem__ мгновенный, мы можем безопасно выставить num_workers=2 или 4, 
    # не боясь зависаний или CUDA ошибок, так как внутри датасета БОЛЬШЕ НЕТ нейросетей.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=(device.type == "cuda")
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
        
        progress_bar = tqdm(train_loader, desc=f"   Эпоха {epoch+1}/{epochs}", leave=True)
        for inputs, class_targets, value_targets in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            class_targets = class_targets.to(device, non_blocking=True)
            value_targets = value_targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            class_logits, value_preds = model(inputs)
            
            loss_class = criterion_class(class_logits, class_targets)
            loss_value = criterion_value(value_preds.view(-1), value_targets.view(-1))
            
            loss = loss_class + alpha * loss_value
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    torch.save(model.state_dict(), weights_path)
    print(f"✅ Веса БОТА успешно обновлены и сохранены в {weights_path}")


def apply_sliding_window(db_path: str, keep_last_n_games: int):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT game_id FROM self_play_moves ORDER BY game_id DESC")
    rows = cursor.fetchall()
    
    if len(rows) > keep_last_n_games:
        last_allowed_id = rows[keep_last_n_games - 1][0]
        cursor.execute("DELETE FROM self_play_moves WHERE game_id < ?", (last_allowed_id,))
        deleted_rows = cursor.rowcount
        conn.commit()
        if deleted_rows > 0:
            print(f"🧹 Sliding Window: Удалено {deleted_rows} старых позиций (сохраняем топ-{keep_last_n_games} игр).")
    conn.close()


def run_continuous_loop(iterations: int, games_per_iter: int, sims: int, epochs: int,
                        keep_last_n: int, db_path: str, temperature: float = 1.2,
                        num_workers: int = 1):
    init_self_play_db(db_path)
    print(f"\n🚀 ЗАПУСК ЦИКЛА ОБУЧЕНИЯ НА {iterations} ИТЕРАЦИЙ (workers={num_workers})")
    
    # Пути весов
    classifier_weights = "models/weights_classifier.pth"
    bot_weights = "models/weights_bot.pth"
    
    if not os.path.exists(bot_weights):
        if os.path.exists(classifier_weights):
            shutil.copy(classifier_weights, bot_weights)
            print(f"📦 Базовые веса скопированы из {classifier_weights} в {bot_weights}")
        else:
            print(f"⚠️ Внимание: {bot_weights} не найден. Обучение начнется с нуля!")

    for i in range(iterations):
        print(f"\n{'='*50}\n🌟 ГЛОБАЛЬНАЯ ИТЕРАЦИЯ {i+1}/{iterations}\n{'='*50}")
        
        # 1. Генерируем партии (Классификатор вызывается ТУТ на CPU один раз в параллельных процессах)
        run_self_play_session(num_games=games_per_iter, num_simulations=sims,
                              temperature=temperature, db_path=db_path,
                              bot_weights=bot_weights, num_workers=num_workers)
        
        # 2. Обучаем сеть на GPU (Мгновенное чтение данных без вызова нейросетей внутри даталоадера!)
        train_rl_iteration(epochs=epochs, batch_size=64, lr=1e-4, alpha=1.0, db_path=db_path)
        
        # 3. Скользящее окно
        apply_sliding_window(db_path, keep_last_n)
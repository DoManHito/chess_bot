import os
import sqlite3
import json
import chess
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import torch.nn.functional as F

from classifiers.move_classifier import MoveClassifier, MoveData
from engine.unified_mcts import UnifiedMCTS
from models.unified_chess_nets import ChessCoreNet, UnifiedMoveClassifierNet
from classifiers.classification_config import CLASS_NAMES
from classifiers.self_play_dataset import ChessSelfPlayDataset, LookaheadChessSelfPlayDataset

_orig_load_state_dict = nn.Module.load_state_dict

def _safe_load_state_dict(self, state_dict, strict=False):
    model_dict = self.state_dict()
    safe_checkpoint = {}
    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                safe_checkpoint[k] = v
            else:
                print(f"⚠️ Слой '{k}' временно пропущен/адаптирован из-за изменения размеров: {v.shape} -> {model_dict[k].shape}")
        elif strict:
            safe_checkpoint[k] = v
    return _orig_load_state_dict(self, safe_checkpoint, strict=False)

nn.Module.load_state_dict = _safe_load_state_dict


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
            result_value REAL,
            lookahead_depth INTEGER,
            future_moves TEXT,
            final_classification TEXT,
            move_sequence_classes TEXT
        )
    """)
    conn.commit()
    conn.close()


def get_next_game_id(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(game_id) FROM self_play_moves")
    row = cursor.fetchone()
    max_id = row[0] if row and row[0] is not None else 0
    conn.close()
    return max_id + 1


def generate_lookahead_sequences(board: chess.Board, root_moves_uci: list, root_mcts_probs: list, lookahead_depth: int, evaluator: MoveClassifier):
    if lookahead_depth <= 0:
        return None

    current_board = board.copy()
    sequence_moves = []
    sequence_classes = []

    if root_moves_uci and len(root_mcts_probs) == len(root_moves_uci):
        chosen_uci = np.random.choice(root_moves_uci, p=root_mcts_probs)
        move = chess.Move.from_uci(chosen_uci)
    else:
        legal_moves = list(current_board.legal_moves)
        if not legal_moves:
            return None
        move = random.choice(legal_moves)
        chosen_uci = move.uci()

    turn_label = "White" if current_board.turn == chess.WHITE else "Black"
    move_data = MoveData(board_fen=current_board.fen(), move_san=current_board.san(move), 
                         turn_num=current_board.fullmove_number, turn_label=turn_label)
    try:
        res = evaluator.classify_moves_batch([move_data])
        cls = res[0].classification if res else "Good"
    except:
        cls = "Good"

    sequence_moves.append(chosen_uci)
    sequence_classes.append(cls)
    current_board.push(move)

    for _ in range(1, lookahead_depth):
        if current_board.is_game_over():
            break
        legal_moves = list(current_board.legal_moves)
        if not legal_moves:
            break

        move = random.choice(legal_moves)
        chosen_uci = move.uci()

        turn_label = "White" if current_board.turn == chess.WHITE else "Black"
        move_data = MoveData(board_fen=current_board.fen(), move_san=current_board.san(move), 
                             turn_num=current_board.fullmove_number, turn_label=turn_label)
        try:
            res = evaluator.classify_moves_batch([move_data])
            cls = res[0].classification if res else "Good"
        except:
            cls = "Good"

        sequence_moves.append(chosen_uci)
        sequence_classes.append(cls)
        current_board.push(move)

    return {
        'lookahead_depth': len(sequence_moves),
        'future_moves': sequence_moves,
        'final_classification': sequence_classes[-1] if sequence_classes else "Good",
        'move_sequence_classes': sequence_classes
    }


def play_one_game_with_lookahead(game_id: int, bot_weights: str, num_simulations: int, temperature: float, db_path: str, lookahead_depth: int = 2):
    in_channels = 13
    if os.path.exists(bot_weights):
        try:
            ckpt = torch.load(bot_weights, map_location="cpu")
            for key in ["core.conv_init.weight", "core_net.conv_init.weight"]:
                if key in ckpt:
                    in_channels = ckpt[key].shape[1]
                    break
        except Exception as e:
            print(f"⚠️ Не удалось прочитать количество каналов из весов: {e}")

    evaluator = MoveClassifier(weights_path=bot_weights, device="cpu")
    
    if hasattr(evaluator, 'model') and evaluator.model is not None:
        current_channels = 13
        for layer in evaluator.model.modules():
            if isinstance(layer, nn.Conv2d):
                current_channels = layer.in_channels
                break
        
        if current_channels != in_channels:
            print(f"🔄 Адаптация внутренней модели MoveClassifier под {in_channels} каналов...")
            if hasattr(evaluator.model, 'core_net'):
                evaluator.model.core_net = ChessCoreNet(in_channels=in_channels)
            elif hasattr(evaluator.model, 'core'):
                evaluator.model.core = ChessCoreNet(in_channels=in_channels)
            
            try:
                checkpoint = torch.load(bot_weights, map_location="cpu")
                model_dict = evaluator.model.state_dict()
                safe_checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
                evaluator.model.load_state_dict(safe_checkpoint, strict=False)
            except Exception as e:
                print(f"⚠️ Ошибка повторной адаптации весов: {e}")

    evaluator.model.eval()

    mcts = UnifiedMCTS(unified_model=evaluator, cpuct=2.0, max_simulations=num_simulations, top_moves_ratio=0.3, policy_output_dim=64)

    board = chess.Board()
    game_history = []

    while not board.is_game_over() and len(game_history) < 250:
        fen_before = board.fen()

        _, visit_dict = mcts.search(board, num_simulations=num_simulations, temperature=temperature)
        if not visit_dict:
            break

        moves_uci_list = list(visit_dict.keys())
        mcts_visits = np.array(list(visit_dict.values()), dtype=np.float32)
        total_visits = mcts_visits.sum()

        if total_visits > 0:
            mcts_probs = mcts_visits / total_visits
        else:
            mcts_probs = np.ones_like(mcts_visits) / len(mcts_visits)

        if temperature > 0:
            exp_visits = mcts_visits ** (1.0 / (temperature + 1e-8))
            sum_exp = exp_visits.sum()
            move_probabilities = exp_visits / sum_exp if sum_exp > 0 else np.ones_like(exp_visits) / len(exp_visits)
            chosen_move_uci = np.random.choice(moves_uci_list, p=move_probabilities)
            move = chess.Move.from_uci(chosen_move_uci)
        else:
            best_idx = np.argmax(mcts_visits)
            chosen_move_uci = moves_uci_list[best_idx]
            move = chess.Move.from_uci(chosen_move_uci)

        mcts_policy_dict = {m: float(p) for m, p in zip(moves_uci_list, mcts_probs)}

        lookahead_data = generate_lookahead_sequences(board, moves_uci_list, list(mcts_probs), lookahead_depth, evaluator)

        game_history.append({
            'fen_before': fen_before,
            'move_uci': move.uci(),
            'mcts_policy_dict': mcts_policy_dict,
            'lookahead_data': lookahead_data,
            'turn': board.turn
        })
        board.push(move)

    result = board.result()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    value_white = 0.0
    if result == "1-0":
        value_white = 1.0
    elif result == "0-1":
        value_white = -1.0

    for history_entry in game_history:
        fen_before = history_entry['fen_before']
        move_uci = history_entry['move_uci']
        mcts_policy_dict = history_entry['mcts_policy_dict']
        lookahead_data = history_entry['lookahead_data']
        turn = history_entry['turn']

        actual_value = value_white if turn == chess.WHITE else -value_white

        if lookahead_data:
            cursor.execute("""
                INSERT INTO self_play_moves (game_id, fen_before, move_uci, mcts_policy, result_value,
                                            lookahead_depth, future_moves, final_classification, move_sequence_classes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (game_id, fen_before, move_uci, json.dumps(mcts_policy_dict), actual_value,
                  lookahead_data['lookahead_depth'], json.dumps(lookahead_data['future_moves']),
                  lookahead_data['final_classification'], json.dumps(lookahead_data['move_sequence_classes'])))
        else:
            cursor.execute("""
                INSERT INTO self_play_moves (game_id, fen_before, move_uci, mcts_policy, result_value)
                VALUES (?, ?, ?, ?, ?)
            """, (game_id, fen_before, move_uci, json.dumps(mcts_policy_dict), actual_value))

    conn.commit()
    conn.close()


def run_self_play_session(num_games: int, num_simulations: int, temperature: float, db_path: str, bot_weights: str, num_workers: int = 1, lookahead_depth: int = 0):
    print(f"🎬 Generating {num_games} self-play games using {num_workers} processes...")
    init_self_play_db(db_path)
    start_game_id = get_next_game_id(db_path)

    if num_workers <= 1:
        for i in tqdm(range(num_games), desc="Self-play games"):
            play_one_game_with_lookahead(start_game_id + i, bot_weights, num_simulations, temperature, db_path, lookahead_depth)
    else:
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(num_workers)
        worker_fn = partial(play_one_game_with_lookahead, bot_weights=bot_weights, num_simulations=num_simulations,
                            temperature=temperature, db_path=db_path, lookahead_depth=lookahead_depth)
        
        game_ids = [start_game_id + i for i in range(num_games)]
        list(tqdm(pool.imap_unordered(worker_fn, game_ids), total=num_games, desc="Parallel Self-play"))
        pool.close()
        pool.join()


def train_unified_model(epochs: int, batch_size: int, lr: float, alpha: float, policy_weight: float, db_path: str, use_lookahead: bool = False, use_stockfish_policy: bool = False):
    print("\n" + "-"*50)
    print("🧠 PHASE 2: UNIFIED REINFORCEMENT LEARNING (OPTION A)")
    print("-" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📡 Target Training Device: {device}")

    if use_lookahead:
        dataset = LookaheadChessSelfPlayDataset(db_path=db_path, device=device)
    else:
        dataset = ChessSelfPlayDataset(db_path=db_path)

    if len(dataset) < batch_size:
        print(f"⚠️ Too little data ({len(dataset)} samples) for training. Skipping iteration...")
        return

    train_size = int(0.9 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    sample_matrix = dataset[0][0]
    in_channels = sample_matrix.shape[0] if hasattr(sample_matrix, 'shape') else 25

    core = ChessCoreNet(in_channels=in_channels)
    model = UnifiedMoveClassifierNet(core_net=core)

    weights_path = "models/weights_bot.pth"
    if os.path.exists(weights_path):
        print(f"📦 Загрузка базовых весов из {weights_path}...")
        try:
            checkpoint = torch.load(weights_path, map_location=device)
            model_dict = model.state_dict()
            safe_checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
            model.load_state_dict(safe_checkpoint, strict=False)
            print("✅ Веса успешно адаптированы под архитектуру текущей итерации!")
        except Exception as e:
            print(f"⚠️ Ошибка при адаптации весов: {e}")
            
    model.to(device)

    criterion_value = nn.MSELoss()
    criterion_policy = nn.KLDivLoss(reduction="batchmean")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"   Epoch {epoch+1}/{epochs}", leave=True)
        for batch in progress_bar:
            inputs = batch[0].to(device, non_blocking=True)
            class_targets = batch[1].to(device, non_blocking=True)
            value_targets = batch[2].to(device, non_blocking=True).view(-1, 1)
            policy_targets = batch[3].to(device, non_blocking=True)

            optimizer.zero_grad()
            
            _, value_preds, policy_logits = model(inputs)

            loss_value = criterion_value(value_preds.view(-1), value_targets.view(-1))
            loss_policy = criterion_policy(policy_logits, class_targets)

            loss = (alpha * loss_value) + (policy_weight * loss_policy)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "ValLoss": f"{loss_value.item():.4f}"})

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    print(f"✅ Unified model RL weights successfully saved to {weights_path}")


def apply_sliding_window(db_path: str, keep_last_n: int = 5000):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM self_play_moves")
        total = cursor.fetchone()[0]
        if total > keep_last_n:
            to_delete = total - keep_last_n
            cursor.execute(f"""
                DELETE FROM self_play_moves 
                WHERE id IN (SELECT id FROM self_play_moves ORDER BY id ASC LIMIT {to_delete})
            """)
            conn.commit()
            print(f"🧹 Sliding window: removed {to_delete} oldest moves from self_play_moves.")
        conn.close()
    except Exception as e:
        print(f"⚠️ Sliding window cleaning failed: {e}")


def run_continuous_loop(iterations: int = 5, games_per_iter: int = 10, sims: int = 40,
                        temperature: float = 1.0, epochs: int = 3, db_path: str = "chess_bot.db",
                        num_workers: int = 1, use_lookahead: bool = False, lookahead_depth: int = 2,
                        use_stockfish_policy: bool = False, keep_last_n: int = 1000):
    init_self_play_db(db_path)
    print(f"\n🚀 STARTING RL TRAINING PIPELINE FOR {iterations} ITERATIONS")

    classifier_weights = "models/weights_classifier.pth"
    bot_weights = "models/weights_bot.pth"

    if not os.path.exists(bot_weights):
        if os.path.exists(classifier_weights):
            shutil.copy(classifier_weights, bot_weights)
            print(f"📦 Base weights copied from {classifier_weights} to {bot_weights}")
        else:
            print(f"⚠️ Warning: {bot_weights} not found. Starting RL from scratch!")

    for i in range(iterations):
        print(f"\n{'='*60}\n🌟 GLOBAL RL ITERATION {i+1}/{iterations}\n{'='*60}")

        run_self_play_session(num_games=games_per_iter, num_simulations=sims,
                              temperature=temperature, db_path=db_path,
                              bot_weights=bot_weights, num_workers=num_workers,
                              lookahead_depth=lookahead_depth if use_lookahead else 0)

        apply_sliding_window(db_path, keep_last_n=keep_last_n)

        train_unified_model(epochs=epochs, batch_size=64, lr=1e-3, alpha=0.5,
                            policy_weight=1.0, db_path=db_path, 
                            use_lookahead=use_lookahead, use_stockfish_policy=use_stockfish_policy)
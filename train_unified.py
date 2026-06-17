"""
Unified Training Script for Chess Neural Network - OPTION A (Lookahead via Value)
"""

import os
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import chess
import numpy as np
import argparse
from classifiers.self_play_dataset import ChessSelfPlayDataset
from models.unified_chess_nets import UnifiedMoveClassifierNet, ChessCoreNet
from engine.rl_trainer import run_self_play_session, init_self_play_db, apply_sliding_window


class ChessMoveDataset(Dataset):
    """
    PyTorch Dataset для загрузки ходов из SQLite.
    Использует уже имеющуюся классификацию ходов для фильтрации Policy Head.
    """
    def __init__(self, db_path="chess_bot.db", game_ids=None, sample_rate=1.0):
        self.samples = []
        self.db_path = db_path
        self.game_ids = set(game_ids) if game_ids is not None else None
        self._load_and_label_data(db_path, sample_rate)
    
    def _load_and_label_data(self, db_path, sample_rate):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"Scanning database (Sample rate: {sample_rate})...")
        cursor.execute("SELECT COUNT(*) FROM moves WHERE fen_after IS NOT NULL AND fen_after != ''")
        total_moves = cursor.fetchone()[0]
        print(f"Total moves in database: {total_moves}")
        
        target_samples = int(total_moves * sample_rate) if sample_rate < 1.0 else total_moves
        
        # ЗАПРОС: Вытаскиваем уже готовую классификацию ходов из вашей базы!
        cursor.execute("""
            SELECT game_id, fen_before, fen_after, evaluation, classification
            FROM moves
            WHERE fen_after IS NOT NULL AND fen_after != ''
            ORDER BY game_id, id
        """)
        
        sample_counter = 0
        while True:
            row = cursor.fetchone()
            if row is None:
                break
                
            game_id, fen_before, fen_after, evaluation, classification = row
            
            if self.game_ids is not None and int(game_id) not in self.game_ids:
                continue
            
            current_eval = evaluation if evaluation is not None else 0.0
            move_class = classification if classification else "Good"
            
            # Сохраняем FEN-ы, оценку и уже готовый класс из вашей базы
            self.samples.append((fen_before, fen_after, move_class, current_eval))
            
            sample_counter += 1
            if sample_counter >= target_samples:
                break
        
        conn.close()
        print(f"Successfully loaded {len(self.samples)} moves into dataset")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        fen_before, fen_after, move_class, evaluation = self.samples[idx]
        
        board_before = chess.Board(fen_before)
        board_after = chess.Board(fen_after)
        
        played_move_idx = 0
        for move in board_before.legal_moves:
            board_before.push(move)
            if board_before.fen().split()[0] == board_after.fen().split()[0]:
                played_move_idx = move.from_square * 64 + move.to_square
                board_before.pop()
                break
            board_before.pop()
            
        tensor = UnifiedMoveClassifierNet.board_to_tensor(board_before)
            
        value_target = np.clip(evaluation / 5.0, -1.0, 1.0)
        
        is_white_turn = (board_before.turn == chess.WHITE)
        player_eval = value_target if is_white_turn else -value_target

        if player_eval > -0.5: 
            policy_weight = 1.0
        else:
            policy_weight = 0.0
        
        return (
            tensor, 
            torch.tensor(value_target, dtype=torch.float32),
            torch.tensor(played_move_idx, dtype=torch.long),
            torch.tensor(policy_weight, dtype=torch.float32)
        )


"""
Unified Training Script for Chess Neural Network - OPTION A (Lookahead via Value)
"""

def train_supervised(
    epochs=10, batch_size=256, lr=1e-3, alpha=0.5, db_path="chess_bot.db",
    sample_rate=1.0, use_val_split=True, val_ratio=0.1, device=None, num_workers=4
):
    """Supervised training phase: Values + Filtered Policy Learning."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"📡 TARGET DEVICE: {device}")

    dataset = ChessMoveDataset(db_path=db_path, sample_rate=sample_rate)
    if len(dataset) < 100:
        print("⚠️ Too little data. Skipping...")
        return None

    if use_val_split:
        val_size = int(len(dataset) * val_ratio)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = None

    core = ChessCoreNet(in_channels=13)
    model = UnifiedMoveClassifierNet(core_net=core).to(device)

    # Losses
    criterion_value = nn.MSELoss()
    # reduction='none' allows custom manual sample-wise weighting before reduction
    criterion_policy_raw = nn.CrossEntropyLoss(reduction='none')

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    weights_dir = "models"
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "weights_bot.pth")

    print(f"🚀 Starting Supervised Training ({epochs} epochs)...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, value_targets, policy_targets, policy_weights in pbar:
            inputs = inputs.to(device)
            value_targets = value_targets.to(device)
            policy_targets = policy_targets.to(device)
            policy_weights = policy_weights.to(device)

            optimizer.zero_grad()

            # Forward propagation (policy_logits are raw un-softmaxed logits)
            _, value_preds, policy_logits = model(inputs)

            # 1. Value Head Loss
            loss_value = criterion_value(value_preds, value_targets)

            # 2. Policy Head Loss (Correct manual weight handling without inflating denominators)
            loss_p_unweighted = criterion_policy_raw(policy_logits, policy_targets)
            
            # Avoid divide-by-zero if entire batch has zero weight
            weight_sum = policy_weights.sum()
            if weight_sum > 0:
                loss_policy = (loss_p_unweighted * policy_weights).sum() / weight_sum
            else:
                loss_policy = torch.tensor(0.0, device=device)

            # Combined total loss
            loss = (alpha * loss_value) + ((1.0 - alpha) * loss_policy)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Value_L": f"{loss_value.item():.3f}", "Policy_L": f"{loss_policy.item():.3f}"})

        # Validation phase
        if val_loader:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for inputs, value_targets, policy_targets, policy_weights in val_loader:
                    inputs = inputs.to(device)
                    value_targets = value_targets.to(device)
                    policy_targets = policy_targets.to(device)
                    policy_weights = policy_weights.to(device)

                    _, value_preds, policy_logits = model(inputs)
                    loss_value = criterion_value(value_preds, value_targets)
                    loss_p_unweighted = criterion_policy_raw(policy_logits, policy_targets)
                    
                    weight_sum = policy_weights.sum()
                    if weight_sum > 0:
                        loss_policy = (loss_p_unweighted * policy_weights).sum() / weight_sum
                    else:
                        loss_policy = torch.tensor(0.0, device=device)

                    loss = (alpha * loss_value) + ((1.0 - alpha) * loss_policy)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            print(f"📊 Summary Epoch {epoch+1} -> Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
            scheduler.step(avg_val_loss)
        
        # Checkpoint state
        torch.save(model.state_dict(), weights_path)
        print(f"💾 Checkpoint saved to {weights_path}")


def train_unified_model(epochs=3, batch_size=64, lr=1e-3, alpha=0.5, db_path="chess_bot.db", device="cpu"):
    """
    Additional phase for Reinforcement Learning training data (MCTS self-play logs).
    Corrected Similarly to handle raw policy logits.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    dataset = ChessSelfPlayDataset(db_path=db_path)
    if len(dataset) < batch_size:
        return
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    core = ChessCoreNet(in_channels=13)
    model = UnifiedMoveClassifierNet(core_net=core).to(device)
    
    # Load base supervised checkpoint if exists
    if os.path.exists("models/weights_bot.pth"):
        model.load_state_dict(torch.load("models/weights_bot.pth", map_location=device), strict=False)

    optimizer = optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=1e-4)
    criterion_value = nn.MSELoss()
    criterion_policy_raw = nn.CrossEntropyLoss(reduction='none')

    print(f"🔄 Fine-tuning model on Self-Play sequences...")
    model.train()
    for epoch in range(epochs):
        for inputs, value_targets, policy_targets, weights in loader:
            inputs = inputs.to(device)
            value_targets = value_targets.to(device)
            policy_targets = policy_targets.to(device)
            weights = weights.to(device)

            optimizer.zero_grad()
            _, value_preds, policy_logits = model(inputs)

            loss_value = criterion_value(value_preds, value_targets)
            loss_p_unweighted = criterion_policy_raw(policy_logits, policy_targets)
            
            w_sum = weights.sum()
            loss_policy = (loss_p_unweighted * weights).sum() / (w_sum + 1e-8) if w_sum > 0 else torch.tensor(0.0, device=device)

            loss = (alpha * loss_value) + ((1.0 - alpha) * loss_policy)
            loss.backward()
            optimizer.step()
            
    torch.save(model.state_dict(), "models/weights_bot.pth")
    print("✅ Self-Play training iteration saved successfully.")

def run_rl_training(iterations=5, games_per_iter=10, sims=800, epochs=3, keep_last_n=1000, db_path="chess_bot.db", temperature=1.2, num_workers=1, bot_weights="models/weights_bot.pth", device=None):
    init_self_play_db(db_path)
    for i in range(iterations):
        print(f"\n🌟 RL ITERATION {i+1}/{iterations}")
        run_self_play_session(num_games=games_per_iter, num_simulations=sims, temperature=temperature, db_path=db_path, bot_weights=bot_weights, num_workers=num_workers)
        train_unified_model(epochs=epochs, batch_size=64, lr=1e-3, alpha=0.5, policy_weight=1.0, db_path=db_path, device=device)
        apply_sliding_window(db_path, keep_last_n_games=keep_last_n)


def main():
    parser = argparse.ArgumentParser(description="Train unified chess model")
    parser.add_argument("--mode", type=str, default="supervised", choices=["supervised", "rl", "combined"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--no-val", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--games-per-iter", type=int, default=10)
    parser.add_argument("--sims", type=int, default=800)
    parser.add_argument("--keep-last-n", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--num-workers", type=int, default=1)
    
    args = parser.parse_args()
    
    if args.mode == "supervised" or args.mode == "combined":
        train_supervised(
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, alpha=args.alpha,
            sample_rate=args.sample_rate, use_val_split=not args.no_val, val_ratio=args.val_ratio, num_workers=args.num_workers
        )
    
    if args.mode == "rl" or args.mode == "combined":
        run_rl_training(
            iterations=args.iterations, games_per_iter=args.games_per_iter, sims=args.sims,
            epochs=args.epochs, keep_last_n=args.keep_last_n, temperature=args.temperature, num_workers=args.num_workers
        )

if __name__ == "__main__":
    main()
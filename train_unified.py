"""
Unified Training Script for Chess Neural Network

This script provides a complete training pipeline for the UnifiedMoveClassifierNet model.
It supports two training modes:

1. SUPERVISED TRAINING: Train on existing Stockfish games from the database
2. SELF-PLAY RL TRAINING: Generate self-play games and train with MCTS

Training Pipeline:
1. Load existing games from chess_bot.db (99k+ games with Stockfish evaluations)
2. Calculate evaluation delta for each move
3. Assign move quality classes based on delta thresholds
4. Train UnifiedMoveClassifierNet with 3 heads:
   - Classification head (6 classes)
   - Value head (-1 to 1)
   - Policy head (64 moves)
5. Optionally continue with self-play RL training

Usage:
    # Supervised training only
    python train_unified.py --mode supervised --epochs 10 --batch-size 256

    # Self-play RL training
    python train_unified.py --mode rl --iterations 5 --games-per-iter 10 --sims 800

    # Combined: supervised first, then RL
    python train_unified.py --mode combined --supervised-epochs 5 --rl-iterations 3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import chess
import numpy as np
import json
import argparse
import random
import time
from collections import Counter
from datetime import datetime
from classifiers.self_play_dataset import ChessSelfPlayDataset
from models.unified_chess_nets import UnifiedMoveClassifierNet, ChessCoreNet
from classifiers.classification_config import CLASS_NAMES, THRESHOLDS
from engine.rl_trainer import run_self_play_session, init_self_play_db, apply_sliding_window


class ChessMoveDataset(Dataset):
    """
    PyTorch Dataset for loading chess moves from SQLite database with Stockfish evaluations.
    """
    
    def __init__(self, db_path="chess_bot.db", game_ids=None, sample_rate=1.0):
        self.samples = []
        self.db_path = db_path
        self.game_ids = set(game_ids) if game_ids is not None else None
        self._load_and_label_data(db_path, sample_rate)
    
    def _load_and_label_data(self, db_path, sample_rate):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"Scanning database (Game filter: {self.game_ids is not None}, Sample rate: {sample_rate})...")
        
        cursor.execute("""
            SELECT COUNT(*) FROM moves
            WHERE fen_after IS NOT NULL AND fen_after != ''
        """)
        total_moves = cursor.fetchone()[0]
        print(f"Total moves in database: {total_moves}")
        
        target_samples = int(total_moves * sample_rate) if sample_rate < 1.0 else total_moves
        print(f"Target samples: {target_samples}")
        
        cursor.execute("""
            SELECT game_id, fen_before, fen_after, evaluation
            FROM moves
            WHERE fen_after IS NOT NULL AND fen_after != ''
            ORDER BY game_id, id
        """)
        
        prev_game_id = None
        prev_eval = 0.3
        sample_counter = 0
        
        while True:
            row = cursor.fetchone()
            if row is None:
                break
                
            game_id, fen_before, fen_after, evaluation = row
            
            if not fen_after:
                continue
            
            try:
                current_id = int(game_id)
            except (ValueError, TypeError):
                continue
            
            if self.game_ids is not None and current_id not in self.game_ids:
                continue
            
            if prev_game_id != current_id:
                prev_game_id = current_id
                prev_eval = 0.3 if evaluation is None else evaluation
            
            current_eval = evaluation if evaluation is not None else prev_eval
            is_white = " w " in fen_before
            
            if is_white:
                delta = prev_eval - current_eval
            else:
                delta = current_eval - prev_eval
            
            if abs(prev_eval) > 3.0:
                delta = delta * 0.3
            
            if delta <= THRESHOLDS[0].max_evaluation:
                class_idx = 0
            else:
                class_idx = 5
                for idx, t in enumerate(THRESHOLDS):
                    if t.min_evaluation <= delta < t.max_evaluation:
                        class_idx = idx
                        break
            
            self.samples.append((fen_before, fen_after, class_idx, current_eval))
            
            if target_samples < float('inf') and sample_counter >= target_samples:
                break
            
            sample_counter += 1
            prev_eval = current_eval
        
        conn.close()
        print(f"Successfully loaded {len(self.samples)} moves into dataset")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        fen_before, fen_after, class_idx, evaluation = self.samples[idx]
        
        board_before = chess.Board(fen_before)
        board_after = chess.Board(fen_after)
        
        played_move_to_square = 0
        for move in board_before.legal_moves:
            board_before.push(move)
            if board_before == board_after:
                played_move_to_square = move.to_square
                board_before.pop()
                break
            board_before.pop()
            
        tensor = UnifiedMoveClassifierNet._board_to_tensor_static(board_before, board_after)
        if tensor.ndim == 4 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
            
        value_target = np.clip(evaluation / 100.0, -1.0, 1.0)
        
        return (
            tensor, 
            torch.tensor(class_idx, dtype=torch.long), 
            torch.tensor(value_target, dtype=torch.float32),
            torch.tensor(played_move_to_square, dtype=torch.long)
        )


def train_supervised(
    epochs=10,
    batch_size=1024,
    lr=1e-3,
    alpha=0.5,
    db_path="chess_bot.db",
    sample_rate=1.0,
    use_val_split=True,
    val_ratio=0.1,
    device="cpu"
):
    """Train the unified model using supervised learning on existing Stockfish games."""
    print("\n" + "="*60)
    print("🎯 SUPERVISED TRAINING: Unified Model on Stockfish Games")
    print("="*60)
    
    # Auto-detect GPU if available
    if device is None or device == "cpu":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"🎮 CUDA detected! Using GPU for training")
        else:
            device = "cpu"
            print("⚠️ CUDA not available, falling back to CPU")
    device = torch.device(device)
    print(f"📡 TARGET DEVICE: {device}")
    
    dataset = ChessMoveDataset(db_path=db_path, device=device)
    
    if len(dataset) < 100:
        print("⚠️ Too little data for training. Skipping...")
        return None
    
    print(f"📊 Dataset size: {len(dataset)} samples")
    
    class_counts = Counter([s[2] for s in dataset.samples])
    print(f"📈 Class distribution: {dict(class_counts)}")
    
    if use_val_split:
        train_size = int(0.9 * len(dataset))
        train_dataset, val_dataset = random_split(
            dataset, [train_size, len(dataset) - train_size]
        )
        print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    else:
        train_dataset, val_dataset = dataset, None
    
    train_loader = None
    val_loader = None

    if use_val_split:
        val_size = int(len(train_dataset) * val_ratio)
        train_size = len(train_dataset) - val_size
        train_sub, val_sub = random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_sub, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_sub, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True,
            persistent_workers=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2, 
            pin_memory=True,
            persistent_workers=True
        )
    
    core = ChessCoreNet(in_channels=25)
    model = UnifiedMoveClassifierNet(core_net=core, num_classes=len(CLASS_NAMES), policy_output_dim=64)
    
    weights_path = "models/weights_bot.pth"
    if os.path.exists(weights_path):
        print(f"📦 Loading pre-trained weights from {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    
    model.to(device)
    model.train()
    
    class_counts = [60901, 12270, 121946, 3689039, 1749495, 949327]
    total_samples = sum(class_counts)
    weights = [total_samples / (len(class_counts) * count) for count in class_counts]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion_class = nn.CrossEntropyLoss(weight=class_weights_tensor)
    criterion_value = nn.MSELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_preds = 0
        total_preds = 0
        
        progress_bar = tqdm(train_loader, desc=f"   Epoch {epoch+1}/{epochs}", leave=True)
        
        for inputs, class_targets, value_targets, policy_targets in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            class_targets = class_targets.to(device, non_blocking=True)
            value_targets = value_targets.to(device, non_blocking=True)
            policy_targets = policy_targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            class_logits, value_preds, policy_logits = model(inputs)
            
            loss_class = criterion_class(class_logits, class_targets)
            loss_value = criterion_value(value_preds.view(-1), value_targets.view(-1))
            
            loss_policy = nn.CrossEntropyLoss()(policy_logits, policy_targets)
            
            loss = loss_class + (0.5 * loss_value) + (1.0 * loss_policy)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            correct_preds += (class_logits.argmax(dim=1) == class_targets).sum().item()
            total_preds += class_targets.size(0)
            
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "ClassAcc": f"{correct_preds/total_preds:.2%}"
            })
        
        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct_preds / total_preds
        
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        if val_loader:
            model.eval()
            with torch.no_grad():
                val_batch_idx = 0
                for inputs, class_targets, value_targets, policy_targets in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    class_targets = class_targets.to(device, non_blocking=True)
                    value_targets = value_targets.to(device, non_blocking=True)
                    policy_targets = policy_targets.to(device, non_blocking=True)
                    
                    class_logits, value_preds, policy_logits = model(inputs)
                    
                    loss_class = criterion_class(class_logits, class_targets)
                    loss_value = criterion_value(value_preds.view(-1), value_targets.view(-1))
                    loss_policy = nn.CrossEntropyLoss()(policy_logits, policy_targets)
                    
                    val_loss += (loss_class + alpha * loss_value + loss_policy).item()

                    val_correct += (class_logits.argmax(dim=1) == class_targets).sum().item()
                    val_total += class_targets.size(0)
                    val_batch_idx += 1
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            scheduler.step(avg_val_loss)
        else:
            avg_val_loss = None
            val_acc = None
        
        if avg_val_loss is not None and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2%}")
        if avg_val_loss is not None:
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2%}")
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    torch.save(model.state_dict(), weights_path)
    print(f"✅ Unified model weights saved to {weights_path}")
    
    final_distribution = {0: 60901, 1: 12270, 2: 121946, 3: 3689039, 4: 1749495, 5: 949327}

    stats = {
        "epochs": epochs, "batch_size": batch_size, "lr": lr, "alpha": alpha,
        "dataset_size": len(dataset),
        "train_size": len(train_dataset) if use_val_split else len(dataset),
        "val_size": len(val_dataset) if use_val_split else 0,
        "best_val_loss": best_val_loss,
        "class_distribution": final_distribution
    }
    
    stats_path = "models/training_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"📊 Training statistics saved to {stats_path}")
    
    return stats


def run_rl_training(
    iterations=5,
    games_per_iter=10,
    sims=800,
    epochs=3,
    keep_last_n=1000,
    db_path="chess_bot.db",
    temperature=1.2,
    num_workers=1,
    bot_weights="models/weights_bot.pth",
    device=None  # Auto-detect GPU
):
    """Run self-play RL training loop."""
    print("\n" + "="*60)
    print("🎯 SELF-PLAY RL TRAINING")
    print("="*60)
    
    init_self_play_db(db_path)
    print(f"\n🚀 STARTING RL TRAINING FOR {iterations} ITERATIONS")
    
    if not os.path.exists(bot_weights):
        if os.path.exists("models/weights_classifier.pth"):
            shutil.copy("models/weights_classifier.pth", bot_weights)
            print(f"📦 Base weights copied from weights_classifier.pth to {bot_weights}")
        else:
            print(f"⚠️ Warning: {bot_weights} not found. Training will start from scratch!")
    
    for i in range(iterations):
        print(f"\n{'='*50}\n🌟 ITERATION {i+1}/{iterations}\n{'='*50}")
        
        run_self_play_session(num_games=games_per_iter, num_simulations=sims,
                              temperature=temperature, db_path=db_path,
                              bot_weights=bot_weights, num_workers=num_workers)
        
        train_unified_model(epochs=epochs, batch_size=64, lr=1e-3, alpha=0.5,
                            policy_weight=1.0, db_path=db_path, device=device)
        
        apply_sliding_window(db_path, keep_last_n_games=keep_last_n)


def train_unified_model(epochs=3, batch_size=64, lr=1e-3, alpha=0.5,
                        policy_weight=1.0, db_path="chess_bot.db", device="cpu"):
    """Train the unified model with lookahead data."""
    print("\n" + "-"*40)
    print("🧠 PHASE 2: UNIFIED MODEL TRAINING (RL + Policy)")
    print("-" * 40)
    
    # Auto-detect GPU if available
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
            print(f"🎮 CUDA detected! Using GPU for training")
        else:
            device = "cpu"
            print("⚠️ CUDA not available, falling back to CPU")
    device = torch.device(device)
    print(f"📡 TARGET DEVICE: {device}")
    
    print("Chess self play dataset loading...")
    dataset = ChessSelfPlayDataset(db_path=db_path)
    
    if len(dataset) < 50:
        print("⚠️ Too little data for training. Skipping...")
        return
    
    train_size = int(0.9 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda")
    )
    
    core = ChessCoreNet(in_channels=25)
    model = UnifiedMoveClassifierNet(core_net=core, num_classes=len(CLASS_NAMES), policy_output_dim=64)
    
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
        
        progress_bar = tqdm(train_loader, desc=f"   Epoch {epoch+1}/{epochs}", leave=True)
        batch_idx = 0
        
        for batch in progress_bar:
            inputs, class_targets, value_targets, *_ = batch
            
            inputs = inputs.to(device, non_blocking=True)
            class_targets = class_targets.to(device, non_blocking=True)
            value_targets = value_targets.to(device, non_blocking=True)
            
            current_batch_size = inputs.size(0)
            policy_targets = torch.ones((current_batch_size, 64), dtype=torch.float32, device=device) / 64.0
            
            optimizer.zero_grad()
            class_logits, value_preds, policy_logits = model(inputs)
            
            loss_class = criterion_class(class_logits, class_targets)
            loss_value = criterion_value(value_preds.view(-1), value_targets.view(-1))
            loss_policy = nn.CrossEntropyLoss()(policy_logits, policy_targets)
            
            loss = loss_class + (alpha * loss_value) + (policy_weight * loss_policy)
            
            loss.backward()
            optimizer.step()
            
            batch_idx += 1
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    torch.save(model.state_dict(), weights_path)
    print(f"✅ Unified model weights saved to {weights_path}")


def main():
    parser = argparse.ArgumentParser(description="Train unified chess model")
    parser.add_argument("--mode", type=str, default="supervised",
                        choices=["supervised", "rl", "combined"],
                        help="Training mode: supervised, rl, or combined")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.5, help="Value loss weight")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Data sample rate")
    parser.add_argument("--no-val", action="store_true", help="Disable validation split")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio")
    
    # RL training parameters
    parser.add_argument("--iterations", type=int, default=5, help="RL iterations")
    parser.add_argument("--games-per-iter", type=int, default=10, help="Games per iteration")
    parser.add_argument("--sims", type=int, default=800, help="MCTS simulations")
    parser.add_argument("--keep-last-n", type=int, default=1000, help="Games to keep")
    parser.add_argument("--temperature", type=float, default=1.2, help="Temperature")
    parser.add_argument("--num-workers", type=int, default=1, help="Workers")
    
    args = parser.parse_args()
    
    if args.mode == "supervised":
        train_supervised(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            alpha=args.alpha,
            sample_rate=args.sample_rate,
            use_val_split=not args.no_val,
            val_ratio=args.val_ratio,
            device=None  # Auto-detect GPU
        )
    
    elif args.mode == "rl":
        run_rl_training(
            iterations=args.iterations,
            games_per_iter=args.games_per_iter,
            sims=args.sims,
            epochs=args.epochs,
            keep_last_n=args.keep_last_n,
            temperature=args.temperature,
            num_workers=args.num_workers,
            device=None  # Auto-detect GPU
        )
    
    elif args.mode == "combined":
        print("\n" + "="*60)
        print("🚀 COMBINED TRAINING: Supervised + RL")
        print("="*60)
        
        print("\n📚 PHASE 1: Supervised Training on Stockfish Games")
        train_supervised(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            alpha=args.alpha,
            sample_rate=args.sample_rate,
            use_val_split=not args.no_val,
            val_ratio=args.val_ratio,
            device=None  # Auto-detect GPU
        )
        
        print("\n🎮 PHASE 2: Self-Play RL Training")
        run_rl_training(
            iterations=args.iterations,
            games_per_iter=args.games_per_iter,
            sims=args.sims,
            epochs=args.epochs,
            keep_last_n=args.keep_last_n,
            temperature=args.temperature,
            num_workers=args.num_workers,
            device=None  # Auto-detect GPU
        )
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    import shutil
    main()

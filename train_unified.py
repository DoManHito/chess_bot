"""
Unified Training Script for Chess Neural Network - OPTION A (Lookahead via Value)
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
        
        # ИСПРАВЛЕНИЕ: Точное определение стабильного индекса сделанного хода (от 0 до 4095)
        played_move_idx = 0
        for move in board_before.legal_moves:
            board_before.push(move)
            # Сравниваем только размещение фигур, отсекая шум в FEN
            if board_before.fen().split()[0] == board_after.fen().split()[0]:
                played_move_idx = move.from_square * 64 + move.to_square
                board_before.pop()
                break
            board_before.pop()
            
        # Кодируем доску в 13 каналов (Вариант А)
        tensor = UnifiedMoveClassifierNet.board_to_tensor(board_before)
            
        # Масштабируем оценку centipawns в диапазон [-1, 1] без жесткого клиппинга
        value_target = np.clip(evaluation / 300.0, -1.0, 1.0)
        
        # Использование вашей базы: выставляем веса для обучения Policy Head
        if move_class in ["Best", "Excellent", "Good"]:
            policy_weight = 1.0   # Идеальные ходы учим в полную силу
        elif move_class == "Inaccuracy":
            policy_weight = 0.2   # Неточности учим слабо
        else:
            policy_weight = 0.0   # Ошибки и Зевки (Mistake, Blunder) полностью игнорируем!
        
        return (
            tensor, 
            torch.tensor(value_target, dtype=torch.float32),
            torch.tensor(played_move_idx, dtype=torch.long),
            torch.tensor(policy_weight, dtype=torch.float32)
        )


def train_supervised(
    epochs=10, batch_size=256, lr=1e-3, alpha=0.5, db_path="chess_bot.db",
    sample_rate=1.0, use_val_split=True, val_ratio=0.1, device=None
):
    """Обучение модели на имеющейся базе данных (Оценка + Фильтрованный Policy)."""
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
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = None
    
    # Модель создается строго с 13 входными каналами
    core = ChessCoreNet(in_channels=13)
    model = UnifiedMoveClassifierNet(core_net=core)
    
    weights_path = "models/weights_bot.pth"
    if os.path.exists(weights_path):
        print(f"📦 Loading pre-trained weights...")
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    
    model.to(device)
    
    criterion_value = nn.MSELoss()
    criterion_policy_raw = nn.CrossEntropyLoss(reduction='none') # Попиксельный лосс для применения маски
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    for epoch in range(epochs):
        model.train()
        total_loss, total_val_loss = 0, 0
        
        progress_bar = tqdm(train_loader, desc=f"   Epoch {epoch+1}/{epochs}", leave=True)
        for inputs, value_targets, policy_targets, policy_weights in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            value_targets = value_targets.to(device, non_blocking=True)
            policy_targets = policy_targets.to(device, non_blocking=True)
            policy_weights = policy_weights.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # class_logits игнорируем (Вариант А)
            _, value_preds, policy_logits = model(inputs)
            
            # 1. Лосс оценки (MSE)
            loss_value = criterion_value(value_preds.view(-1), value_targets.view(-1))
            
            loss_policy = (loss_p_unweighted * policy_weights).sum() / (policy_weights.sum() + 1e-8)
            
            # Общий лосс
            loss = (alpha * loss_value) + loss_policy
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "ValLoss": f"{loss_value.item():.4f}", "PolLoss": f"{loss_policy.item():.4f}"})
            
        # Валидация
        if val_loader:
            model.eval()
            with torch.no_grad():
                for inputs, value_targets, policy_targets, policy_weights in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    value_targets = value_targets.to(device, non_blocking=True)
                    policy_targets = policy_targets.to(device, non_blocking=True)
                    policy_weights = policy_weights.to(device, non_blocking=True)
                    
                    _, value_preds, policy_logits = model(inputs)
                    loss_value = criterion_value(value_preds.view(-1), value_targets.view(-1))
                    loss_p_unweighted = criterion_policy_raw(policy_logits, policy_targets)
                    loss_policy = (loss_p_unweighted * policy_weights).mean()
                    
                    total_val_loss += ((alpha * loss_value) + loss_policy).item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"  Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
            scheduler.step(avg_val_loss)
            
    torch.save(model.state_dict(), weights_path)
    print(f"✅ Unified model weights saved to {weights_path}")


def train_unified_model(epochs=3, batch_size=64, lr=1e-3, alpha=0.5, policy_weight=1.0, db_path="chess_bot.db", device="cpu"):
    """Дополнительная фаза тренировки Lookahead данных (RL Self-Play)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    dataset = ChessSelfPlayDataset(db_path=db_path)
    if len(dataset) < 50:
        return
        
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    core = ChessCoreNet(in_channels=13)
    model = UnifiedMoveClassifierNet(core_net=core)
    
    weights_path = "models/weights_bot.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model.to(device)
    
    criterion_value = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            inputs, _, value_targets, *_ = batch # Игнорируем старые таргеты классов
            
            inputs = inputs.to(device, non_blocking=True)
            value_targets = value_targets.to(device, non_blocking=True)
            
            # Превращаем в 4096-размерный плейсхолдер для совместимости с MCTS логикой самоигры
            current_batch_size = inputs.size(0)
            policy_targets = torch.ones((current_batch_size, 4096), dtype=torch.float32, device=device) / 4096.0
            
            optimizer.zero_grad()
            _, value_preds, policy_logits = model(inputs)
            
            loss_value = criterion_value(value_preds.view(-1), value_targets.view(-1))
            loss_policy = nn.CrossEntropyLoss()(policy_logits, policy_targets)
            
            loss = (alpha * loss_value) + (policy_weight * loss_policy)
            loss.backward()
            optimizer.step()
            
    torch.save(model.state_dict(), weights_path)


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
            sample_rate=args.sample_rate, use_val_split=not args.no_val, val_ratio=args.val_ratio
        )
    
    if args.mode == "rl" or args.mode == "combined":
        run_rl_training(
            iterations=args.iterations, games_per_iter=args.games_per_iter, sims=args.sims,
            epochs=args.epochs, keep_last_n=args.keep_last_n, temperature=args.temperature, num_workers=args.num_workers
        )

if __name__ == "__main__":
    main()
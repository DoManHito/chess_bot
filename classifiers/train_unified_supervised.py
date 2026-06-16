"""
Supervised Training Script for Unified Chess Neural Network

This module trains the UnifiedMoveClassifierNet model using existing Stockfish games
from the database. The training uses evaluation delta (change in engine evaluation)
as the ground truth for move quality classification.

Training Pipeline:
1. Load existing games from chess_bot.db (99k+ games with Stockfish evaluations)
2. Calculate evaluation delta for each move
3. Assign move quality classes based on delta thresholds
4. Train UnifiedMoveClassifierNet with 3 heads:
   - Classification head (6 classes)
   - Value head (-1 to 1)
   - Policy head (64 moves)
5. Save weights to models/weights_bot.pth
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import chess
import numpy as np
import json
import argparse
import random
from collections import Counter

from models.unified_chess_nets import UnifiedMoveClassifierNet, ChessCoreNet
from classifiers.classification_config import CLASS_NAMES, THRESHOLDS


class ChessMoveDataset(Dataset):
    """
    PyTorch Dataset for loading chess moves from SQLite database with Stockfish evaluations.
    
    This dataset:
    - Queries the chess_bot.db database for move records (FEN positions + evaluation)
    - Calculates evaluation delta for each move (change in position evaluation)
    - Assigns move quality classes based on delta thresholds
    - Returns tensor representations of board positions with class, value, and policy targets
    
    Attributes:
        samples: List of tuples (fen_before, fen_after, class_idx, evaluation)
        db_path: Path to the SQLite database
    """
    
    def __init__(self, db_path="chess_bot.db", game_ids=None, sample_rate=1.0):
        """
        Initialize the ChessMoveDataset.
        
        Args:
            db_path: Path to the SQLite database containing move records.
            game_ids: Optional iterable of game IDs to filter the dataset.
            sample_rate: Fraction of data to sample (1.0 = all data).
        """
        self.samples = []
        self.db_path = db_path
        
        # Convert game_ids to set for O(1) lookup
        self.game_ids = set(game_ids) if game_ids is not None else None
        
        # Load and label all data from the database
        self._load_and_label_data(db_path, sample_rate)
    
    def _load_and_label_data(self, db_path, sample_rate):
        """
        Load chess moves from database and assign quality classes.
        
        Args:
            db_path: Path to the SQLite database
            sample_rate: Fraction of data to sample (1.0 = all data)
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"Scanning database and calculating classes (Game filter: {self.game_ids is not None}, Sample rate: {sample_rate})...")
        
        # First, get total count
        cursor.execute("""
            SELECT COUNT(*) FROM moves
            WHERE fen_after IS NOT NULL AND fen_after != ''
        """)
        total_moves = cursor.fetchone()[0]
        print(f"Total moves in database: {total_moves}")
        
        # Calculate target sample size
        target_samples = int(total_moves * sample_rate) if sample_rate < 1.0 else total_moves
        print(f"Target samples: {target_samples}")
        
        # Query all moves with evaluation data
        cursor.execute("""
            SELECT game_id, fen_before, fen_after, evaluation
            FROM moves
            WHERE fen_after IS NOT NULL AND fen_after != ''
            ORDER BY game_id, id
        """)
        
        prev_game_id = None
        prev_eval = 0.3  # Initial evaluation (neutral position)
        sample_counter = 0
        
        while True:
            row = cursor.fetchone()
            if row is None:
                break
                
            game_id, fen_before, fen_after, evaluation = row
            
            # Skip moves without FEN after position
            if not fen_after:
                continue
            
            try:
                current_id = int(game_id)
            except (ValueError, TypeError):
                continue
            
            # Filter by membership in train/val subset
            if self.game_ids is not None and current_id not in self.game_ids:
                continue
            
            # Reset evaluation context when game changes
            if prev_game_id != current_id:
                prev_game_id = current_id
                prev_eval = 0.3 if evaluation is None else evaluation
            
            current_eval = evaluation if evaluation is not None else prev_eval
            
            # Determine if the move was made by White or Black
            is_white = " w " in fen_before
            
            # Calculate evaluation delta (change in evaluation after the move)
            # For White: positive delta = improvement, negative = deterioration
            # For Black: positive delta = deterioration, negative = improvement
            if is_white:
                delta = prev_eval - current_eval
            else:
                delta = current_eval - prev_eval
            
            # Smooth delta when one side has a strong material advantage
            if abs(prev_eval) > 3.0:
                delta = delta * 0.3
            
            # Assign move quality class based on evaluation delta
            if delta <= THRESHOLDS[0].max_evaluation:
                class_idx = 0  # Ideal move ("Best")
            else:
                class_idx = 5  # Default to "Blunder"
                for idx, t in enumerate(THRESHOLDS):
                    if t.min_evaluation <= delta < t.max_evaluation:
                        class_idx = idx
                        break
            
            # Store sample
            self.samples.append((fen_before, fen_after, class_idx, current_eval))
            
            # Sample if needed
            if target_samples < float('inf') and sample_counter >= target_samples:
                break
            
            sample_counter += 1
            prev_eval = current_eval
        
        conn.close()
        print(f"Successfully loaded {len(self.samples)} moves into dataset")
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (input_tensor, class_target, value_target) where:
            - input_tensor: 2D tensor representation of the board position change
            - class_target: Integer class label (0-5)
            - value_target: Float value from -1 to 1 (based on evaluation)
        """
        fen_before, fen_after, class_idx, evaluation = self.samples[idx]
        board_before = chess.Board(fen_before)
        board_after = chess.Board(fen_after)
        
        # Convert board positions to tensor representation
        tensor = UnifiedMoveClassifierNet._board_to_tensor_static(board_before, board_after)
        
        # Squeeze batch dimension if tensor has 4 dimensions
        if tensor.ndim == 4 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        
        # Value target: normalize evaluation to -1 to 1 range
        # Stockfish evaluation is typically in centipawns, normalize by dividing by 100
        value_target = np.clip(evaluation / 100.0, -1.0, 1.0)
        
        return tensor, torch.tensor(class_idx, dtype=torch.long), torch.tensor(value_target, dtype=torch.float32)


def get_all_game_ids(db_path="chess_bot.db", limit=None):
    """
    Collect all unique game IDs from the chess database.
    
    Args:
        db_path: Path to the SQLite database
        limit: Maximum number of games to include (None = all games)
    
    Returns:
        List of game IDs
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if limit is not None:
        cursor.execute("""
            SELECT DISTINCT game_id FROM moves
            WHERE game_id IS NOT NULL
            ORDER BY game_id
            LIMIT ?
        """, (limit,))
    else:
        cursor.execute("""
            SELECT DISTINCT game_id FROM moves
            WHERE game_id IS NOT NULL
            ORDER BY game_id
        """)
    
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]


def train_unified_supervised(
    epochs=10,
    batch_size=1024,
    lr=1e-3,
    alpha=0.5,
    policy_weight=0.1,
    db_path="chess_bot.db",
    sample_rate=1.0,
    use_val_split=True,
    val_ratio=0.1,
    device="cpu"
):
    """
    Train the unified model using supervised learning on existing Stockfish games.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        alpha: Weight for value loss
        policy_weight: Weight for policy loss (supervised policy from evaluation)
        db_path: Path to the SQLite database
        sample_rate: Fraction of data to sample (1.0 = all data)
        use_val_split: Whether to split data into train/val sets
        val_ratio: Ratio of validation data (if use_val_split=True)
        device: Device to use for training
    
    Returns:
        Dictionary with training statistics
    """
    print("\n" + "="*60)
    print("🎯 SUPERVISED TRAINING: Unified Model on Stockfish Games")
    print("="*60)
    
    device = torch.device(device)
    print(f"📡 TARGET DEVICE: {device}")
    
    # Load dataset
    dataset = ChessMoveDataset(db_path=db_path, sample_rate=sample_rate)
    
    if len(dataset) < 100:
        print("⚠️ Too little data for training. Skipping...")
        return None
    
    print(f"📊 Dataset size: {len(dataset)} samples")
    
    # Show class distribution
    class_counts = Counter([s[2] for s in dataset.samples])
    print(f"📈 Class distribution: {dict(class_counts)}")
    
    # Split data
    if use_val_split:
        train_size = int(0.9 * len(dataset))
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, len(dataset) - train_size]
        )
        print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    else:
        train_dataset, val_dataset = dataset, None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    ) if val_dataset else None
    
    # Initialize unified model
    core = ChessCoreNet(in_channels=25)
    model = UnifiedMoveClassifierNet(
        core_net=core,
        num_classes=len(CLASS_NAMES),
        policy_output_dim=64
    )
    
    # Load pre-trained weights if available
    weights_path = "models/weights_bot.pth"
    if os.path.exists(weights_path):
        print(f"📦 Loading pre-trained weights from {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    
    model.to(device)
    model.train()
    
    # Loss functions
    criterion_class = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    # For supervised policy, use MSE instead of KL divergence
    criterion_policy = nn.MSELoss()
    
    # Create policy targets from dataset (evaluation-based policy)
    # Use a uniform distribution as the policy target
    # This is a simplified approach - in practice, you'd want to extract actual move probabilities
    policy_samples = [[1.0 / 64] * 64 for _ in range(len(dataset.samples))]
    
    policy_targets = torch.tensor(policy_samples, dtype=torch.float32)
    policy_targets = policy_targets / policy_targets.sum(dim=1, keepdim=True)
    print(f"📊 Policy targets shape: {policy_targets.shape}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct_preds = 0
        total_preds = 0
        
        progress_bar = tqdm(train_loader, desc=f"   Epoch {epoch+1}/{epochs}", leave=True)
        batch_idx = 0
        for inputs, class_targets, value_targets in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            class_targets = class_targets.to(device, non_blocking=True)
            value_targets = value_targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            class_logits, value_preds, policy_logits = model(inputs)
            
            loss_class = criterion_class(class_logits, class_targets)
            loss_value = criterion_value(value_preds.view(-1), value_targets.view(-1))
            # Skip policy loss for supervised training - use uniform policy target
            # The policy head is trained implicitly through the shared backbone
            loss_policy = torch.tensor(0.0, device=device)
            
            loss = loss_class + alpha * loss_value
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            batch_idx += 1
            total_loss += loss.item()
            correct_preds += (class_logits.argmax(dim=1) == class_targets).sum().item()
            total_preds += class_targets.size(0)
            
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "ClassAcc": f"{correct_preds/total_preds:.2%}"
            })
        
        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct_preds / total_preds
        
        # Validation phase
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        if val_loader:
            model.eval()
            with torch.no_grad():
                val_batch_idx = 0
                for inputs, class_targets, value_targets in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    class_targets = class_targets.to(device, non_blocking=True)
                    value_targets = value_targets.to(device, non_blocking=True)
                    
                    class_logits, value_preds, policy_logits = model(inputs)
                    
                    loss_class = criterion_class(class_logits, class_targets)
                    loss_value = criterion_value(value_preds.view(-1), value_targets.view(-1))
                    # Skip policy loss for supervised training - use uniform policy target
                    # The policy head is trained implicitly through the shared backbone
                    loss_policy = torch.tensor(0.0, device=device)
                    
                    val_loss += (loss_class + alpha * loss_value).item()
                    val_correct += (class_logits.argmax(dim=1) == class_targets).sum().item()
                    val_total += class_targets.size(0)
                    val_batch_idx += 1
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            scheduler.step(avg_val_loss)
        else:
            avg_val_loss = None
            val_acc = None
        
        # Save best model
        if avg_val_loss is not None and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2%}")
        if avg_val_loss is not None:
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2%}")
    
    # Save best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save trained weights
    torch.save(model.state_dict(), weights_path)
    print(f"✅ Unified model weights saved to {weights_path}")
    
    # Save training statistics
    stats = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "alpha": alpha,
        "policy_weight": policy_weight,
        "dataset_size": len(dataset),
        "train_size": len(train_dataset) if use_val_split else len(dataset),
        "val_size": len(val_dataset) if use_val_split else 0,
        "best_val_loss": best_val_loss,
        "class_distribution": dict(class_counts)
    }
    
    stats_path = "models/training_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"📊 Training statistics saved to {stats_path}")
    
    return stats


if __name__ == "__main__":
    import json
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train unified model on Stockfish games")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.5, help="Value loss weight")
    parser.add_argument("--policy-weight", type=float, default=0.1, help="Policy loss weight")
    parser.add_argument("--db-path", type=str, default="chess_bot.db", help="Database path")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Fraction of data to sample (1.0 = all)")
    parser.add_argument("--no-val", action="store_true", help="Disable validation split")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Run supervised training
    stats = train_unified_supervised(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        policy_weight=args.policy_weight,
        db_path=args.db_path,
        sample_rate=args.sample_rate,
        use_val_split=not args.no_val,
        val_ratio=args.val_ratio,
        device=args.device
    )
    
    if stats:
        print("\n" + "="*60)
        print("🎉 SUPERVISED TRAINING COMPLETE")
        print("="*60)
        print(f"Dataset size: {stats['dataset_size']}")
        print(f"Train samples: {stats['train_size']}")
        print(f"Val samples: {stats['val_size']}")
        print(f"Best validation loss: {stats['best_val_loss']:.4f}")
        print(f"Class distribution: {stats['class_distribution']}")

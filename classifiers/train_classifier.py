import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import numpy as np
import random
from collections import Counter

from models.chess_nets import ChessCoreNet, MoveClassifierNet
from classifiers.move_classifier import MoveClassifier
from classifiers.classification_config import CLASS_NAMES, THRESHOLDS, NEGATIVE_THRESHOLD

class ChessDataset(Dataset):
    """Optimized dataset: loads fen_before and ready fen_after directly from the database."""
    def __init__(self, db_path="chess_bot.db", game_ids=None):
        self.samples = [] 
        # Avoid rigid initialization here to prevent process conflicts
        self.classifier_utils = None 
        self.db_path = db_path
        
        # Convert to set for O(1) lookup during iteration
        self.game_ids = set(game_ids) if game_ids is not None else None
        
        self._load_and_label_data_lightweight(db_path)

    def _load_and_label_data_lightweight(self, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"Scanning database and calculating classes (Game filter: {self.game_ids is not None})...")
        cursor.execute("""
            SELECT game_id, fen_before, fen_after, evaluation 
            FROM moves 
            ORDER BY game_id, id
        """)
        
        prev_game_id = None
        prev_eval = 0.3
        
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

            # Filter by membership in train/val subset
            if self.game_ids is not None and current_id not in self.game_ids:
                continue
                
            # Reset evaluation context when game changes
            if prev_game_id != current_id:
                prev_game_id = current_id
                prev_eval = 0.3 if evaluation is None else evaluation
            
            current_eval = evaluation if evaluation is not None else prev_eval
            
            is_white = " w " in fen_before
            
            # Calculate evaluation loss (difference)
            if is_white:
                delta = prev_eval - current_eval
            else:
                delta = current_eval - prev_eval

            # Smooth delta when one side has a strong material advantage
            if abs(prev_eval) > 3.0:
                delta = delta * 0.3
                
            
            # If move maintained or improved evaluation (delta <= max value for Best, e.g., 0.02)
            if delta <= THRESHOLDS[0].max_evaluation:
                class_idx = 0  # Ideal move ("Best")
            else:
                class_idx = 5  # Default "Blunder" (if delta is huge and doesn't fall into ranges below)
                for idx, t in enumerate(THRESHOLDS):
                    # Use strict upper threshold comparison < to avoid overlaps
                    if t.min_evaluation <= delta < t.max_evaluation:
                        class_idx = idx
                        break
            
            # Append data tuple to samples list
            self.samples.append((fen_before, fen_after, class_idx))
            
            prev_eval = current_eval

        conn.close()
        print(f"Successfully loaded moves into dataset: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fen_before, fen_after, class_idx = self.samples[idx]
        board_before = chess.Board(fen_before)
        board_after = chess.Board(fen_after)
            
        tensor = MoveClassifier._board_to_tensor_static(board_before, board_after)
        
        if tensor.ndim == 4 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
            
        return tensor, class_idx


def get_all_game_ids(db_path="chess_bot.db"):
    """Helper function for quickly collecting unique game IDs from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT game_id FROM moves WHERE game_id IS NOT NULL")
    
    game_ids = []
    for row in cursor.fetchall():
        try:
            game_ids.append(int(row[0]))
        except (ValueError, TypeError):
            continue
            
    conn.close()
    return game_ids


if __name__ == "__main__":
    db_path = "chess_bot.db"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load and fairly split unique game IDs (80% to 20%)
    all_games = get_all_game_ids(db_path)
    print(f"Total unique games in database: {len(all_games)}")
    
    random.seed(42) # Fixing seed for reproducibility of the split
    random.shuffle(all_games)
    
    split_idx = int(0.8 * len(all_games))
    train_game_ids = all_games[:split_idx]
    val_game_ids = all_games[split_idx:]
    
    print(f"Games allocated for training (train): {len(train_game_ids)}")
    print(f"Games allocated for validation (val): {len(val_game_ids)}")
    
    # 2. Initialize TWO independent dataset objects
    print("\nInitializing training dataset...")
    train_dataset = ChessDataset(db_path=db_path, game_ids=train_game_ids)
    
    print("\nInitializing validation dataset...")
    val_dataset = ChessDataset(db_path=db_path, game_ids=val_game_ids)
    
    # 3. Wrap in DataLoader (Different optimal batch sizes)
    num_workers = 4 

    train_loader = DataLoader(
        train_dataset, 
        batch_size=2048, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=2048, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    # 4. Calculate class weights STRICTLY from the training subset
    train_targets = [sample[-1] for sample in train_dataset.samples]
    target_counts = Counter(train_targets)
    total_train_samples = len(train_targets)
    
    # CORRECTED: Smoothing via square root (Square Root Smoothing)
    # This prevents rare classes from getting destructively huge weights
    class_weights = []
    for i in range(len(CLASS_NAMES)):
        count = target_counts.get(i, 1)
        weight = (total_train_samples / count) ** 0.5
        class_weights.append(weight)
        
    # Normalize obtained weights so their mean value equals 1.0
    mean_weight = sum(class_weights) / len(class_weights)
    class_weights = [w / mean_weight for w in class_weights]
        
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Calculated (SMOOTHED and NORMALIZED) weights for classes: {class_weights}")
    
    # 5. Assemble the classifier model
    core = ChessCoreNet(in_channels=25)
    model = MoveClassifierNet(core_net=core, num_classes=len(CLASS_NAMES))
    model = model.to(device)
    
    weights_path = "models/weights_classifier.pth"
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Successfully loaded existing weights from {weights_path}.")
    except FileNotFoundError:
        print("Weights not found. Starting model training from scratch.")
    
    # 6. Setup loss function and optimizer with regularization
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4) 

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.5)
    
    # 7. Main combined training and validation loop
    epochs = 20  
    best_val_loss = float('inf')
    
    print(f"\nStarting classifier training on device: {device}")
    for epoch in range(epochs):
        
        # --- TRAINING PHASE (Train) ---
        model.train() 
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
        epoch_train_loss = total_train_loss / len(train_loader)
        epoch_train_acc = 100.0 * train_correct / train_total
        print(f"Epoch {epoch+1}/{epochs} | Training Loss: {epoch_train_loss:.4f} | Training Accuracy: {epoch_train_acc:.2f}%")
        
        # --- VALIDATION PHASE (Validation) ---
        model.eval() 
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(): 
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                
                val_outputs = model(val_inputs)
                loss = criterion(val_outputs, val_targets)
                
                total_val_loss += loss.item()
                _, val_predicted = val_outputs.max(1)
                val_total += val_targets.size(0)
                val_correct += val_predicted.eq(val_targets).sum().item()
        
        epoch_val_loss = total_val_loss / len(val_loader)
        epoch_val_acc = 100.0 * val_correct / val_total
        print(f"--> Validation | Loss: {epoch_val_loss:.4f} | HONEST Accuracy: {epoch_val_acc:.2f}%")

        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current training step (LR): {current_lr:.6f}")
        
        # --- SAVE BEST RESULTS (Checkpoint) ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), weights_path)
            print(f"Weights saved! Validation loss decreased to best: {best_val_loss:.4f}")
        else:
            print(f"Weights not overwritten. Best loss remains: {best_val_loss:.4f}")
            
        print("-" * 65)
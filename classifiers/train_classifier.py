"""
Chess Move Classifier Training Script

This module trains a neural network classifier to evaluate chess moves based on
evaluation delta (change in engine evaluation after a move). The classifier learns
to categorize moves into quality classes: Best, Excellent, Good, Inaccuracy, Mistake, Blunder.

The training process:
1. Loads chess moves from a SQLite database containing FEN positions and engine evaluations
2. Calculates evaluation delta for each move (how much the position evaluation changed)
3. Assigns move quality classes based on delta thresholds
4. Trains a MoveClassifierNet model using the labeled dataset
5. Validates performance and saves the best model weights
"""

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
    """
    PyTorch Dataset for loading and labeling chess moves from a SQLite database.
    
    This dataset:
    - Queries the chess_bot.db database for move records (FEN positions + evaluation)
    - Calculates evaluation delta for each move (change in position evaluation)
    - Assigns move quality classes based on predefined thresholds
    - Returns tensor representations of board positions with class labels
    
    The evaluation delta represents how much the engine evaluation changed after
    a move. A positive delta (for White) or negative delta (for Black) indicates
    an improvement in position, while the opposite indicates deterioration.
    
    Attributes:
        samples: List of tuples (fen_before, fen_after, class_idx)
        db_path: Path to the SQLite database
        game_ids: Optional set of game IDs to filter the dataset
    """
    
    def __init__(self, db_path="chess_bot.db", game_ids=None):
        """
        Initialize the ChessDataset.
        
        Args:
            db_path: Path to the SQLite database containing move records.
                     Default is "chess_bot.db" in the project root.
            game_ids: Optional iterable of game IDs to filter the dataset.
                     If provided, only moves from these games will be included.
        """
        self.samples = []  # List of (fen_before, fen_after, class_idx) tuples
        self.classifier_utils = None  # Placeholder for future utility objects
        self.db_path = db_path
        
        # Convert game_ids to set for O(1) lookup during iteration
        self.game_ids = set(game_ids) if game_ids is not None else None
        
        # Load and label all data from the database
        self._load_and_label_data_lightweight(db_path)

    def _load_and_label_data_lightweight(self, db_path):
        """
        Load chess moves from database and assign quality classes.
        
        This method queries the 'moves' table and processes each move to:
        1. Calculate evaluation delta (change in engine evaluation)
        2. Determine move quality class based on delta thresholds
        3. Store FEN positions and class index in samples list
        
        Args:
            db_path: Path to the SQLite database
            
        Note:
            - Evaluation delta is calculated as: prev_eval - current_eval for White
              and current_eval - prev_eval for Black
            - When one side has a strong material advantage (|eval| > 3.0),
              the delta is smoothed (multiplied by 0.3) to reduce noise
            - Class assignment follows the THRESHOLDS configuration
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"Scanning database and calculating classes (Game filter: {self.game_ids is not None})...")
        cursor.execute("""
            SELECT game_id, fen_before, fen_after, evaluation
            FROM moves
            ORDER BY game_id, id
        """)
        
        prev_game_id = None
        prev_eval = 0.3  # Initial evaluation (neutral position)
        
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
            # This reduces noise from positions with obvious material differences
            if abs(prev_eval) > 3.0:
                delta = delta * 0.3
                
            
            # Assign move quality class based on evaluation delta
            if delta <= THRESHOLDS[0].max_evaluation:
                class_idx = 0  # Ideal move ("Best") - minimal evaluation change
            else:
                class_idx = 5  # Default to "Blunder" for very large deltas
                for idx, t in enumerate(THRESHOLDS):
                    # Use strict upper threshold comparison < to avoid overlaps
                    if t.min_evaluation <= delta < t.max_evaluation:
                        class_idx = idx
                        break
            
            # Append data tuple to samples list
            self.samples.append((fen_before, fen_after, class_idx))
            
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
            Tuple of (tensor, class_idx) where:
            - tensor: 2D tensor representation of the board position change
            - class_idx: Integer class label (0-5)
        """
        fen_before, fen_after, class_idx = self.samples[idx]
        board_before = chess.Board(fen_before)
        board_after = chess.Board(fen_after)
            
        # Convert board positions to tensor representation
        tensor = MoveClassifier._board_to_tensor_static(board_before, board_after)
            
        # Squeeze batch dimension if tensor has 4 dimensions (batch_size=1)
        if tensor.ndim == 4 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
            
        return tensor, class_idx


def get_all_game_ids(db_path="chess_bot.db"):
    """
    Collect all unique game IDs from the chess database.
    
    This function queries the 'moves' table to retrieve all distinct game IDs,
    filtering out NULL values. It's used to split the dataset into training
    and validation sets based on game boundaries (not individual moves).
    
    Args:
        db_path: Path to the SQLite database containing move records.
        
    Returns:
        List of unique game IDs (as integers) present in the database.
        
    Note:
        This function loads all game IDs into memory, which may be memory
        intensive for very large databases. Consider using a generator or
        database cursor for streaming in production scenarios.
    """
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
    """
    Main execution block for training the chess move classifier.
    
    This script performs the following steps:
    1. Load all unique game IDs from the database
    2. Split games into training (80%) and validation (20%) sets
    3. Initialize separate datasets for each split
    4. Calculate class weights based on training distribution
    5. Initialize the MoveClassifierNet model
    6. Load pre-trained weights if available
    7. Train the model with validation and checkpointing
    """
    
    db_path = "chess_bot.db"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ========================================================================
    # STEP 1: Load and Split Game IDs
    # ========================================================================
    # Load and fairly split unique game IDs (80% to 20%)
    all_games = get_all_game_ids(db_path)
    print(f"Total unique games in database: {len(all_games)}")
    
    random.seed(42)  # Fixing seed for reproducibility of the split
    random.shuffle(all_games)
    
    split_idx = int(0.8 * len(all_games))
    train_game_ids = all_games[:split_idx]
    val_game_ids = all_games[split_idx:]
    
    print(f"Games allocated for training (train): {len(train_game_ids)}")
    print(f"Games allocated for validation (val): {len(val_game_ids)}")
    
    # ========================================================================
    # STEP 2: Initialize Datasets
    # ========================================================================
    # Initialize TWO independent dataset objects
    print("\nInitializing training dataset...")
    train_dataset = ChessDataset(db_path=db_path, game_ids=train_game_ids)
    
    print("\nInitializing validation dataset...")
    val_dataset = ChessDataset(db_path=db_path, game_ids=val_game_ids)
    
    # ========================================================================
    # STEP 3: Create DataLoaders
    # ========================================================================
    # Wrap datasets in DataLoader for efficient batching
    num_workers = 4  # Number of parallel data loading workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=2048,  # Large batch size for efficient GPU utilization
        shuffle=True,      # Shuffle training data for better convergence
        num_workers=num_workers,
        pin_memory=True    # Pin memory for faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2048,  # Same batch size for validation
        shuffle=False,     # Don't shuffle validation data
        num_workers=num_workers,
        pin_memory=True
    )
    
    # ========================================================================
    # STEP 4: Calculate Class Weights
    # ========================================================================
    # Calculate class weights STRICTLY from the training subset
    # This ensures the model is trained to handle class imbalance appropriately
    train_targets = [sample[-1] for sample in train_dataset.samples]
    target_counts = Counter(train_targets)
    total_train_samples = len(train_targets)
    
    class_weights = []
    for i in range(len(CLASS_NAMES)):
        count = target_counts.get(i, 1)  # Default to 1 if class not present
        # Square root of inverse frequency (smoothed weights)
        weight = (total_train_samples / count) ** 0.5
        class_weights.append(weight)
        
    # Normalize obtained weights so their mean value equals 1.0
    # This ensures weights are relative and don't scale the loss excessively
    mean_weight = sum(class_weights) / len(class_weights)
    class_weights = [w / mean_weight for w in class_weights]
        
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Calculated (SMOOTHED and NORMALIZED) weights for classes: {class_weights}")
    
    # ========================================================================
    # STEP 5: Initialize Model
    # ========================================================================
    # Assemble the classifier model
    core = ChessCoreNet(in_channels=25)  # Core network with 25 input channels
    model = MoveClassifierNet(core_net=core, num_classes=len(CLASS_NAMES))
    model = model.to(device)
    
    weights_path = "models/weights_classifier.pth"
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Successfully loaded existing weights from {weights_path}.")
    except FileNotFoundError:
        print("Weights not found. Starting model training from scratch.")
    
    # ========================================================================
    # STEP 6: Setup Loss and Optimizer
    # ========================================================================
    # Setup loss function and optimizer with regularization
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Adam with L2 regularization

    # Learning rate scheduler: reduce LR when validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=1, factor=0.5
    )
    
    # ========================================================================
    # STEP 7: Training Loop
    # ========================================================================
    # Main combined training and validation loop
    epochs = 20  # Number of training epochs
    best_val_loss = float('inf')  # Track best validation loss
    
    print(f"\nStarting classifier training on device: {device}")
    for epoch in range(epochs):
        
        # --------------------------------------------------------------------
        # TRAINING PHASE
        # --------------------------------------------------------------------
        model.train()  # Enable dropout and batch normalization training mode
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # Clear gradients from previous iteration
            
            logits, _ = model(inputs)  # Forward pass
            loss = criterion(logits, targets)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            total_train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
        epoch_train_loss = total_train_loss / len(train_loader)
        epoch_train_acc = 100.0 * train_correct / train_total
        print(f"Epoch {epoch+1}/{epochs} | Training Loss: {epoch_train_loss:.4f} | Training Accuracy: {epoch_train_acc:.2f}%")
        
        # --------------------------------------------------------------------
        # VALIDATION PHASE
        # --------------------------------------------------------------------
        model.eval()  # Disable dropout and batch normalization training mode
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # Disable gradient calculation for inference
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_logits, _ = model(val_inputs)
                loss = criterion(val_logits, val_targets)
                total_val_loss += loss.item()
                _, val_predicted = val_logits.max(1)
                val_total += val_targets.size(0)
                val_correct += val_predicted.eq(val_targets).sum().item()
        
        epoch_val_loss = total_val_loss / len(val_loader)
        epoch_val_acc = 100.0 * val_correct / val_total
        print(f"--> Validation | Loss: {epoch_val_loss:.4f} | HONEST Accuracy: {epoch_val_acc:.2f}%")

        # Update learning rate based on validation loss
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current training step (LR): {current_lr:.6f}")
        
        # --------------------------------------------------------------------
        # CHECKPOINTING
        # --------------------------------------------------------------------
        # Save best results (checkpoint)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), weights_path)
            print(f"Weights saved! Validation loss decreased to best: {best_val_loss:.4f}")
        else:
            print(f"Weights not overwritten. Best loss remains: {best_val_loss:.4f}")
            
        print("-" * 65)
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
    """
    Initialize the self-play database.

    Creates the self_play_moves table if it doesn't exist. This table stores
    positions generated during self-play training, including FEN strings,
    MCTS policies, and evaluation values.

    Args:
        db_path: Path to the SQLite database file
    """
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
    """
    Get the next game ID for self-play games.

    Queries the database for the maximum game_id and returns the next value.
    If no games exist, returns 1.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        Next available game ID
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(game_id) FROM self_play_moves")
    row = cursor.fetchone()
    conn.close()
    return (row[0] + 1) if row[0] is not None else 1


def play_one_game_parallel(game_id, bot_weights, num_simulations, temperature, db_path):
    """
    Play a single self-play game using MCTS and save the experience.

    This function:
    1. Initializes an MCTS engine with the bot's neural network
    2. Plays a game move-by-move, using MCTS to select moves
    3. For each move, collects MCTS visit counts and uses them as a policy
    4. Applies temperature scaling to add exploration
    5. Uses the neural network to classify all legal moves and build a policy
    6. Saves each position with its policy and evaluation value

    Args:
        game_id: Unique identifier for this game
        bot_weights: Path to the bot's neural network weights
        num_simulations: Number of MCTS simulations per position
        temperature: Temperature for move selection (higher = more exploration)
        db_path: Path to the SQLite database
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Initialize evaluator and MCTS engine
    evaluator = MoveClassifier(weights_path=bot_weights, device="cpu")
    evaluator.model.eval()
    mcts = MoveClassifierMCTS(classifier=evaluator)

    board = chess.Board()
    game_history = []

    # Play until game over or 300 moves
    while not board.is_game_over() and len(game_history) < 300:
        fen_before = board.fen()

        # Get MCTS visit counts for all legal moves
        _, visit_dict = mcts.search(board, num_simulations=num_simulations)
        if not visit_dict:
            break

        moves_uci_list = list(visit_dict.keys())
        mcts_visits = np.array(list(visit_dict.values()), dtype=np.float32)

        # Temperature-scaled move selection
        if temperature > 0:
            # Softmax over visit counts
            exp_visits = mcts_visits ** (1.0 / (temperature + 1e-8))
            sum_exp = exp_visits.sum()
            move_probabilities = exp_visits / sum_exp if sum_exp > 0 else np.ones_like(exp_visits) / len(exp_visits)

            chosen_move_uci = np.random.choice(moves_uci_list, p=move_probabilities)
            move = chess.Move.from_uci(chosen_move_uci)
        else:
            # Greedy selection: choose move with most visits
            best_idx = np.argmax(mcts_visits)
            move = chess.Move.from_uci(moves_uci_list[best_idx])

        # Calculate normalized MCTS probabilities
        total_visits = mcts_visits.sum()
        mcts_probs = mcts_visits / total_visits if total_visits > 0 else np.ones_like(mcts_visits) / len(mcts_visits)

        # Build class target distribution from neural network evaluation
        moves_san = []
        valid_indices = []
        class_target_distribution = np.zeros(len(CLASS_NAMES), dtype=np.float32)

        for idx, mu in enumerate(moves_uci_list):
            try:
                m_obj = chess.Move.from_uci(mu)
                moves_san.append(board.san(m_obj))
                valid_indices.append(idx)
            except Exception:
                # Invalid move, add its probability to "Unknown" class
                class_target_distribution[0] += mcts_probs[idx]

        if moves_san:
            # Get neural network classification for all moves
            classes, _, _ = evaluator.classify_moves_batch(board, moves_san)
            for cls, idx in zip(classes, valid_indices):
                c_idx = CLASS_NAMES.index(cls) if cls in CLASS_NAMES else 0
                class_target_distribution[c_idx] += mcts_probs[idx]

        # Normalize the distribution
        sum_dist = class_target_distribution.sum()
        if sum_dist > 0:
            class_target_distribution /= sum_dist
        else:
            class_target_distribution = np.ones(len(CLASS_NAMES), dtype=np.float32) / len(CLASS_NAMES)

        # Convert to dictionary
        class_policy_dict = {CLASS_NAMES[i]: float(class_target_distribution[i]) for i in range(len(CLASS_NAMES))}

        # Record this position
        game_history.append((fen_before, move.uci(), class_policy_dict))
        board.push(move)

    # Determine game result value
    result = board.result()
    if result == "1-0":
        value_white = 1.0
    elif result == "0-1":
        value_white = -1.0
    else:
        value_white = 0.0

    # Save each position with its policy and the correct evaluation value
    for fen_before, move_uci, class_policy_dict in game_history:
        current_board = chess.Board(fen_before)
        # Flip the value if it's Black's turn
        actual_value = value_white if current_board.turn == chess.WHITE else -value_white

        cursor.execute("""
            INSERT INTO self_play_moves (game_id, fen_before, move_uci, mcts_policy, result_value)
            VALUES (?, ?, ?, ?, ?)
        """, (game_id, fen_before, move_uci, json.dumps(class_policy_dict), actual_value))

    conn.commit()
    conn.close()


def run_self_play_session(num_games: int, num_simulations: int, temperature: float,
                          db_path: str, bot_weights: str, num_workers: int):
    """
    Run a self-play session generating multiple games.

    This function orchestrates the generation of self-play games, either
    synchronously or using multiprocessing for parallel generation.

    Args:
        num_games: Number of games to generate
        num_simulations: MCTS simulations per position
        temperature: Temperature for move selection
        db_path: Path to the SQLite database
        bot_weights: Path to bot's neural network weights
        num_workers: Number of parallel workers (1 = synchronous)
    """
    print(f"\n----------------------------------------")
    print(f"🎮 PHASE 1: SELF-PLAY GAME GENERATION ({num_games} games)")
    print(f"----------------------------------------")
    print(f"🚀 Starting {num_workers} processes to generate {num_games} games...")

    start_game_id = get_next_game_id(db_path)
    game_ids = list(range(start_game_id, start_game_id + num_games))

    if num_workers <= 1:
        # Synchronous generation
        for gid in tqdm(game_ids, desc="Synchronous generation"):
            play_one_game_parallel(gid, bot_weights, num_simulations, temperature, db_path)
    else:
        # Parallel generation using multiprocessing
        worker_func = partial(play_one_game_parallel, bot_weights=bot_weights,
                              num_simulations=num_simulations, temperature=temperature, db_path=db_path)
        with mp.Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap_unordered(worker_func, game_ids), total=num_games, desc="Game generation"))

    print(f"✅ All {num_games} games generated and saved.")


def train_rl_iteration(epochs: int, batch_size: int, lr: float, alpha: float, db_path: str):
    """
    Train the bot's neural network using reinforcement learning.

    This function:
    1. Loads self-play data from the database
    2. Splits data into train/validation sets
    3. Initializes the neural network model
    4. Loads existing weights if available
    5. Trains using cross-entropy loss for classification and MSE for value
    6. Saves the updated weights

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        alpha: Weight for value loss (total loss = class_loss + alpha * value_loss)
        db_path: Path to the SQLite database
    """
    print("\n" + "-"*40)
    print("🧠 PHASE 2: BOT NEURAL NETWORK TRAINING (RL Update)")
    print("-"*40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📡 TARGET DEVICE FOR TRAINING PHASE: {device}")
    if device.type == "cuda":
        print(f"🔥 Training accelerated on GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    print("-" * 40)

    # Load self-play data
    dataset = ChessSelfPlayDataset(db_path=db_path)

    if len(dataset) < 50:
        print("⚠️ Too little data for training. Skipping...")
        return

    # Split into train/validation sets (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    # Initialize model
    core = ChessCoreNet(in_channels=25)
    model = MoveClassifierNet(core_net=core, num_classes=len(CLASS_NAMES))

    weights_path = "models/weights_bot.pth"
    # Load existing weights if available
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)

    model.to(device)

    # Loss functions and optimizer
    criterion_class = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"   Epoch {epoch+1}/{epochs}", leave=True)
        for inputs, class_targets, value_targets in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            class_targets = class_targets.to(device, non_blocking=True)
            value_targets = value_targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            # Forward pass: get classification logits and value predictions
            class_logits, value_preds = model(inputs)

            # Compute losses
            loss_class = criterion_class(class_logits, class_targets)
            loss_value = criterion_value(value_preds.view(-1), value_targets.view(-1))

            # Total loss is weighted sum
            loss = loss_class + alpha * loss_value
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    # Save trained weights
    torch.save(model.state_dict(), weights_path)
    print(f"✅ BOT weights successfully updated and saved to {weights_path}")


def apply_sliding_window(db_path: str, keep_last_n_games: int):
    """
    Apply sliding window to limit database size.

    Keeps only the most recent N games and deletes older positions.
    This prevents the database from growing indefinitely during long training sessions.

    Args:
        db_path: Path to the SQLite database
        keep_last_n_games: Number of most recent games to keep
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Get all game IDs in descending order
    cursor.execute("SELECT DISTINCT game_id FROM self_play_moves ORDER BY game_id DESC")
    rows = cursor.fetchall()

    if len(rows) > keep_last_n_games:
        # Calculate the cutoff game ID
        last_allowed_id = rows[keep_last_n_games - 1][0]
        # Delete all positions from games before the cutoff
        cursor.execute("DELETE FROM self_play_moves WHERE game_id < ?", (last_allowed_id,))
        deleted_rows = cursor.rowcount
        conn.commit()
        if deleted_rows > 0:
            print(f"🧹 Sliding Window: Deleted {deleted_rows} old positions (keeping top-{keep_last_n_games} games).")
    conn.close()


def run_continuous_loop(iterations: int, games_per_iter: int, sims: int, epochs: int,
                        keep_last_n: int, db_path: str, temperature: float = 1.2,
                        num_workers: int = 1):
    """
    Run the complete reinforcement learning training loop.

    This function orchestrates the entire training process:
    1. Initialize the self-play database
    2. For each iteration:
       a. Generate self-play games using MCTS
       b. Train the neural network on the generated data
       c. Apply sliding window to limit database size
    3. Optionally copy classifier weights as initial bot weights

    Args:
        iterations: Number of training iterations
        games_per_iter: Number of games to generate per iteration
        sims: MCTS simulations per position
        epochs: Training epochs per iteration
        keep_last_n: Number of games to keep in database (sliding window)
        db_path: Path to the SQLite database
        temperature: Temperature for move selection
        num_workers: Number of parallel workers
    """
    init_self_play_db(db_path)
    print(f"\n🚀 STARTING TRAINING LOOP FOR {iterations} ITERATIONS (workers={num_workers})")

    classifier_weights = "models/weights_classifier.pth"
    bot_weights = "models/weights_bot.pth"

    # Copy classifier weights as initial bot weights if bot weights don't exist
    if not os.path.exists(bot_weights):
        if os.path.exists(classifier_weights):
            shutil.copy(classifier_weights, bot_weights)
            print(f"📦 Base weights copied from {classifier_weights} to {bot_weights}")
        else:
            print(f"⚠️ Warning: {bot_weights} not found. Training will start from scratch!")

    for i in range(iterations):
        print(f"\n{'='*50}\n🌟 GLOBAL ITERATION {i+1}/{iterations}\n{'='*50}")

        # 1. Generate self-play games
        run_self_play_session(num_games=games_per_iter, num_simulations=sims,
                              temperature=temperature, db_path=db_path,
                              bot_weights=bot_weights, num_workers=num_workers)

        # 2. Train network on GPU
        train_rl_iteration(epochs=epochs, batch_size=64, lr=1e-4, alpha=1.0, db_path=db_path)

        # 3. Apply sliding window to limit database size
        apply_sliding_window(db_path, keep_last_n)
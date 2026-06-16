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
from engine.unified_mcts import UnifiedMCTS
from models.unified_chess_nets import ChessCoreNet, UnifiedMoveClassifierNet, LookaheadMoveData
from classifiers.classification_config import CLASS_NAMES
from classifiers.self_play_dataset import ChessSelfPlayDataset, LookaheadChessSelfPlayDataset


def init_self_play_db(db_path: str):
    """
    Initialize the self-play database with lookahead support.

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
    """
    Get the next game ID for self-play games.

    Args:
        db_path: Path to the SQLite database

    Returns:
        Next available game ID
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(game_id) FROM self_play_moves")
    row = cursor.fetchone()
    conn.close()
    return (row[0] + 1) if row[0] is not None else 1


def play_one_game_with_lookahead(game_id, bot_weights, num_simulations, temperature, db_path, lookahead_depth=2):
    """
    Play a single self-play game using MCTS and save the experience with lookahead data.

    This function:
    1. Initializes an MCTS engine with the bot's neural network
    2. Plays a game move-by-move, using MCTS to select moves
    3. For each move, collects MCTS visit counts and uses them as a policy
    4. Applies temperature scaling to add exploration
    5. Uses the neural network to classify all legal moves and build a policy
    6. Generates lookahead sequences for training
    7. Saves each position with its policy, evaluation value, and lookahead data

    Args:
        game_id: Unique identifier for this game
        bot_weights: Path to the bot's neural network weights
        num_simulations: Number of MCTS simulations per position
        temperature: Temperature for move selection (higher = more exploration)
        db_path: Path to the SQLite database
        lookahead_depth: Depth of lookahead sequences to generate
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Initialize evaluator and MCTS engine
    evaluator = MoveClassifier(weights_path=bot_weights, device="cpu")
    evaluator.model.eval()
    mcts = UnifiedMCTS(unified_model=evaluator.model, cpuct=2.0, max_simulations=num_simulations, top_moves_ratio=0.3, policy_output_dim=64)

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
            exp_visits = mcts_visits ** (1.0 / (temperature + 1e-8))
            sum_exp = exp_visits.sum()
            move_probabilities = exp_visits / sum_exp if sum_exp > 0 else np.ones_like(exp_visits) / len(exp_visits)

            chosen_move_uci = np.random.choice(moves_uci_list, p=move_probabilities)
            move = chess.Move.from_uci(chosen_move_uci)
        else:
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
                class_target_distribution[0] += mcts_probs[idx]

        if moves_san:
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

        class_policy_dict = {CLASS_NAMES[i]: float(class_target_distribution[i]) for i in range(len(CLASS_NAMES))}

        # Generate lookahead sequences
        lookahead_data = generate_lookahead_sequences(board, moves_san, mcts_probs, moves_uci_list, lookahead_depth, evaluator)

        # Record this position
        game_history.append({
            'fen_before': fen_before,
            'move_uci': move.uci(),
            'class_policy_dict': class_policy_dict,
            'lookahead_data': lookahead_data
        })
        board.push(move)

    # Determine game result value
    result = board.result()
    if result == "1-0":
        value_white = 1.0
    elif result == "0-1":
        value_white = -1.0
    else:
        value_white = 0.0

    # Save each position with its policy, evaluation value, and lookahead data
    for history_entry in game_history:
        fen_before = history_entry['fen_before']
        move_uci = history_entry['move_uci']
        class_policy_dict = history_entry['class_policy_dict']
        lookahead_data = history_entry['lookahead_data']

        current_board = chess.Board(fen_before)
        actual_value = value_white if current_board.turn == chess.WHITE else -value_white

        # Save basic data
        cursor.execute("""
            INSERT INTO self_play_moves (game_id, fen_before, move_uci, mcts_policy, result_value)
            VALUES (?, ?, ?, ?, ?)
        """, (game_id, fen_before, move_uci, json.dumps(class_policy_dict), actual_value))

        # Save lookahead data if available
        if lookahead_data:
            lookahead_depth = lookahead_data.get('lookahead_depth', 0)
            future_moves = lookahead_data.get('future_moves', [])
            final_classification = lookahead_data.get('final_classification', 'Good')
            move_sequence_classes = lookahead_data.get('move_sequence_classes', [CLASS_NAMES[3]])

            cursor.execute("""
                INSERT INTO self_play_moves (game_id, fen_before, move_uci, mcts_policy, result_value,
                                            lookahead_depth, future_moves, final_classification, move_sequence_classes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (game_id, fen_before, move_uci, json.dumps(class_policy_dict), actual_value,
                  lookahead_depth, json.dumps(future_moves), final_classification, json.dumps(move_sequence_classes), None))

    conn.commit()
    conn.close()


def generate_lookahead_sequences(board, moves_san, mcts_probs, moves_uci_list, lookahead_depth, evaluator):
    """
    Generate lookahead sequences for a given position.

    Args:
        board: Current chess board
        moves_san: List of legal moves in SAN notation
        mcts_probs: MCTS probabilities for each move (indexed by position in moves_uci_list)
        moves_uci_list: List of UCI moves that MCTS explored
        lookahead_depth: How many moves ahead to look
        evaluator: Move classifier for evaluating moves

    Returns:
        dict with lookahead sequence data
    """
    if lookahead_depth <= 0:
        return None

    # Generate lookahead sequences
    sequences = []
    for depth in range(1, lookahead_depth + 1):
        current_board = board.copy()
        sequence_moves = []
        sequence_classes = []
        sequence_probs = []

        for _ in range(depth):
            if current_board.is_game_over():
                break

            # Get legal moves
            legal_moves = list(current_board.legal_moves)
            if not legal_moves:
                break

            # Convert to SAN and probabilities
            move_sans = [current_board.san(m) for m in legal_moves]

            # Create a mapping from UCI move to index in mcts_probs
            # mcts_probs is indexed by position in moves_uci_list (MCTS explored moves only)
            move_to_idx = {move: i for i, move in enumerate(moves_uci_list)}
            
            # Filter to only include moves that were explored by MCTS
            mcts_legal_moves = [m for m in legal_moves if m in move_to_idx]
            if not mcts_legal_moves:
                # If no moves were explored, use uniform distribution
                move_probs = [1.0 / len(move_sans)] * len(move_sans)
            else:
                # Get probabilities for the legal moves that MCTS explored
                mcts_indices = [move_to_idx[m] for m in mcts_legal_moves]
                move_probs = [mcts_probs[i] for i in mcts_indices]

            # Normalize probabilities
            total_prob = sum(move_probs)
            if total_prob > 0:
                move_probs = [p / total_prob for p in move_probs]
            else:
                move_probs = [1.0 / len(move_sans)] * len(move_sans)

            # Select next move based on probabilities
            chosen_idx = np.random.choice(len(move_sans), p=move_probs)
            chosen_move_san = move_sans[chosen_idx]

            sequence_moves.append(chosen_move_san)
            sequence_probs.append(move_probs[chosen_idx])

            # Evaluate the move
            try:
                classes, _, _ = evaluator.classify_moves_batch(current_board, [chosen_move_san])
                sequence_classes.append(classes[0] if classes else "Good")
            except Exception:
                sequence_classes.append("Good")

            # Make the move
            try:
                move = current_board.parse_san(chosen_move_san)
                current_board.push(move)
            except Exception:
                break

        if sequence_moves:
            sequences.append({
                'moves': sequence_moves,
                'classes': sequence_classes,
                'probs': sequence_probs
            })

    if not sequences:
        return None

    # Use the longest sequence
    best_sequence = max(sequences, key=lambda x: len(x['moves']))

    return {
        'lookahead_depth': len(best_sequence['moves']),
        'future_moves': best_sequence['moves'],
        'final_classification': best_sequence['classes'][-1] if best_sequence['classes'] else "Good",
        'move_sequence_classes': best_sequence['classes']
    }


def run_self_play_session(num_games: int, num_simulations: int, temperature: float,
                          db_path: str, bot_weights: str, num_workers: int, lookahead_depth: int = 2):
    """
    Run a self-play session generating multiple games with lookahead data.

    Args:
        num_games: Number of games to generate
        num_simulations: MCTS simulations per position
        temperature: Temperature for move selection
        db_path: Path to the SQLite database
        bot_weights: Path to bot's neural network weights
        num_workers: Number of parallel workers (1 = synchronous)
        lookahead_depth: Depth of lookahead sequences to generate
    """
    print(f"\n----------------------------------------")
    print(f"🎮 PHASE 1: SELF-PLAY GAME GENERATION ({num_games} games)")
    print(f"----------------------------------------")
    print(f"🚀 Starting {num_workers} processes to generate {num_games} games with lookahead depth={lookahead_depth}...")

    start_game_id = get_next_game_id(db_path)
    game_ids = list(range(start_game_id, start_game_id + num_games))

    if num_workers <= 1:
        for gid in tqdm(game_ids, desc="Synchronous generation"):
            play_one_game_with_lookahead(gid, bot_weights, num_simulations, temperature, db_path, lookahead_depth)
    else:
        worker_func = partial(play_one_game_with_lookahead, bot_weights=bot_weights,
                              num_simulations=num_simulations, temperature=temperature, db_path=db_path,
                              lookahead_depth=lookahead_depth)
        with mp.Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap_unordered(worker_func, game_ids), total=num_games, desc="Game generation"))

    print(f"✅ All {num_games} games generated and saved with lookahead data.")


def train_unified_model(epochs: int, batch_size: int, lr: float, alpha: float, policy_weight: float, db_path: str, use_lookahead: bool = False, use_stockfish_policy: bool = False):
    """
    Train the unified model with classification, value, and policy heads.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        alpha: Weight for value loss
        policy_weight: Weight for policy loss
        db_path: Path to the SQLite database
        use_lookahead: Whether to use lookahead data for training
        use_stockfish_policy: Whether to use Stockfish policy data for training
    """
    print("\n" + "-"*40)
    print("🧠 PHASE 2: UNIFIED MODEL TRAINING (RL + Policy)")
    print("-" * 40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📡 TARGET DEVICE: {device}")

    # Load dataset
    if use_lookahead:
        dataset = LookaheadChessSelfPlayDataset(db_path=db_path, device=device)
    else:
        dataset = ChessSelfPlayDataset(db_path=db_path)

    if len(dataset) < 50:
        print("⚠️ Too little data for training. Skipping...")
        return

    # Split data
    train_size = int(0.9 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    # Initialize unified model
    core = ChessCoreNet(in_channels=25)
    model = UnifiedMoveClassifierNet(core_net=core, num_classes=len(CLASS_NAMES), policy_output_dim=64)

    weights_path = "models/weights_bot.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)

    model.to(device)
    model.eval()

    # Loss functions
    criterion_class = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    criterion_policy = nn.KLDivLoss(log_target=True)

    # Create policy targets from dataset
    if use_stockfish_policy:
        print("📊 Loading Stockfish policy data...")
        from classifiers.stockfish_policy_dataset import StockfishPolicyDataset
        policy_dataset = StockfishPolicyDataset(json_path="stockfish_policy_data.json", device=device)
        policy_targets = torch.tensor([d['policy_score'] for d in policy_dataset.data], dtype=torch.float32)
        # Convert single scores to distributions (one-hot-like based on move index)
        # For now, use the score as a scalar policy target
        policy_targets = policy_targets.unsqueeze(1)  # Shape: (N, 1)
    else:
        policy_dataset = LookaheadChessSelfPlayDataset(db_path=db_path, device=device)
        policy_samples = []
        for sample in policy_dataset:
            policy_samples.append(sample['policy_dict'])
        policy_targets = torch.tensor([list(p.values()) for p in policy_samples], dtype=torch.float32)
        policy_targets = policy_targets / policy_targets.sum(dim=1, keepdim=True)  # Normalize

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
            class_logits, value_preds, policy_logits = model(inputs)

            loss_class = criterion_class(class_logits, class_targets)
            loss_value = criterion_value(value_preds.view(-1), value_targets.view(-1))

            # Policy loss - different handling for Stockfish vs lookahead data
            if use_stockfish_policy:
                # Stockfish policy: scalar score, use MSE instead of KL divergence
                policy_targets = policy_targets.to(device).view(-1, 1)  # Shape: (N, 1)
                loss_policy = criterion_value(F.softmax(policy_logits, dim=1).squeeze(1), policy_targets)
            else:
                # Lookahead policy: distribution, use KL divergence
                loss_policy = criterion_policy(F.log_softmax(policy_logits, dim=1), policy_targets.to(device))

            loss = loss_class + alpha * loss_value + policy_weight * loss_policy
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    # Save trained weights
    torch.save(model.state_dict(), weights_path)
    print(f"✅ Unified model weights saved to {weights_path}")


def apply_sliding_window(db_path: str, keep_last_n_games: int):
    """
    Apply sliding window to limit database size.

    Args:
        db_path: Path to the SQLite database
        keep_last_n_games: Number of most recent games to keep
    """
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
            print(f"🧹 Sliding Window: Deleted {deleted_rows} old positions (keeping top-{keep_last_n_games} games).")
    conn.close()


def run_continuous_loop(iterations: int, games_per_iter: int, sims: int, epochs: int,
                        keep_last_n: int, db_path: str, temperature: float = 1.2,
                        num_workers: int = 1, use_lookahead: bool = False, lookahead_depth: int = 2,
                        use_stockfish_policy: bool = False):
    """
    Run the complete reinforcement learning training loop.

    Args:
        iterations: Number of training iterations
        games_per_iter: Number of games to generate per iteration
        sims: MCTS simulations per position
        epochs: Training epochs per iteration
        keep_last_n: Number of games to keep in database
        db_path: Path to the SQLite database
        temperature: Temperature for move selection
        num_workers: Number of parallel workers
        use_lookahead: Whether to use lookahead data
        lookahead_depth: Depth of lookahead sequences
        use_stockfish_policy: Whether to use Stockfish policy data for training
    """
    init_self_play_db(db_path)
    print(f"\n🚀 STARTING TRAINING LOOP FOR {iterations} ITERATIONS (workers={num_workers}, lookahead={use_lookahead}, stockfish_policy={use_stockfish_policy})")

    classifier_weights = "models/weights_classifier.pth"
    bot_weights = "models/weights_bot.pth"

    if not os.path.exists(bot_weights):
        if os.path.exists(classifier_weights):
            shutil.copy(classifier_weights, bot_weights)
            print(f"📦 Base weights copied from {classifier_weights} to {bot_weights}")
        else:
            print(f"⚠️ Warning: {bot_weights} not found. Training will start from scratch!")

    for i in range(iterations):
        print(f"\n{'='*50}\n🌟 GLOBAL ITERATION {i+1}/{iterations}\n{'='*50}")

        run_self_play_session(num_games=games_per_iter, num_simulations=sims,
                              temperature=temperature, db_path=db_path,
                              bot_weights=bot_weights, num_workers=num_workers,
                              lookahead_depth=lookahead_depth if use_lookahead else 0)

        train_unified_model(epochs=epochs, batch_size=64, lr=1e-3, alpha=0.5,
                            policy_weight=1.0, db_path=db_path, use_lookahead=use_lookahead,
                            use_stockfish_policy=use_stockfish_policy)

        apply_sliding_window(db_path, keep_last_n_games=keep_last_n)

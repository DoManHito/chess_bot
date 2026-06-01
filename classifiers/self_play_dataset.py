"""
Self-Play Dataset for Reinforcement Learning Training.

This module implements a PyTorch Dataset class that loads self-play chess positions
from a SQLite database. Each sample consists of:
- Board state before and after a move (as 25-channel tensors)
- MCTS policy distribution over move classes
- Evaluation value from the game result

The dataset is used to train the neural network classifier through reinforcement
learning, where the network learns to predict move quality based on game outcomes.
"""

import sqlite3
import json
import torch
from torch.utils.data import Dataset
import chess
import numpy as np
from classifiers.classification_config import CLASS_NAMES


class ChessSelfPlayDataset(Dataset):
    """
    PyTorch Dataset for self-play chess positions.

    Loads game data from a SQLite database containing self-play positions generated
    by the MCTS engine. Each sample represents a position before and after a move,
    along with the MCTS policy distribution and evaluation value.

    The dataset is designed for reinforcement learning training where the network
    learns to classify move quality based on actual game outcomes.

    Args:
        db_path: Path to SQLite database containing self_play_moves table

    Sample Format:
        - input_tensor: (25, 8, 8) tensor representing board states
        - class_target: (6,) tensor with MCTS policy distribution
        - value_target: scalar evaluation value
    """
    def __init__(self, db_path="chess_bot.db"):
        """
        Initialize the self-play dataset.

        Args:
            db_path: Path to SQLite database (default: "chess_bot.db")
        """
        self.db_path = db_path
        self.samples = []

        # Set to single thread for database safety during data loading
        torch.set_num_threads(1)
        self._load_data()

    def _load_data(self):
        """
        Load self-play positions from the database.

        Connects to the SQLite database and queries the self_play_moves table
        for all positions. Each row contains:
        - fen_before: FEN string before the move
        - move_uci: UCI notation of the move
        - mcts_policy: JSON string with MCTS policy distribution
        - result_value: Evaluation value from game result

        The data is parsed and stored as a list of dictionaries, each containing:
        - fen: FEN string of the starting position
        - move_uci: UCI notation of the move
        - policy_dict: Dictionary with MCTS policy distribution
        - value: Float evaluation value
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Verify the self_play_moves table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='self_play_moves'")
        if not cursor.fetchone():
            print("Table self_play_moves not found!")
            conn.close()
            return

        print("Loading Self-Play experience from database...")
        # Query all positions from the database
        cursor.execute("SELECT fen_before, move_uci, mcts_policy, result_value FROM self_play_moves")
        rows = cursor.fetchall()
        conn.close()

        # Parse each row into a sample dictionary
        for fen_before, move_uci, mcts_policy_json, result_value in rows:
            self.samples.append({
                "fen": fen_before,
                "move_uci": move_uci,
                "policy_dict": json.loads(mcts_policy_json),
                "value": float(result_value)
            })
        print(f"Successfully loaded {len(self.samples)} states for training.")

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            Number of loaded self-play positions
        """
        return len(self.samples)

    def _board_to_tensor_static(self, board_before: chess.Board, board_after: chess.Board) -> torch.Tensor:
        """
        Convert chess board states to a 25-channel tensor representation.

        This method creates a tensor that encodes both the board state before and
        after a move, allowing the neural network to learn the effect of moves.

        Tensor Structure (25 channels x 8x8 board):
        - Channels 0-5: White pieces before move (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
        - Channels 6-11: Black pieces before move
        - Channels 12-17: White pieces after move
        - Channels 18-23: Black pieces after move
        - Channel 24: Turn indicator (1 = White to move, 0 = Black to move)

        Args:
            board_before: Chess board state before the move
            board_after: Chess board state after the move

        Returns:
            torch.Tensor of shape (25, 8, 8) with board representation
        """
        tensor = np.zeros((25, 8, 8), dtype=np.float32)

        pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        # Encode white pieces from board_before (channels 0-5)
        for i, piece in enumerate(pieces):
            for sq in board_before.pieces(piece, chess.WHITE):
                tensor[i, chess.square_rank(sq), chess.square_file(sq)] = 1.0
            # Encode black pieces from board_before (channels 6-11)
            for sq in board_before.pieces(piece, chess.BLACK):
                tensor[i + 6, chess.square_rank(sq), chess.square_file(sq)] = 1.0

        # Encode white pieces from board_after (channels 12-17)
        for i, piece in enumerate(pieces):
            for sq in board_after.pieces(piece, chess.WHITE):
                tensor[i + 12, chess.square_rank(sq), chess.square_file(sq)] = 1.0
            # Encode black pieces from board_after (channels 18-23)
            for sq in board_after.pieces(piece, chess.BLACK):
                tensor[i + 18, chess.square_rank(sq), chess.square_file(sq)] = 1.0

        # Layer 24: Whose turn BEFORE (White=1, Black=0)
        if board_before.turn == chess.WHITE:
            tensor[24, :, :] = 1.0

        return torch.from_numpy(tensor)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        For each sample, this method:
        1. Creates board states before and after the move
        2. Converts boards to tensor representation
        3. Extracts MCTS policy distribution
        4. Normalizes the distribution if needed
        5. Returns input tensor, class target distribution, and value target

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (input_tensor, class_target, value_target):
            - input_tensor: (25, 8, 8) tensor with board representation
            - class_target: (6,) tensor with MCTS policy distribution
            - value_target: scalar evaluation value
        """
        torch.set_num_threads(1)

        sample = self.samples[idx]
        board = chess.Board(sample["fen"])
        value_target = torch.tensor(sample["value"], dtype=torch.float32)

        # Create board after the move
        board_after = board.copy()
        try:
            move = chess.Move.from_uci(sample["move_uci"])
            if move in board_after.legal_moves:
                board_after.push(move)
        except Exception:
            pass

        # Convert board states to tensor
        input_tensor = self._board_to_tensor_static(board, board_after)

        # Extract MCTS policy distribution
        policy_dict = sample["policy_dict"]
        class_target_distribution = np.zeros(len(CLASS_NAMES), dtype=np.float32)

        for i, class_name in enumerate(CLASS_NAMES):
            class_target_distribution[i] = policy_dict.get(class_name, 0.0)

        # Normalize distribution if sum is too small (fallback to uniform)
        sum_dist = class_target_distribution.sum()
        if sum_dist <= 0:
            class_target_distribution = np.ones(len(CLASS_NAMES), dtype=np.float32) / len(CLASS_NAMES)

        return input_tensor, torch.tensor(class_target_distribution, dtype=torch.float32), value_target
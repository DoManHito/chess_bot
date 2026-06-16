"""
Self-Play Dataset for Reinforcement Learning Training with Lookahead Support.

This module implements PyTorch Dataset classes that load self-play chess positions
from a SQLite database. Each sample consists of:
- Board state before and after a move (as 25-channel tensors)
- MCTS policy distribution over move classes
- Evaluation value from the game result
- Lookahead sequences (optional) for training on move consequences

The dataset is used to train the unified neural network through reinforcement
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
    PyTorch Dataset for self-play chess positions (legacy format).

    Loads game data from a SQLite database containing self-play positions generated
    by the MCTS engine. Each sample represents a position before and after a move,
    along with the MCTS policy distribution and evaluation value.

    Args:
        db_path: Path to SQLite database containing self_play_moves table
        device: Torch device to use
    """
    def __init__(self, db_path="chess_bot.db", device='cpu'):
        self.db_path = db_path
        self.device = torch.device(device)
        self.samples = []
        torch.set_num_threads(1)
        self._load_data()

    def _load_data(self):
        """Load self-play positions from the database (legacy format)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check for both legacy and new table formats
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='self_play_moves'")
        if not cursor.fetchone():
            print("Table self_play_moves not found!")
            conn.close()
            return

        print("Loading Self-Play experience from database...")
        
        # Try new format first (with lookahead data)
        cursor.execute("""
            SELECT fen_before, move_uci, mcts_policy, result_value, 
                   lookahead_depth, future_moves, final_classification, move_sequence_classes
            FROM self_play_moves
        """)
        rows = cursor.fetchall()
        conn.close()

        # Parse each row into a sample dictionary
        for fen_before, move_uci, mcts_policy_json, result_value, lookahead_depth, future_moves, final_classification, move_sequence_classes in rows:
            sample = {
                "fen": fen_before,
                "move_uci": move_uci,
                "policy_dict": json.loads(mcts_policy_json) if mcts_policy_json else {},
                "value": float(result_value) if result_value else 0.0
            }
            
            # Add lookahead data if present
            if lookahead_depth and future_moves:
                sample["lookahead_depth"] = int(lookahead_depth)
                sample["future_moves"] = json.loads(future_moves) if future_moves else []
                sample["final_classification"] = final_classification if final_classification else "Good"
                sample["move_sequence_classes"] = json.loads(move_sequence_classes) if move_sequence_classes else [CLASS_NAMES[3]]  # Default to "Good"
            
            self.samples.append(sample)
        
        print(f"Successfully loaded {len(self.samples)} states for training.")

    def __len__(self):
        return len(self.samples)

    def _board_to_tensor_static(self, board_before: chess.Board, board_after: chess.Board) -> torch.Tensor:
        """Convert chess board states to a 25-channel tensor representation."""
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


class LookaheadMoveSequence:
    """
    Data class for lookahead move sequences.

    Represents a move with its lookahead consequences for training the unified model.
    """
    def __init__(
        self,
        board_fen: str,
        move_san: str,
        lookahead_depth: int = 2,
        future_moves: list = None,
        final_evaluation: float = 0.0,
        final_classification: str = "Good",
        move_sequence_classifications: list = None
    ):
        self.board_fen = board_fen
        self.move_san = move_san
        self.lookahead_depth = lookahead_depth
        self.future_moves = future_moves or []
        self.final_evaluation = final_evaluation
        self.final_classification = final_classification
        self.move_sequence_classifications = move_sequence_classifications or [move_san]

    def to_dict(self):
        return {
            'board_fen': self.board_fen,
            'move_san': self.move_san,
            'lookahead_depth': self.lookahead_depth,
            'future_moves': self.future_moves,
            'final_evaluation': self.final_evaluation,
            'final_classification': self.final_classification,
            'move_sequence_classifications': self.move_sequence_classifications
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            board_fen=data['board_fen'],
            move_san=data['move_san'],
            lookahead_depth=data.get('lookahead_depth', 2),
            future_moves=data.get('future_moves', []),
            final_evaluation=data.get('final_evaluation', 0.0),
            final_classification=data.get('final_classification', 'Good'),
            move_sequence_classifications=data.get('move_sequence_classifications', [data['move_san']])
        )


class LookaheadChessSelfPlayDataset(Dataset):
    """
    PyTorch Dataset for self-play chess positions with lookahead sequences.

    This dataset is designed for training the unified model with lookahead capability.
    Each sample contains:
    - Board state before a move
    - The move to evaluate
    - Lookahead sequence (future moves and their consequences)
    - Final evaluation and classification after the lookahead sequence

    Args:
        db_path: Path to SQLite database containing self_play_moves table
        device: Torch device to use
    """
    def __init__(self, db_path="chess_bot.db", device='cpu'):
        self.db_path = db_path
        self.device = torch.device(device)
        self.samples = []
        torch.set_num_threads(1)
        self._load_data()

    def _load_data(self):
        """Load self-play positions with lookahead data from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check for table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='self_play_moves'")
        if not cursor.fetchone():
            print("Table self_play_moves not found!")
            conn.close()
            return

        print("Loading Self-Play experience with lookahead data...")
        
        cursor.execute("""
            SELECT fen_before, move_uci, mcts_policy, result_value, 
                   lookahead_depth, future_moves, final_classification, move_sequence_classes
            FROM self_play_moves
        """)
        rows = cursor.fetchall()
        conn.close()

        # Parse each row into a sample dictionary
        for fen_before, move_uci, mcts_policy_json, result_value, lookahead_depth, future_moves, final_classification, move_sequence_classes in rows:
            sample = {
                "fen": fen_before,
                "move_uci": move_uci,
                "policy_dict": json.loads(mcts_policy_json) if mcts_policy_json else {},
                "value": float(result_value) if result_value else 0.0
            }
            
            # Add lookahead data if present
            if lookahead_depth and future_moves:
                sample["lookahead_depth"] = int(lookahead_depth)
                sample["future_moves"] = json.loads(future_moves) if future_moves else []
                sample["final_classification"] = final_classification if final_classification else "Good"
                sample["move_sequence_classes"] = json.loads(move_sequence_classes) if move_sequence_classes else [CLASS_NAMES[3]]
            
            self.samples.append(sample)
        
        print(f"Successfully loaded {len(self.samples)} states with lookahead data for training.")

    def __len__(self):
        return len(self.samples)

    def _board_to_tensor_static(self, board_before: chess.Board, board_after: chess.Board) -> torch.Tensor:
        """Convert chess board states to a 25-channel tensor representation."""
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
        Get a single sample from the dataset with lookahead data.

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

        policy_target = np.ones(64, dtype=np.float32) / 64.0
        policy_target_tensor = torch.from_numpy(policy_target)

        return (
            input_tensor,                             # inputs
            torch.from_numpy(class_target_distribution), # class_targets
            value_target,                             # value_targets
            policy_target_tensor                      # policy_targets
        )

"""
Unified Chess Neural Network Architecture with Lookahead Capability.

This module implements a unified deep neural network for chess that combines:
1. Move Classification (Best, Excellent, Good, Inaccuracy, Mistake, Blunder)
2. Position Evaluation (-1 to 1)
3. Policy Prediction (move probabilities for MCTS acceleration)

The model also supports lookahead learning - training on sequences of moves
to understand the consequences of each move.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np
from typing import List, Tuple


class ChessResidualBlock(nn.Module):
    """Residual block for chess neural network."""
    def __init__(self, channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class ChessCoreNet(nn.Module):
    """Core convolutional network for chess board representation processing."""
    def __init__(self, in_channels=25, num_blocks=4, hidden_channels=128):
        super().__init__()
        self.conv_init = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(hidden_channels)
        self.blocks = nn.ModuleList([ChessResidualBlock(hidden_channels) for _ in range(num_blocks)])

    def forward(self, x):
        x = F.relu(self.bn_init(self.conv_init(x)))
        for block in self.blocks:
            x = block(x)
        return x


class LookaheadMoveData:
    """Data class for lookahead move sequences."""
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


class UnifiedMoveClassifierNet(nn.Module):
    """
    Unified multi-task neural network for chess move classification, evaluation, and policy prediction.

    This network takes a 25-channel board representation and produces three outputs:
    1. **Classification logits**: 6-class output for move quality
    2. **Value prediction**: Scalar evaluation normalized to [-1, 1]
    3. **Policy probabilities**: Move probabilities for MCTS acceleration

    Input Tensor Shape: (batch_size, 25, 8, 8)
    - Layers 0-5: White pieces (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
    - Layers 6-11: Black pieces
    - Layers 12-17: White pieces after move
    - Layers 18-23: Black pieces after move
    - Layer 24: Turn indicator (1 = White to move, 0 = Black to move)

    Output Shapes:
    - class_logits: (batch_size, 6) - raw logits for 6 move quality classes
    - value: (batch_size,) - normalized evaluation score in [-1, 1]
    - policy: (batch_size, policy_output_dim) - policy probabilities
    """
    def __init__(self, core_net, num_classes=6, hidden_channels=128, policy_output_dim=64):
        super().__init__()
        self.core = core_net
        self.conv_reduce = nn.Conv2d(hidden_channels, 16, kernel_size=1)
        self.bn_reduce = nn.BatchNorm2d(16)

        # Classification head: 16*8*8=1024 -> 256 -> num_classes
        self.fc_class1 = nn.Linear(16 * 8 * 8, 256)
        self.dropout = nn.Dropout(p=0.4)
        self.fc_class2 = nn.Linear(256, num_classes)

        # Value head: 16*8*8=1024 -> 32 -> 1
        self.fc_value1 = nn.Linear(16 * 8 * 8, 32)
        self.fc_value2 = nn.Linear(32, 1)

        # Policy head: 16*8*8=1024 -> 128 -> 32 -> policy_output_dim
        self.fc_policy1 = nn.Linear(16 * 8 * 8, 128)
        self.fc_policy2 = nn.Linear(128, 32)
        self.policy_output = nn.Linear(32, policy_output_dim)
        self.policy_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass producing classification logits, value prediction, and policy.

        Args:
            x: Input tensor of shape (batch_size, 25, 8, 8)

        Returns:
            Tuple of (class_logits, value, policy):
            - class_logits: (batch_size, num_classes) raw classification logits
            - value: (batch_size,) normalized evaluation in [-1, 1]
            - policy: (batch_size, policy_output_dim) policy probabilities
        """
        features = self.core(x)
        x_shared = F.relu(self.bn_reduce(self.conv_reduce(features)))
        flattened = x_shared.view(x_shared.size(0), -1)

        # Classification branch
        xc = F.relu(self.fc_class1(flattened))
        xc = self.dropout(xc)
        class_logits = self.fc_class2(xc)

        # Value prediction branch
        xv = F.relu(self.fc_value1(flattened))
        value = torch.tanh(self.fc_value2(xv)).squeeze(-1)

        # Policy branch
        xp = F.relu(self.fc_policy1(flattened))
        xp = F.relu(self.fc_policy2(xp))
        policy_logits = self.policy_output(xp)
        policy = self.policy_softmax(policy_logits)

        return class_logits, value, policy

    def classify_moves_batch(self, board_before: chess.Board, moves_san: List[str]) -> Tuple[List[str], List[float], List[float]]:
        """
        Classify multiple moves in a single batch.

        Args:
            board_before: Chess board state before the moves
            moves_san: List of moves in UCI or SAN notation

        Returns:
            Tuple of (classes, confidences, values):
            - classes: List of classification strings
            - confidences: List of confidence scores
            - values: List of value predictions
        """
        tensors = []
        for move_notation in moves_san:
            board_after = board_before.copy()
            try:
                # Try to parse as UCI first (e.g., "e2e4"), then as SAN
                move = board_after.parse_uci(move_notation)
                board_after.push(move)
                tensor = self._board_to_tensor_static(board_before, board_after).squeeze(0)
            except Exception:
                # Fall back to SAN parsing
                try:
                    move = board_after.parse_san(move_notation)
                    board_after.push(move)
                    tensor = self._board_to_tensor_static(board_before, board_after).squeeze(0)
                except Exception:
                    tensor = torch.zeros(25, 8, 8)
            tensors.append(tensor)

        if not tensors:
            return [], [], []

        batch = torch.stack(tensors)
        with torch.no_grad():
            logits, values, _ = self(batch)
            probs = torch.softmax(logits, dim=1)
            max_probs, indices = probs.max(dim=1)
            classes = [CLASS_NAMES[idx] for idx in indices.tolist()]
            confidences = max_probs.tolist()
            value_list = values.flatten().tolist()

        return classes, confidences, value_list

    def get_policy(self, board: chess.Board, top_k: int = 64) -> torch.Tensor:
        """
        Get policy probabilities for top K moves.

        Args:
            board: Chess board position
            top_k: Number of top moves to return (default: 64)

        Returns:
            Tensor of shape (top_k,) with policy probabilities
        """
        tensor = self._board_to_tensor_static(board, board).unsqueeze(0)
        with torch.no_grad():
            _, _, policy = self(tensor)
        return policy[0, :top_k]

    @staticmethod
    def _board_to_tensor_static(board_before: chess.Board, board_after: chess.Board) -> torch.Tensor:
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

        return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)


CLASS_NAMES = ["Blunder", "Mistake", "Inaccuracy", "Good", "Excellent", "Best"]

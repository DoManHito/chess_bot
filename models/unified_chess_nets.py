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
    """Core convolutional network for chess board - ТЕПЕРЬ 13 КАНАЛОВ ДЛЯ ВАРИАНТА А"""
    def __init__(self, in_channels=13, num_blocks=6, hidden_channels=128): # Увеличили блоки до 6 для точности
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
    def __init__(self, core_net, hidden_channels=128):
        super().__init__()
        self.core = core_net
        self.conv_reduce = nn.Conv2d(hidden_channels, 16, kernel_size=1)
        self.bn_reduce = nn.BatchNorm2d(16)

        # Value head: Оценка позиции от -1.0 (черные ведут) до +1.0 (белые ведут)
        self.fc_value1 = nn.Linear(16 * 8 * 8, 64)
        self.fc_value2 = nn.Linear(64, 1)

        # Policy head: Фиксированный размер 4096 (64 начальных кв. * 64 конечных кв.)
        self.fc_policy1 = nn.Linear(16 * 8 * 8, 256)
        self.policy_output = nn.Linear(256, 4096)
        self.policy_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        features = self.core(x)
        x_shared = F.relu(self.bn_reduce(self.conv_reduce(features)))
        flattened = x_shared.view(x_shared.size(0), -1)

        # Value branch
        xv = F.relu(self.fc_value1(flattened))
        value = torch.tanh(self.fc_value2(xv)).squeeze(-1)

        # Policy branch
        xp = F.relu(self.fc_policy1(flattened))
        policy_logits = self.policy_output(xp)

        return None, value, policy_logits

    def get_policy(self, board: chess.Board, top_k: int = 64) -> torch.Tensor:
        tensor = self.board_to_tensor(board).unsqueeze(0).to(next(self.parameters()).device)
        with torch.no_grad():
            _, _, policy_logits = self(tensor)
        policy = torch.softmax(policy_logits, dim=1)
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
    
    @staticmethod
    def board_to_tensor(board: chess.Board) -> torch.Tensor:
        """Статический метод конвертации одной доски в 13-канальный тензор"""
        tensor = np.zeros((13, 8, 8), dtype=np.float32)
        pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

        # 0-5: Белые фигуры, 6-11: Черные фигуры
        for i, piece in enumerate(pieces):
            for sq in board.pieces(piece, chess.WHITE):
                tensor[i, chess.square_rank(sq), chess.square_file(sq)] = 1.0
            for sq in board.pieces(piece, chess.BLACK):
                tensor[i + 6, chess.square_rank(sq), chess.square_file(sq)] = 1.0

        # 12: Чей ход (1 = Белые, 0 = Черные)
        if board.turn == chess.WHITE:
            tensor[12, :, :] = 1.0

        return torch.tensor(tensor, dtype=torch.float32)


CLASS_NAMES = ["Blunder", "Mistake", "Inaccuracy", "Good", "Excellent", "Best"]

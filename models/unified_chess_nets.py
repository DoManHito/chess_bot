"""
Unified Chess Neural Network Architecture - OPTION A (Lookahead via Value)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np
from typing import List, Tuple

CLASS_NAMES = ["Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"]

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
    """Core convolutional network for chess board - 13 CHANNELS FOR OPTION A"""
    def __init__(self, in_channels=13, num_blocks=6, hidden_channels=128):
        super().__init__()
        self.conv_init = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(hidden_channels)
        self.blocks = nn.ModuleList([ChessResidualBlock(hidden_channels) for _ in range(num_blocks)])

    def forward(self, x):
        x = F.relu(self.bn_init(self.conv_init(x)))
        for block in self.blocks:
            x = block(x)
        return x


class UnifiedMoveClassifierNet(nn.Module):
    def __init__(self, core_net: ChessCoreNet, hidden_dim: int = 256):
        super().__init__()
        self.core = core_net
        
        # Reduced feature map before FC layers
        self.conv_reduce = nn.Conv2d(128, 32, kernel_size=1)
        self.bn_reduce = nn.BatchNorm2d(32)
        
        # Flatten size: 32 channels * 8 * 8 = 2048
        flattened_size = 32 * 8 * 8
        
        # Value Head
        self.fc_value1 = nn.Linear(flattened_size, hidden_dim)
        self.fc_value2 = nn.Linear(hidden_dim, 1)
        
        # Policy Head - outputs 4096 logits (64x64 from/to square space)
        self.fc_policy1 = nn.Linear(flattened_size, hidden_dim)
        self.policy_output = nn.Linear(hidden_dim, 4096)

    def forward(self, x):
        features = self.core(x)
        x_shared = F.relu(self.bn_reduce(self.conv_reduce(features)))
        flattened = x_shared.view(x_shared.size(0), -1)
        
        # Value branch
        xv = F.relu(self.fc_value1(flattened))
        value = torch.tanh(self.fc_value2(xv)).squeeze(-1)
        
        # Policy branch (Returns RAW LOGITS for correct cross-entropy loss)
        xp = F.relu(self.fc_policy1(flattened))
        policy_logits = self.policy_output(xp)
        
        # Returns None for classification head, as Option A computes it logically
        return None, value, policy_logits

    def get_policy(self, board: chess.Board, top_k: int = 64) -> dict:
        """Get move probability distribution for the given board.
        Returns dict {move_uci: probability} for all legal moves."""
        tensor = self.board_to_tensor(board).unsqueeze(0).to(next(self.parameters()).device)
        self.eval()
        with torch.no_grad():
            _, _, policy_logits = self(tensor)
            policy = torch.softmax(policy_logits[0], dim=0)  # shape: [4096]

        legal_moves = list(board.legal_moves)
        result = {}
        for move in legal_moves:
            idx = move.from_square * 64 + move.to_square
            result[move.uci()] = policy[idx].item()

        # Renormalize over legal moves only
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        return result

    @staticmethod
    def board_to_tensor(board: chess.Board) -> torch.Tensor:
        """Static method converting a single board state to a 13-channel tensor representation."""
        tensor = np.zeros((13, 8, 8), dtype=np.float32)
        pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

        # 0-5: White pieces, 6-11: Black pieces
        for i, piece in enumerate(pieces):
            for sq in board.pieces(piece, chess.WHITE):
                tensor[i, chess.square_rank(sq), chess.square_file(sq)] = 1.0
            for sq in board.pieces(piece, chess.BLACK):
                tensor[i + 6, chess.square_rank(sq), chess.square_file(sq)] = 1.0

        # Channel 12: Active turn (White=1, Black=0)
        if board.turn == chess.WHITE:
            tensor[12, :, :] = 1.0

        return torch.tensor(tensor, dtype=torch.float32)

class LookaheadMoveData:
    """Data structure for storing lookahead sequence information during self-play."""
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
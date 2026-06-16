import torch
import chess
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from models.unified_chess_nets import ChessCoreNet, UnifiedMoveClassifierNet, CLASS_NAMES
from .classification_config import CLASS_NAMES as CLASS_NAMES_CONFIG


@dataclass
class MoveData:
    """
    Data class representing a chess move with associated metadata.

    Used as input for batch move classification. Each move includes:
    - The board FEN before the move
    - The move in SAN notation
    - Optional evaluation value (from previous analysis)
    - Turn number and label for tracking game progress

    Attributes:
        board_fen: FEN string representing the board before the move
        move_san: Move in Standard Algebraic Notation
        evaluation: Evaluation value (default: 0.0)
        turn_num: Turn number (default: 1)
        turn_label: Whose turn it is ("White" or "Black", default: "White")
    """
    board_fen: str = ""
    move_san: str = ""
    evaluation: float = 0.0
    turn_num: int = 1
    turn_label: str = "White"


@dataclass
class MoveClassificationResult:
    """
    Result of classifying a single chess move.

    Contains the classification, confidence score, evaluation, and board states
    for the move being analyzed.

    Attributes:
        evaluation: Position evaluation after the move
        turn_num: Turn number
        turn_label: Whose turn it is
        classification: Move quality classification (Best, Excellent, Good, etc.)
        confidence: Neural network confidence in the classification
        move_san: The move in SAN notation
        fen_before: FEN before the move
        fen_after: FEN after the move

    Properties:
        move_number: Alias for turn_num
    """
    evaluation: float
    turn_num: int
    turn_label: str
    classification: str
    confidence: float
    move_san: str = ""
    fen_before: str = ""
    fen_after: str = ""

    @property
    def move_number(self) -> int:
        return self.turn_num

    def __str__(self) -> str:
        return f"Turn {self.turn_num} ({self.turn_label}): {self.classification} (eval={self.evaluation:.2f}, conf={self.confidence:.2f})"


class MoveClassifier:
    """
    Neural network-based chess move classifier.

    This class uses a trained neural network to classify chess moves into quality
    categories (Best, Excellent, Good, Inaccuracy, Mistake, Blunder) and predict
    the evaluation of the position after the move.

    Uses the unified model (UnifiedMoveClassifierNet) with policy head.

    Args:
        weights_path: Path to trained weights file
        device: Torch device to use ("cuda" or "cpu", None = auto-detect)
        policy_output_dim: Dimension of policy output for unified model (default: 64)
    """
    def __init__(self, weights_path: str = "models/weights_bot.pth", device: str = None,
                 policy_output_dim: int = 64) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Initialize the unified neural network
        core = ChessCoreNet(in_channels=25)
        self.model = UnifiedMoveClassifierNet(core_net=core, num_classes=len(CLASS_NAMES), policy_output_dim=policy_output_dim)

        try:
            # Load trained weights
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state, strict=False)
            self.model.to(self.device)
            self.model.eval()
            self.has_weights = True
        except FileNotFoundError:
            print(f"Warning: Weights {weights_path} not found. Classifier outputs random values.")
            self.has_weights = False

    def classify_move(self, board_fen: str, move_san: str, evaluation: float = 0.0, turn_num: int = 1, turn_label: str = "White"):
        """
        Classify a single chess move.

        Args:
            board_fen: FEN string representing the board before the move
            move_san: Move in Standard Algebraic Notation
            evaluation: Optional evaluation value (default: 0.0)
            turn_num: Turn number (default: 1)
            turn_label: Whose turn it is (default: "White")

        Returns:
            Tuple of (MoveClassificationResult, predicted_value):
            - MoveClassificationResult: Contains classification, confidence, and metadata
            - predicted_value: Neural network's value prediction for the position
        """
        board_before = chess.Board(board_fen)
        board_after = board_before.copy()
        try:
            move = board_after.parse_san(move_san)
            board_after.push(move)
        except Exception:
            return MoveClassificationResult(evaluation=evaluation, turn_num=turn_num, turn_label=turn_label,
                classification="Unknown", confidence=0.0, move_san=move_san, fen_before=board_fen, fen_after=board_after.fen()), 0.0

        # Convert board states to tensor
        tensor = self._board_to_tensor_static(board_before, board_after).to(self.device)
        # Forward pass (no gradient tracking for inference)
        with torch.no_grad():
            logits, value_tensor, _ = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)
            max_idx = torch.argmax(probabilities).item()
            confidence = probabilities[max_idx].item()
            predicted_value = value_tensor.item()
        return MoveClassificationResult(evaluation=evaluation, turn_num=turn_num, turn_label=turn_label,
            classification=CLASS_NAMES[max_idx], confidence=confidence, move_san=move_san,
            fen_before=board_fen, fen_after=board_after.fen()), predicted_value

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

        batch = torch.stack(tensors).to(self.device)
        # Forward pass (unified model)
        with torch.no_grad():
            logits, values, policy = self.model(batch)
            probs = torch.softmax(logits, dim=1)
            max_probs, indices = probs.max(dim=1)
            classes = [CLASS_NAMES[idx] for idx in indices.tolist()]
            confidences = max_probs.tolist()

            # Handle value predictions (can be scalar or list)
            if values.dim() == 0:
                value_list = [values.item()]
            else:
                value_list = values.flatten().tolist()
        return classes, confidences, value_list

    def classify_moves(self, moves: List[MoveData]) -> List[MoveClassificationResult]:
        """
        Classify a list of moves using MoveData objects.

        Args:
            moves: List of MoveData objects

        Returns:
            List of MoveClassificationResult objects
        """
        results = []
        for move in moves:
            res, _ = self.classify_move(move.board_fen, move.move_san, move.evaluation, move.turn_num, move.turn_label)
            results.append(res)
        return results

    def get_policy(self, board: chess.Board, top_k: int = 64) -> dict:
        """
        Get policy probabilities for top K moves (unified model only).

        Args:
            board: Chess board position
            top_k: Number of top moves to return (default: 64)

        Returns:
            Dictionary mapping move UCI to probability
        """
        if not self.use_unified_model:
            raise NotImplementedError("Policy head not available in legacy model. Use use_unified_model=True.")

        # Create tensor from current board only (for policy head)
        tensor = self._board_to_tensor_for_policy(board).to(self.device)
        with torch.no_grad():
            _, _, policy = self.model(tensor)
        policy = policy[0, :top_k].cpu().numpy()

        # Get legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}

        # Convert to dictionary
        policy_dict = {}
        for i, move in enumerate(legal_moves[:top_k]):
            policy_dict[move.uci()] = policy[i]

        # Normalize
        total = sum(policy_dict.values())
        if total > 0:
            policy_dict = {k: v / total for k, v in policy_dict.items()}

        return policy_dict

    @staticmethod
    def _board_to_tensor_static(board_before: chess.Board, board_after: chess.Board) -> torch.Tensor:
        """
        Convert chess board states to a 25-channel tensor representation.

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

        return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)

    @staticmethod
    def _board_to_tensor_for_policy(board: chess.Board) -> torch.Tensor:
        """
        Convert chess board state to a 25-channel tensor for policy head.
        Uses current board for all piece channels (simplified representation).

        Args:
            board: Current chess board state

        Returns:
            torch.Tensor of shape (1, 25, 8, 8) with board representation
        """
        tensor = np.zeros((25, 8, 8), dtype=np.float32)

        pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

        # Encode white pieces (channels 0-5)
        for i, piece in enumerate(pieces):
            for sq in board.pieces(piece, chess.WHITE):
                tensor[i, chess.square_rank(sq), chess.square_file(sq)] = 1.0
            # Encode black pieces (channels 6-11)
            for sq in board.pieces(piece, chess.BLACK):
                tensor[i + 6, chess.square_rank(sq), chess.square_file(sq)] = 1.0

        # Encode white pieces (channels 12-17)
        for i, piece in enumerate(pieces):
            for sq in board.pieces(piece, chess.WHITE):
                tensor[i + 12, chess.square_rank(sq), chess.square_file(sq)] = 1.0
            # Encode black pieces (channels 18-23)
            for sq in board.pieces(piece, chess.BLACK):
                tensor[i + 18, chess.square_rank(sq), chess.square_file(sq)] = 1.0

        # Layer 24: Whose turn (White=1, Black=0)
        if board.turn == chess.WHITE:
            tensor[24, :, :] = 1.0

        return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)

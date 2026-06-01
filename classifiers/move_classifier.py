import torch
import chess
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from models.chess_nets import ChessCoreNet, MoveClassifierNet
from .classification_config import CLASS_NAMES


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
        """
        Get the move number as a property.

        Returns:
            The turn number
        """
        return self.turn_num

    def __str__(self) -> str:
        """
        Return a string representation of the classification result.

        Returns:
            Formatted string: "Turn N (TurnLabel): Classification (eval=X.XX, conf=X.XX)"
        """
        return f"Turn {self.turn_num} ({self.turn_label}): {self.classification} (eval={self.evaluation:.2f}, conf={self.confidence:.2f})"


class MoveClassifier:
    """
    Neural network-based chess move classifier.

    This class uses a trained neural network to classify chess moves into quality
    categories (Best, Excellent, Good, Inaccuracy, Mistake, Blunder) and predict
    the evaluation of the position after the move.

    The classifier:
    1. Converts chess board states to 25-channel tensors
    2. Uses a convolutional neural network to extract features
    3. Produces classification logits and value predictions
    4. Supports both single-move and batch move classification

    Args:
        weights_path: Path to trained weights file (default: "models/weights_classifier.pth")
        device: Torch device to use ("cuda" or "cpu", None = auto-detect)

    Attributes:
        device: Torch device being used
        model: The neural network model
        has_weights: Whether trained weights were successfully loaded
    """
    def __init__(self, weights_path: str = "models/weights_classifier.pth", device: str = None) -> None:
        """
        Initialize the move classifier.

        Args:
            weights_path: Path to trained weights file
            device: Torch device ("cuda" or "cpu", None = auto-detect)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Initialize the neural network
        core = ChessCoreNet(in_channels=25)
        self.model = MoveClassifierNet(core_net=core, num_classes=len(CLASS_NAMES))

        try:
            # Load trained weights
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state, strict=False)
            self.model.to(self.device)
            self.model.eval()
            self.has_weights = True
            print(f"MoveClassifier loaded on {self.device}")
        except FileNotFoundError:
            print(f"Warning: Weights {weights_path} not found. Classifier outputs random values.")
            self.has_weights = False

    @staticmethod
    def _board_to_tensor_static(board_before: chess.Board, board_after: chess.Board) -> torch.Tensor:
        """
        Convert chess board states to a 25-channel tensor representation.

        This static method creates a tensor that encodes both the board state before
        and after a move, allowing the neural network to learn the effect of moves.

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
        piece_to_layer = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        # Encode white pieces from board_before (channels 0-5)
        for square in chess.SQUARES:
            row = 7 - (square // 8)
            col = square % 8
            piece_before = board_before.piece_at(square)
            if piece_before:
                layer = piece_to_layer[piece_before.piece_type]
                if piece_before.color == chess.WHITE:
                    tensor[layer, row, col] = 1.0
                else:
                    tensor[layer + 6, row, col] = 1.0
            # Encode white pieces from board_after (channels 12-17)
            piece_after = board_after.piece_at(square)
            if piece_after:
                layer = piece_to_layer[piece_after.piece_type]
                if piece_after.color == chess.WHITE:
                    tensor[layer + 12, row, col] = 1.0
                else:
                    tensor[layer + 18, row, col] = 1.0
        # Layer 24: Whose turn BEFORE (White=1, Black=0)
        if board_before.turn == chess.WHITE:
            tensor[24, :, :] = 1.0
        return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)

    def classify_move(self, board_fen: str, move_san: str, evaluation: float = 0.0, turn_num: int = 1, turn_label: str = "White"):
        """
        Classify a single chess move.

        This method:
        1. Parses the move from SAN notation
        2. Creates board states before and after the move
        3. Converts to tensor representation
        4. Runs the neural network forward pass
        5. Returns classification and confidence

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
            # Invalid move, return Unknown classification
            return MoveClassificationResult(evaluation=evaluation, turn_num=turn_num, turn_label=turn_label,
                classification="Unknown", confidence=0.0, move_san=move_san, fen_before=board_fen, fen_after=board_after.fen()), 0.0

        # Convert board states to tensor
        tensor = self._board_to_tensor_static(board_before, board_after).to(self.device)
        # Forward pass (no gradient tracking for inference)
        with torch.no_grad():
            logits, value_tensor = self.model(tensor)
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

        This method is more efficient than calling classify_move() multiple times
        because it processes all moves through the neural network in one forward pass.

        Args:
            board_before: Chess board state before the moves
            moves_san: List of moves in Standard Algebraic Notation

        Returns:
            Tuple of (classes, confidences, values):
            - classes: List of classification strings
            - confidences: List of confidence scores
            - values: List of value predictions
        """
        tensors = []
        for san in moves_san:
            board_after = board_before.copy()
            try:
                move = board_after.parse_san(san)
                board_after.push(move)
                tensor = self._board_to_tensor_static(board_before, board_after).squeeze(0)
            except Exception:
                # Invalid move, use zero tensor
                tensor = torch.zeros(25, 8, 8)
            tensors.append(tensor)

        if not tensors:
            return [], [], []

        # Stack tensors into batch
        batch = torch.stack(tensors).to(self.device)
        # Forward pass
        with torch.no_grad():
            logits, values = self.model(batch)
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

        Convenience method that wraps classify_move() for use with MoveData objects.

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
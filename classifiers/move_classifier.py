import torch
import chess
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from models.chess_nets import ChessCoreNet, MoveClassifierNet
from .classification_config import CLASS_NAMES

@dataclass
class MoveData:
    board_fen: str = ""
    move_san: str = ""
    evaluation: float = 0.0
    turn_num: int = 1
    turn_label: str = "White"

@dataclass
class MoveClassificationResult:
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
    def __init__(self, weights_path: str = "models/weights_classifier.pth", device: str = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        core = ChessCoreNet(in_channels=25)
        self.model = MoveClassifierNet(core_net=core, num_classes=len(CLASS_NAMES))
        
        try:
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
        tensor = np.zeros((25, 8, 8), dtype=np.float32)
        piece_to_layer = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
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
            piece_after = board_after.piece_at(square)
            if piece_after:
                layer = piece_to_layer[piece_after.piece_type]
                if piece_after.color == chess.WHITE:
                    tensor[layer + 12, row, col] = 1.0
                else:
                    tensor[layer + 18, row, col] = 1.0
        if board_before.turn == chess.WHITE:
            tensor[24, :, :] = 1.0
        return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)

    def classify_move(self, board_fen: str, move_san: str, evaluation: float = 0.0, turn_num: int = 1, turn_label: str = "White"):
        board_before = chess.Board(board_fen)
        board_after = board_before.copy()
        try:
            move = board_after.parse_san(move_san)
            board_after.push(move)
        except Exception:
            return MoveClassificationResult(evaluation=evaluation, turn_num=turn_num, turn_label=turn_label,
                classification="Unknown", confidence=0.0, move_san=move_san, fen_before=board_fen, fen_after=board_after.fen()), 0.0

        tensor = self._board_to_tensor_static(board_before, board_after).to(self.device)
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
        tensors = []
        for san in moves_san:
            board_after = board_before.copy()
            try:
                move = board_after.parse_san(san)
                board_after.push(move)
                tensor = self._board_to_tensor_static(board_before, board_after).squeeze(0)
            except Exception:
                tensor = torch.zeros(25, 8, 8)
            tensors.append(tensor)
            
        if not tensors:
            return [], [], []
            
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            logits, values = self.model(batch)
            probs = torch.softmax(logits, dim=1)
            max_probs, indices = probs.max(dim=1)
            classes = [CLASS_NAMES[idx] for idx in indices.tolist()]
            confidences = max_probs.tolist()
            
            if values.dim() == 0:
                value_list = [values.item()]
            else:
                value_list = values.flatten().tolist()
        return classes, confidences, value_list

    def classify_moves(self, moves: List[MoveData]) -> List[MoveClassificationResult]:
        results = []
        for move in moves:
            res, _ = self.classify_move(move.board_fen, move.move_san, move.evaluation, move.turn_num, move.turn_label)
            results.append(res)
        return results
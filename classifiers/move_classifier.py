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
    evaluation: float
    turn_num: int
    turn_label: str
    classification: str
    confidence: float
    move_san: str = ""
    fen_before: str = ""
    fen_after: str = ""

    def __str__(self) -> str:
        return f"Turn {self.turn_num} ({self.turn_label}): {self.classification} (eval={self.evaluation:.2f}, conf={self.confidence:.2f})"

class MoveClassifier:
    def __init__(self, weights_path: str = "models/weights_bot.pth", device: str = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Инициализируем сеть ядра на 13 каналов
        core = ChessCoreNet(in_channels=13)
        self.model = UnifiedMoveClassifierNet(core_net=core)

        try:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state, strict=False)
            self.model.to(self.device)
            self.model.eval()
            self.has_weights = True
        except FileNotFoundError:
            print(f"Warning: Weights {weights_path} not found. Running on random initialization.")
            self.has_weights = False

    def encode_board(self, board: chess.Board) -> torch.Tensor:
        """Использует правильный 13-канальный конвертер из класса сети"""
        return UnifiedMoveClassifierNet.board_to_tensor(board)
    
    def classify_move(self, fen: str, move_san: str, evaluation: float = 0.0, turn_num: int = 1, turn_label: str = "White"):
        board = chess.Board(fen)
        try:
            move = board.parse_san(move_san)
        except ValueError:
            move = chess.Move.from_uci(move_san)

        who_moves = board.turn 

        # --- СТЕП 1: Оценка ДО хода ---
        tensor_before = self.encode_board(board).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            _, value_out, policy_out = self.model(tensor_before)
            v_before = value_out.item()

        # --- СТЕП 2: Ход на доске ---
        fen_before = board.fen()
        board.push(move)
        fen_after = board.fen()

        # --- СТЕП 3: Оценка ПОСЛЕ хода ---
        tensor_after = self.encode_board(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value_out_after, _ = self.model(tensor_after)
            v_after = value_out_after.item()

        # --- СТЕП 4: Расчет потери качества позиции ---
        if who_moves == chess.WHITE:
            loss = v_before - v_after
        else:
            loss = v_after - v_before

        # Поправка: если игра завершилась матом в пользу ходившего, это всегда Best ход
        if board.is_checkmate():
            loss = -1.0 

        # --- СТЕП 5: Маппинг в классы (Пороги для диапазона ценности [-1, 1]) ---
        if loss <= 0.02:
            classification = "Best"
        elif loss <= 0.07:
            classification = "Excellent"
        elif loss <= 0.15:
            classification = "Good"
        elif loss <= 0.30:
            classification = "Inaccuracy"
        elif loss <= 0.55:
            classification = "Mistake"
        else:
            classification = "Blunder"

        # Расчет уверенности на основе близости хода к предсказанию Policy Head
        move_idx = move.from_square * 64 + move.to_square
        policy_prob = policy_out[0, move_idx].item() if move_idx < 4096 else 0.0
        confidence = float(min(100.0, (policy_prob * 50) + (np.exp(-max(0, loss)) * 50)))

        return MoveClassificationResult(
            evaluation=v_after,
            turn_num=turn_num,
            turn_label=turn_label,
            classification=classification,
            confidence=confidence,
            move_san=move_san,
            fen_before=fen_before,
            fen_after=fen_after
        ), fen_after
    
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

    def get_policy(self, board: chess.Board) -> dict:
        """Исправленный стабильный метод получения вероятностей ходов"""
        tensor = self.encode_board(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, policy = self.model(tensor)
        policy = policy[0].cpu().numpy()

        policy_dict = {}
        for move in board.legal_moves:
            # Уникальный стабильный индекс для каждого хода в пространстве 64х64
            move_idx = move.from_square * 64 + move.to_square
            policy_dict[move.uci()] = float(policy[move_idx])

        # Нормализация
        total = sum(policy_dict.values())
        if total > 0:
            policy_dict = {k: v / total for k, v in policy_dict.items()}
        return policy_dict

    @staticmethod
    def _board_to_tensor_static(board_before: chess.Board, board_after: chess.Board) -> torch.Tensor:
        """
        Convert chess board states to a 25-channel tensor.
        Channels 0-11: board_before, Channels 12-23: board_after, Channel 24: active turn
        """
        tensor = np.zeros((25, 8, 8), dtype=np.float32)
        pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

        # 1. Кодируем board_before (каналы 0-11)
        for i, piece in enumerate(pieces):
            for sq in board_before.pieces(piece, chess.WHITE):
                tensor[i, chess.square_rank(sq), chess.square_file(sq)] = 1.0
            for sq in board_before.pieces(piece, chess.BLACK):
                tensor[i + 6, chess.square_rank(sq), chess.square_file(sq)] = 1.0

        # 2. Кодируем board_after (каналы 12-23) — ИСПРАВЛЕНО ТУТ
        for i, piece in enumerate(pieces):
            for sq in board_after.pieces(piece, chess.WHITE):
                tensor[i + 12, chess.square_rank(sq), chess.square_file(sq)] = 1.0
            for sq in board_after.pieces(piece, chess.BLACK):
                tensor[i + 18, chess.square_rank(sq), chess.square_file(sq)] = 1.0

        # 3. Кодируем ход (канал 24)
        if board_before.turn == chess.WHITE:
            tensor[24, :, :] = 1.0  # 1.0 если ходят белые, 0.0 если черные

        return torch.tensor(tensor, dtype=torch.float32)

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

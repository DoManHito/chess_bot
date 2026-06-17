import torch
import chess
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from models.unified_chess_nets import ChessCoreNet, UnifiedMoveClassifierNet, CLASS_NAMES

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

    def __str__(self) -> str:
        return f"Turn {self.turn_num} ({self.turn_label}): {self.classification} (eval={self.evaluation:.2f}, conf={self.confidence:.2f})"

class MoveClassifier:
    def __init__(self, weights_path: str = "models/weights_bot.pth", device: str = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        core = ChessCoreNet(in_channels=13)
        self.model = UnifiedMoveClassifierNet(core_net=core)

        try:
            state = torch.load(weights_path, map_location=self.device)
            filtered_state = {}
            for key, value in state.items():
                if key in self.model.state_dict() and value.shape == self.model.state_dict()[key].shape:
                    filtered_state[key] = value
                else:
                    print(f"⚠️ Skipping incompatible weight: {key} (shape: {value.shape})")
            
            if filtered_state:
                self.model.load_state_dict(filtered_state, strict=False)
                self.model.to(self.device)
                self.model.eval()
                self.has_weights = True
            else:
                raise RuntimeError("No compatible weights found")
        except FileNotFoundError:
            print(f"Warning: Weights {weights_path} not found. Running on random initialization.")
            self.has_weights = False
        except RuntimeError as e:
            if "No compatible weights found" in str(e):
                print(f"Warning: No compatible weights found. Running on random initialization.")
                self.has_weights = False
            else:
                raise

    def encode_board(self, board: chess.Board) -> torch.Tensor:
        """Uses the correct 13-channel encoder."""
        return UnifiedMoveClassifierNet.board_to_tensor(board)
    
    def classify_move(self, fen: str, move_san: str, evaluation: float = 0.0, turn_num: int = 1, turn_label: str = "White") -> MoveClassificationResult:
        board = chess.Board(fen)
        try:
            move = board.parse_san(move_san)
        except ValueError:
            move = chess.Move.from_uci(move_san)

        who_moves = board.turn

        # Step 1: Eval BEFORE move
        tensor_before = self.encode_board(board).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            _, value_out, policy_logits = self.model(tensor_before)
            v_before = value_out.item()
            policy_out = torch.softmax(policy_logits, dim=1)

        # Step 2: Make the move
        fen_before = board.fen()
        board.push(move)
        fen_after = board.fen()

        # Step 3: Eval AFTER move
        tensor_after = self.encode_board(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value_after_out, _ = self.model(tensor_after)
            v_after = value_after_out.item()

        loss = v_before + v_after

        if board.is_checkmate():
            loss = -1.0  # Checkmate is always best

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

        # Confidence via Policy Head - use from_square * 64 + to_square as index into 4096-dim output
        move_idx = move.from_square * 64 + move.to_square
        policy_prob = policy_out[0, move_idx].item()
        confidence = float(policy_prob)

        return MoveClassificationResult(
            evaluation=v_after * 5.0,
            turn_num=turn_num,
            turn_label=turn_label,
            classification=classification,
            confidence=confidence,
            move_san=move_san,
            fen_before=fen_before,
            fen_after=fen_after
        )

    def classify_moves_batch(self, batch_moves: List[MoveData]) -> List[MoveClassificationResult]:
        """Processes a batch of moves efficiently using 13 channels. Evaluates position before and after for each move in parallel."""
        if not batch_moves:
            return []

        results = []
        boards_before = []
        boards_after = []
        moves_parsed = []
        legal_moves_list = []

        # Prepare game boards and get legal moves for each
        for item in batch_moves:
            board = chess.Board(item.board_fen)
            boards_before.append(board.copy())
            try:
                m = board.parse_san(item.move_san)
            except ValueError:
                m = chess.Move.from_uci(item.move_san)
            moves_parsed.append(m)
            
            # Get legal moves for policy indexing
            legal_moves_list.append(list(board.legal_moves))
            
            board.push(m)
            boards_after.append(board)

        # Build tensors
        tensors_b = torch.stack([self.encode_board(b) for b in boards_before]).to(self.device)
        tensors_a = torch.stack([self.encode_board(b) for b in boards_after]).to(self.device)

        self.model.eval()
        with torch.no_grad():
            _, values_before, policy_logits = self.model(tensors_b)
            _, values_after, _ = self.model(tensors_a)
            policies = torch.softmax(policy_logits, dim=1)

        # Parse output for each batch item
        for idx, item in enumerate(batch_moves):
            b_before = boards_before[idx]
            b_after = boards_after[idx]
            m = moves_parsed[idx]
            legal_moves = legal_moves_list[idx]
            
            v_b = values_before[idx].item()
            v_a = values_after[idx].item()
            
            loss = v_b + v_a

            if b_after.is_checkmate():
                loss = -1.0

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

            # Use from_square * 64 + to_square as index into 4096-dim policy output
            move_idx = m.from_square * 64 + m.to_square
            prob = policies[idx, move_idx].item()
            conf = float(min(100.0, prob * 100.0))

            results.append(MoveClassificationResult(
                evaluation=v_a * 5.0,
                turn_num=item.turn_num,
                turn_label=item.turn_label,
                classification=classification,
                confidence=conf,
                move_san=item.move_san,
                fen_before=b_before.fen(),
                fen_after=b_after.fen()
            ))

        return results
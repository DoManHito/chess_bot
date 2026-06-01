import sqlite3
import json
import torch
from torch.utils.data import Dataset
import chess
import numpy as np
from classifiers.classification_config import CLASS_NAMES

class ChessSelfPlayDataset(Dataset):
    def __init__(self, db_path="chess_bot.db"):
        self.db_path = db_path
        self.samples = []
        
        torch.set_num_threads(1)
        self._load_data()

    def _load_data(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='self_play_moves'")
        if not cursor.fetchone():
            print("Table self_play_moves not found!")
            conn.close()
            return

        print("Loading Self-Play experience from database...")
        cursor.execute("SELECT fen_before, move_uci, mcts_policy, result_value FROM self_play_moves")
        rows = cursor.fetchall()
        conn.close()

        for fen_before, move_uci, mcts_policy_json, result_value in rows:
            self.samples.append({
                "fen": fen_before,
                "move_uci": move_uci,
                "policy_dict": json.loads(mcts_policy_json),
                "value": float(result_value)
            })
        print(f"Successfully loaded {len(self.samples)} states for training.")

    def __len__(self):
        return len(self.samples)

    # Static method for converting board to tensor (extracted from MoveClassifier for autonomy)
    def _board_to_tensor_static(self, board_before: chess.Board, board_after: chess.Board) -> torch.Tensor:
        tensor = np.zeros((25, 8, 8), dtype=np.float32)
        
        pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        for i, piece in enumerate(pieces):
            for sq in board_before.pieces(piece, chess.WHITE):
                tensor[i, chess.square_rank(sq), chess.square_file(sq)] = 1.0
            for sq in board_before.pieces(piece, chess.BLACK):
                tensor[i + 6, chess.square_rank(sq), chess.square_file(sq)] = 1.0
                
        for i, piece in enumerate(pieces):
            for sq in board_after.pieces(piece, chess.WHITE):
                tensor[i + 12, chess.square_rank(sq), chess.square_file(sq)] = 1.0
            for sq in board_after.pieces(piece, chess.BLACK):
                tensor[i + 18, chess.square_rank(sq), chess.square_file(sq)] = 1.0
                
        # Layer 24: Whose turn BEFORE (White=1, Black=0)
        if board_before.turn == chess.WHITE:
            tensor[24, :, :] = 1.0
            
        return torch.from_numpy(tensor)

    def __getitem__(self, idx):
        torch.set_num_threads(1)
        
        sample = self.samples[idx]
        board = chess.Board(sample["fen"])
        value_target = torch.tensor(sample["value"], dtype=torch.float32)
        
        board_after = board.copy()
        try:
            move = chess.Move.from_uci(sample["move_uci"])
            if move in board_after.legal_moves:
                board_after.push(move)
        except Exception:
            pass
            
        input_tensor = self._board_to_tensor_static(board, board_after)
    
        policy_dict = sample["policy_dict"]
        class_target_distribution = np.zeros(len(CLASS_NAMES), dtype=np.float32)

        for i, class_name in enumerate(CLASS_NAMES):
            class_target_distribution[i] = policy_dict.get(class_name, 0.0)

        sum_dist = class_target_distribution.sum()
        if sum_dist <= 0:
            class_target_distribution = np.ones(len(CLASS_NAMES), dtype=np.float32) / len(CLASS_NAMES)

        return input_tensor, torch.tensor(class_target_distribution, dtype=torch.float32), value_target
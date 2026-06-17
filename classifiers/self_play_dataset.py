import sqlite3
import json
import torch
from torch.utils.data import Dataset
import chess
import numpy as np
from classifiers.classification_config import CLASS_NAMES
from models.unified_chess_nets import UnifiedMoveClassifierNet


class ChessSelfPlayDataset(Dataset):
    def __init__(self, db_path="chess_bot.db", device='cpu'):
        self.db_path = db_path
        self.device = torch.device(device)
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
        
        cursor.execute("""
            SELECT fen_before, move_uci, mcts_policy, result_value, 
                   lookahead_depth, future_moves, final_classification, move_sequence_classes
            FROM self_play_moves
        """)
        rows = cursor.fetchall()
        conn.close()

        for fen_before, move_uci, mcts_policy_json, result_value, lookahead_depth, future_moves, final_classification, move_sequence_classes in rows:
            sample = {
                "fen": fen_before,
                "move_uci": move_uci,
                "policy_dict": json.loads(mcts_policy_json) if mcts_policy_json else {},
                "value": float(result_value) if result_value else 0.0
            }
            
            if lookahead_depth and future_moves:
                sample["lookahead_depth"] = int(lookahead_depth)
                sample["future_moves"] = json.loads(future_moves) if future_moves else []
                sample["final_classification"] = final_classification if final_classification else "Good"
                sample["move_sequence_classes"] = json.loads(move_sequence_classes) if move_sequence_classes else [CLASS_NAMES[3]]  # Default to "Good"
            
            self.samples.append(sample)
        
        print(f"Successfully loaded {len(self.samples)} states for training.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        torch.set_num_threads(1)

        sample = self.samples[idx]
        board = chess.Board(sample["fen"])
        value_target = torch.tensor(sample["value"], dtype=torch.float32)

        # Convert board state to 13-channel tensor (matches inference architecture)
        input_tensor = UnifiedMoveClassifierNet.board_to_tensor(board)

        # Extract MCTS policy distribution
        policy_dict = sample["policy_dict"]
        class_target_distribution = np.zeros(len(CLASS_NAMES), dtype=np.float32)

        for i, class_name in enumerate(CLASS_NAMES):
            class_target_distribution[i] = policy_dict.get(class_name, 0.0)

        # Normalize distribution if sum is too small (fallback to uniform)
        sum_dist = class_target_distribution.sum()
        if sum_dist <= 0:
            class_target_distribution = np.ones(len(CLASS_NAMES), dtype=np.float32) / len(CLASS_NAMES)

        # Create uniform policy target distribution (64 actions)
        policy_target = np.ones(64, dtype=np.float32) / 64.0
        policy_target_tensor = torch.from_numpy(policy_target)

        return input_tensor, torch.from_numpy(class_target_distribution), value_target, policy_target_tensor


class LookaheadMoveSequence:
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
    def __init__(self, db_path="chess_bot.db", device='cpu'):
        self.db_path = db_path
        self.device = torch.device(device)
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

    def __getitem__(self, idx):
        torch.set_num_threads(1)

        sample = self.samples[idx]
        board = chess.Board(sample["fen"])
        value_target = torch.tensor(sample["value"], dtype=torch.float32)

        # Convert board state to 13-channel tensor (matches inference architecture)
        input_tensor = UnifiedMoveClassifierNet.board_to_tensor(board)

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
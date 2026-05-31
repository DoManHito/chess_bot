import sqlite3
import json
import os
import torch
from torch.utils.data import Dataset
import chess
import numpy as np

from classifiers.move_classifier import MoveClassifier
from classifiers.classification_config import CLASS_NAMES

class ChessSelfPlayDataset(Dataset):
    def __init__(self, db_path="chess_bot.db", device=None):
        self.db_path = db_path
        self.samples = []
        
        self.weights_path = "models/weights_classifier.pth"
        if not os.path.exists(self.weights_path):
            print(f"⚠️ Предупреждение: Файл {self.weights_path} не найден!")
            
        # Ограничиваем внутренние потоки PyTorch, чтобы DataLoader не зависал
        torch.set_num_threads(1)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if self.device == "cpu":
            torch.set_num_threads(1)
        
        self.classifier_utils = MoveClassifier(weights_path=self.weights_path, device=device)
        self.classifier_utils.model.eval()  # Гарантируем режим валидации
        self._load_data()

    def _load_data(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='self_play_moves'")
        if not cursor.fetchone():
            print("Таблица self_play_moves не найдена!")
            conn.close()
            return

        print("Загрузка опыта Self-Play из базы данных...")
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
        print(f"Успешно загружено {len(self.samples)} состояний для обучения.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Защита внутренних воркеров DataLoader от взаимной блокировки потоков CPU
        if self.device == "cpu":
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
            
        input_tensor = self.classifier_utils._board_to_tensor_static(board, board_after).squeeze(0)
    
        policy_dict = sample["policy_dict"]
        class_target_distribution = np.zeros(len(CLASS_NAMES), dtype=np.float32)

        if not policy_dict:
            class_target_distribution = np.ones(len(CLASS_NAMES), dtype=np.float32) / len(CLASS_NAMES)
            return input_tensor, torch.tensor(class_target_distribution, dtype=torch.float32), value_target
        
        moves_uci_list = list(policy_dict.keys())
        mcts_probs = list(policy_dict.values())

        moves_san = []
        valid_indices = []

        for i, mu in enumerate(moves_uci_list):
            try:
                m_obj = chess.Move.from_uci(mu)
                moves_san.append(board.san(m_obj))
                valid_indices.append(i)
            except Exception:
                class_target_distribution[0] += mcts_probs[i]  # Дефолт в Best

        if moves_san:
            classes, _, _ = self.classifier_utils.classify_moves_batch(board, moves_san)
            for cls, i in zip(classes, valid_indices):
                c_idx = CLASS_NAMES.index(cls) if cls in CLASS_NAMES else 0
                class_target_distribution[c_idx] += mcts_probs[i]
                
        sum_dist = class_target_distribution.sum()
        if sum_dist > 0:
            class_target_distribution /= sum_dist
        else:
            class_target_distribution = np.ones(len(CLASS_NAMES), dtype=np.float32) / len(CLASS_NAMES)

        return input_tensor, torch.tensor(class_target_distribution, dtype=torch.float32), value_target
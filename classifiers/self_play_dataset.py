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
    def __init__(self, db_path="chess_bot.db"):
        self.db_path = db_path
        self.samples = []
        
        self.weights_path = "models/weights_classifier.pth"
        if not os.path.exists(self.weights_path):
            print(f"⚠️ Предупреждение: Файл {self.weights_path} не найден!")
            
        self.classifier_utils = MoveClassifier(weights_path=self.weights_path) 
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
        sample = self.samples[idx]
        board = chess.Board(sample["fen"])
        
        board_after = board.copy()
        try:
            move = chess.Move.from_uci(sample["move_uci"])
            if move in board_after.legal_moves:
                board_after.push(move)
        except Exception:
            pass
            
        input_tensor = self.classifier_utils._board_to_tensor_static(board, board_after).squeeze(0)
    
        # Превращаем MCTS политику ходов в распределение по КЛАССАМ качества
        policy_dict = sample["policy_dict"]

        if not policy_dict:
            class_target_distribution = np.ones(len(CLASS_NAMES)) / len(CLASS_NAMES)
            print("policy_dict is empty")
            return input_tensor, torch.tensor(class_target_distribution, dtype=torch.float32), value_target
        
        class_target_distribution = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        
        for move_uci, mcts_prob in policy_dict.items():
            try:
                m_obj = chess.Move.from_uci(move_uci)
                m_san = board.san(m_obj)
                res, _ = self.classifier_utils.classify_move(sample["fen"], m_san)
                c_idx = CLASS_NAMES.index(res.classification)
                class_target_distribution[c_idx] += mcts_prob
            except Exception:
                class_target_distribution[0] += mcts_prob # Дефолт в Best
                
        # Нормализуем распределение (smooth)
        sum_dist = class_target_distribution.sum()
        if sum_dist > 0:
            class_target_distribution /= sum_dist
        else:
            class_target_distribution = np.ones(len(CLASS_NAMES)) / len(CLASS_NAMES)

        value_target = torch.tensor(sample["value"], dtype=torch.float32)
        return input_tensor, torch.tensor(class_target_distribution, dtype=torch.float32), value_target
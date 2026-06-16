"""
Dataset for loading policy data from Stockfish evaluations.
"""

from torch.utils.data import Dataset
import json
import torch
import numpy as np


class StockfishPolicyDataset(Dataset):
    """
    Dataset for training policy head using Stockfish evaluations.
    
    Loads pre-extracted policy data from JSON file.
    """
    
    def __init__(self, json_path="stockfish_policy_data.json", device='cpu'):
        """
        Initialize dataset.
        
        Args:
            json_path: Path to JSON file with policy data
            device: Device to use for tensors
        """
        self.device = torch.device(device)
        self.data = self._load_data(json_path)
        print(f"Loaded {len(self.data)} samples from {json_path}")
    
    def _load_data(self, json_path):
        """Load policy data from JSON file."""
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get item at index."""
        item = self.data[idx]
        
        # Convert FEN to tensor (simplified - should use existing board encoding)
        fen = item['fen_before']
        
        # For now, return basic data - need to implement FEN to tensor conversion
        # This will be handled by the main dataset class
        return {
            'fen': fen,
            'policy_score': item['policy_score'],
            'move_san': item['move_san'],
            'evaluation': item.get('evaluation'),
            'classification': item.get('classification')
        }

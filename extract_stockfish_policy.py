"""
Extract policy data from Stockfish evaluations in existing games.

Stockfish evaluations can be used to create policy targets:
- Positive evaluation (+) -> favor moves that lead to better positions
- Negative evaluation (-) -> favor moves that lead to worse positions
- Absolute evaluation magnitude -> indicates move quality
"""

import sqlite3
import json
import numpy as np
from collections import defaultdict

def extract_policy_from_evaluations(db_path="chess_bot.db", limit=None):
    """
    Extract policy data from Stockfish evaluations.
    
    Policy is derived from absolute evaluation values:
    - Higher absolute evaluation -> more decisive position -> higher policy score
    - Lower absolute evaluation -> more balanced position -> lower policy score
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all moves with evaluations
    query = """
        SELECT 
            m.id,
            m.game_id,
            m.move_number,
            m.fen_before,
            m.fen_after,
            m.move_san,
            m.classification,
            m.evaluation,
            g.result
        FROM moves m
        JOIN games g ON m.game_id = g.id
        WHERE m.evaluation IS NOT NULL
        ORDER BY m.game_id, m.move_number
    """
    
    if limit:
        query = f"SELECT * FROM ({query}) LIMIT {limit}"
    
    cursor.execute(query)
    moves = cursor.fetchall()
    
    print(f"Loaded {len(moves)} moves with evaluations")
    
    # Group moves by game and position
    games = defaultdict(list)
    for move in moves:
        move_id, game_id, move_num, fen_before, fen_after, move_san, classification, eval_val, result = move
        games[game_id].append({
            'move_id': move_id,
            'move_number': move_num,
            'fen_before': fen_before,
            'fen_after': fen_after,
            'move_san': move_san,
            'classification': classification,
            'evaluation': eval_val,
            'result': result
        })
    
    print(f"Found {len(games)} unique game positions")
    
    # Create policy targets from evaluation magnitudes
    policy_data = []
    
    for game_id, game_moves in games.items():
        # Sort by move number
        game_moves.sort(key=lambda x: x['move_number'])
        
        # For each move, create policy based on evaluation magnitude
        for i, move in enumerate(game_moves):
            # Policy: favor moves that lead to more decisive positions
            eval_val = move['evaluation']
            
            # Use sigmoid of absolute evaluation to create policy score
            # This gives higher scores for more decisive positions
            abs_eval = abs(eval_val)
            
            # Normalize to reasonable range (Stockfish evals can be large)
            # Use sigmoid: 1 / (1 + exp(-x))
            policy_score = 1 / (1 + np.exp(-abs_eval / 10))
            
            # Clamp to [0.01, 0.99] to avoid extreme values
            policy_score = max(0.01, min(0.99, policy_score))
            
            policy_data.append({
                'game_id': game_id,
                'move_number': move['move_number'],
                'fen_before': move['fen_before'],
                'move_san': move['move_san'],
                'policy_score': policy_score,
                'classification': move['classification'],
                'evaluation': eval_val
            })
    
    # Convert to numpy array for training
    policy_scores = np.array([d['policy_score'] for d in policy_data], dtype=np.float32)
    policy_scores = policy_scores / policy_scores.sum()  # Normalize to sum to 1
    
    print(f"Created policy data for {len(policy_data)} moves")
    print(f"Policy scores range: [{policy_scores.min():.6f}, {policy_scores.max():.6f}]")
    print(f"Policy scores sum: {policy_scores.sum():.4f}")
    
    # Save to JSON for later use
    output_file = "stockfish_policy_data.json"
    with open(output_file, 'w') as f:
        json.dump(policy_data, f, indent=2)
    
    print(f"Saved policy data to {output_file}")
    
    conn.close()
    return policy_data


if __name__ == "__main__":
    # Extract policy data from all moves
    extract_policy_from_evaluations(limit=10000)
    
    # Or extract from a subset
    # extract_policy_from_evaluations(limit=1000)

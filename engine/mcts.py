import math
import random
import chess
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from classifiers.move_classifier import MoveClassifier

class MCTSNode:
    """MCTS tree node."""
    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None, move: Optional[chess.Move] = None):
        self.board = board
        self.parent = parent
        self.move = move  # Move that led to this position
        
        self.is_expanded = False
        self.children: Dict[chess.Move, 'MCTSNode'] = {}
        
        # MCTS statistics
        self.visit_count = 0
        self.total_value = 0.0  # Cumulative value
        
        # Prior probabilities from classifier (Policy/Prior)
        self.priors: Dict[chess.Move, float] = {}

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def get_ucb_score(self, move: chess.Move, cpuct: float = 1.5) -> float:
        """Calculate UCB (Upper Confidence Bound) metric for move selection."""
        child = self.children.get(move)
        p_score = self.priors.get(move, 0.0)
        
        if child:
            q_score = child.value
            u_score = cpuct * p_score * math.sqrt(self.visit_count) / (1 + child.visit_count)
            return q_score + u_score
        else:
            # If node not yet visited, it has high potential due to Prior
            return cpuct * p_score * math.sqrt(self.visit_count + 1e-6)


class MoveClassifierMCTS:
    """MCTS engine integrated with your neural network."""
    def __init__(self, classifier: MoveClassifier, cpuct: float = 1.5):
        self.classifier = classifier
        self.cpuct = cpuct
        
        # Weights for move types (converting classification to "Policy" probabilities)
        self.class_priority = {
            "Best": 1.0,
            "Excellent": 0.8,
            "Good": 0.5,
            "Inaccuracy": 0.2,
            "Mistake": 0.05,
            "Blunder": 0.001
        }

    def _evaluate_node_priors(self, node: MCTSNode):
        """Evaluates all legal moves from node using neural network (filling Priors)."""
        legal_moves = list(node.board.legal_moves)
        if not legal_moves:
            return

        # Convert neural network classes to probability distribution (Priors)
        raw_scores = []
        
        # Make predictions (ideally batch here, but for now — quick loop)
        for move in legal_moves:
            move_san = node.board.san(move)
            
            # Virtual move
            node.board.push(move)
            fen_after = node.board.fen()
            node.board.pop()
            
            # Network inference
            res = self.classifier.classify_move(node.board.fen(), move_san)
            
            # Get base class weight and multiply by network confidence
            score = self.class_priority.get(res.classification, 0.1) * res.confidence
            raw_scores.append(score)
            
        # Normalize via Softmax (sum of probabilities equals 1.0)
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        probabilities = exp_scores / exp_scores.sum()
        
        for move, prob in zip(legal_moves, probabilities):
            node.priors[move] = prob

    def search(self, initial_board: chess.Board, num_simulations: int = 100) -> chess.Move:
        """Main search loop for best move."""
        if initial_board.is_game_over():
            return None
        
        root = MCTSNode(board=initial_board.copy())
        self._evaluate_node_priors(root)
        root.is_expanded = True
        
        for _ in range(num_simulations):
            node = root
            
            # 1. SELECTION - Descend to leaf using UCB
            while node.is_expanded and node.board.legal_moves:
                legal_moves = list(node.board.legal_moves)
                
                # If moves not yet evaluated by network for this node, evaluate them
                if not node.priors:
                    self._evaluate_node_priors(node)
                    
                best_move = max(legal_moves, key=lambda m: node.get_ucb_score(m, self.cpuct))
                
                if best_move not in node.children:
                    # Create new leaf node
                    next_board = node.board.copy()
                    next_board.push(best_move)
                    node.children[best_move] = MCTSNode(board=next_board, parent=node, move=best_move)
                
                node = node.children[best_move]
            
            # 2. EXPANSION and SIMULATION
            # Since we don't have a "Value" head yet (which would say +0.5 or -0.7),
            # we do a classic rollout (random play to finish) or use material evaluation.
            value = self._rollout(node.board)
            
            # If this was a black move, invert evaluation for backpropagation
            if node.board.turn != initial_board.turn:
                value = -value
                
            # 3. BACKPROPAGATION - Go up and update statistics
            while node is not None:
                node.visit_count += 1
                node.total_value += value
                value = -value  # Flip sign at each tree level (minimax logic)
                node = node.parent
                
        # Select move visited most often (most reliable)
        best_move = max(root.children.keys(), key=lambda m: root.children[m].visit_count)
        return best_move

    def _rollout(self, board: chess.Board) -> float:
        """Fast heuristic position evaluation (material balance).
        Returns value from -1.0 (black wins) to 1.0 (white wins)."""
        if board.is_game_over():
            result = board.result()
            if result == "1-0": return 1.0
            if result == "0-1": return -1.0
            return 0.0

        # Calculate material on board
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                val = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += val
                else:
                    black_material += val
                    
        # Normalize difference to interval [-1, 1]
        diff = white_material - black_material
        # 15 pawn advantage limit (anything above — absolute win)
        value = math.tanh(diff / 15.0)
        return value
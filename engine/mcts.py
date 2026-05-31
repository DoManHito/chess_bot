import math
import random
import chess
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

class MCTSNode:
    """Узел дерева MCTS."""
    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None, move: Optional[chess.Move] = None):
        self.board = board
        self.parent = parent
        self.move = move  
        
        self.is_expanded = False
        self.children: Dict[chess.Move, 'MCTSNode'] = {}
        
        self.visit_count = 0
        self.total_value = 0.0  
        self.priors: Dict[chess.Move, float] = {}

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def get_ucb_score(self, move: chess.Move, cpuct: float = 1.5) -> float:
        child = self.children.get(move)
        p_score = self.priors.get(move, 0.0)
        
        child_visits = child.visit_count if child else 0
        q_score = -child.value if child else 0.0
        u_score = cpuct * p_score * math.sqrt(self.visit_count) / (1 + child_visits)
        return q_score + u_score


class MoveClassifierMCTS:
    """MCTS движок, работающий в связке с классификатором ходов."""
    def __init__(self, classifier, cpuct: float = 1.5):
        self.classifier = classifier
        self.cpuct = cpuct
        
        self.class_priority = {
            "Best": 1.0,
            "Excellent": 0.8,
            "Good": 0.5,
            "Inaccuracy": 0.2,
            "Mistake": 0.05,
            "Blunder": 0.001
        }

    def _get_terminal_value(self, board: chess.Board) -> float:
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 1.0 if board.turn == chess.WHITE else -1.0
            if result == "0-1":
                return 1.0 if board.turn == chess.BLACK else -1.0
        return 0.0

    def _evaluate_and_expand_node(self, node: MCTSNode) -> float:
        if node.board.is_game_over():
            return self._get_terminal_value(node.board)

        legal_moves = list(node.board.legal_moves)
        raw_scores = []
        leaf_values = []
        
        for move in legal_moves:
            move_san = node.board.san(move)
            res, val = self.classifier.classify_move(node.board.fen(), move_san)
            
            # Инвертируем оценку позиции, если сейчас ход Чёрных,
            # так как нейросеть предсказывает Value относительно текущего игрока.
            if node.board.turn == chess.BLACK:
                val = -val

            score = self.class_priority.get(res.classification, 0.1) * res.confidence
            raw_scores.append(score)
            leaf_values.append(val)
            
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        probabilities = exp_scores / exp_scores.sum()
        
        for move, prob in zip(legal_moves, probabilities):
            node.priors[move] = prob
            
        node.is_expanded = True
        return float(np.mean(leaf_values))

    def search(self, initial_board: chess.Board, num_simulations: int = 100) -> Tuple[Optional[chess.Move], Dict[str, float]]:
        """Выполняет MCTS поиск. Возвращает выбранный ход и словарь визитов для политики."""
        if initial_board.is_game_over():
            return None, {}
        
        root = MCTSNode(board=initial_board.copy())
        _ = self._evaluate_and_expand_node(root)
        
        for _ in range(num_simulations):
            node = root
            
            # 1. SELECTION
            while node.is_expanded and not node.board.is_game_over():
                legal_moves = list(node.board.legal_moves)
                if not legal_moves:
                    break
                    
                best_move = max(legal_moves, key=lambda m: node.get_ucb_score(m, self.cpuct))
                
                if best_move not in node.children:
                    next_board = node.board.copy()
                    next_board.push(best_move)
                    node.children[best_move] = MCTSNode(board=next_board, parent=node, move=best_move)
                
                node = node.children[best_move]
            
            # 2. EXPANSION & EVALUATION
            if not node.is_expanded:
                value = self._evaluate_and_expand_node(node)
            else:
                value = self._get_terminal_value(node.board)
                
            # 3. BACKPROPAGATION
            while node is not None:
                node.visit_count += 1
                node.total_value += value
                value = -value
                node = node.parent
                
        legal_moves = list(initial_board.legal_moves)
        if not root.children:
            default_move = random.choice(legal_moves) if legal_moves else None
            return default_move, {m.uci(): 1.0/len(legal_moves) for m in legal_moves} if legal_moves else {}
            
        # Формируем словарь визитов ходов
        visit_dict = {m.uci(): float(root.children[m].visit_count) if m in root.children else 0.0 for m in legal_moves}
        best_move = max(root.children.keys(), key=lambda m: root.children[m].visit_count)
        
        return best_move, visit_dict
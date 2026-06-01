import math
import random
import chess
import numpy as np
from typing import Dict, Optional, Tuple
from functools import lru_cache
import time

class MCTSNode:
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
    def __init__(self, classifier, cpuct: float = 2.0, max_simulations: int = 20, top_moves_ratio: float = 0.3):
        self.classifier = classifier
        self.cpuct = cpuct
        self.max_simulations = max_simulations
        self.top_moves_ratio = top_moves_ratio
        self.class_priority = {
            "Best": 1.0, "Excellent": 0.8, "Good": 0.5,
            "Inaccuracy": 0.2, "Mistake": 0.05, "Blunder": 0.001
        }
        # Board state cache: FEN -> (visit_count, total_value)
        self.board_cache: Dict[str, Tuple[int, float]] = {}
        # Move evaluation cache: (FEN, move_san) -> (class, confidence, value)
        self.move_cache: Dict[Tuple[str, str], Tuple[str, float, float]] = {}

    def _get_terminal_value(self, board: chess.Board) -> float:
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 1.0 if board.turn == chess.WHITE else -1.0
            if result == "0-1":
                return 1.0 if board.turn == chess.BLACK else -1.0
        return 0.0

    def _get_fen(self, board: chess.Board) -> str:
        """Get normalized FEN for caching"""
        return board.fen()

    def _evaluate_and_expand_node(self, node: MCTSNode) -> float:
        if node.board.is_game_over():
            return self._get_terminal_value(node.board)

        legal_moves = list(node.board.legal_moves)
        if not legal_moves:
            node.is_expanded = True
            return self._get_terminal_value(node.board)

        # Get FEN for caching
        board_fen = self._get_fen(node.board)
        
        # Check if we already evaluated this position
        if board_fen in self.board_cache:
            cached_visits, cached_value = self.board_cache[board_fen]
            # If cached visits are high enough, use cached value
            if cached_visits >= 3:
                node.visit_count = cached_visits
                node.total_value = cached_value
                node.is_expanded = True
                return cached_value

        # Move prioritization: only evaluate top N moves
        num_to_evaluate = max(1, int(len(legal_moves) * self.top_moves_ratio))
        
        # Sort moves by some heuristic (prioritize captures and checks)
        move_scores = []
        for m in legal_moves:
            score = 0.0
            move_san = node.board.san(m)
            
            # Prioritize captures
            captured = node.board.piece_at(m.to_square)
            if captured:
                piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
                score += piece_values.get(captured.piece_type, 0) * 10
            
            # Prioritize checks
            board_copy = node.board.copy()
            board_copy.push(m)
            if board_copy.is_check():
                score += 5.0
            
            move_scores.append((m, score))
        
        # Sort by score descending
        move_scores.sort(key=lambda x: x[1], reverse=True)
        top_moves = [m for m, _ in move_scores[:num_to_evaluate]]
        
        # If we have fewer moves than we need to evaluate, evaluate all
        if len(top_moves) < num_to_evaluate:
            top_moves = legal_moves
        
        moves_san = [node.board.san(m) for m in top_moves]

        try:
            classes, confidences, values = self.classifier.classify_moves_batch(node.board, moves_san)
        except Exception as e:
            print(f"Batch error, falling back to sequential: {e}")
            classes, confidences, values = [], [], []
            for move in top_moves:
                res, val = self.classifier.classify_move(node.board.fen(), node.board.san(move))
                classes.append(res.classification)
                confidences.append(res.confidence)
                values.append(val)

        raw_scores = []
        leaf_values = []
        for cls, conf, val in zip(classes, confidences, values):
            score = self.class_priority.get(cls, 0.1) * conf
            raw_scores.append(score)
            leaf_values.append(val)

        if not raw_scores:
            raw_scores = [0.1] * len(top_moves)
            leaf_values = [0.0] * len(top_moves)

        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        probabilities = exp_scores / exp_scores.sum()

        if node.parent is None:
            dirichlet_alpha = 0.03
            noise = np.random.dirichlet([dirichlet_alpha] * len(top_moves))
            probabilities = 0.75 * probabilities + 0.25 * noise

        for move, prob in zip(top_moves, probabilities):
            node.priors[move] = prob

        node.is_expanded = True
        
        # Cache the board evaluation
        self.board_cache[board_fen] = (len(top_moves), float(np.mean(leaf_values)))
        
        return float(np.mean(leaf_values))

    def search(self, initial_board: chess.Board, num_simulations: int = None) -> Tuple[Optional[chess.Move], Dict[str, float]]:
        start_time = time.time()
        print(f"\n[DEBUG] MCTS search started with {num_simulations or self.max_simulations} simulations")
        print(f"[DEBUG] Legal moves count: {len(list(initial_board.legal_moves))}")
        
        if initial_board.is_game_over():
            return None, {}

        root = MCTSNode(board=initial_board.copy())
        
        # Timeout mechanism
        timeout_seconds = 5.0
        max_iterations = num_simulations or self.max_simulations
        
        for i in range(max_iterations):
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                print(f"[DEBUG] Timeout reached after {i+1} simulations")
                break
            
            node = root
            # Selection
            selection_start = time.time()
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
            selection_time = time.time() - selection_start
            
            # Expansion & Evaluation
            if not node.is_expanded:
                eval_start = time.time()
                value = self._evaluate_and_expand_node(node)
                eval_time = time.time() - eval_start
            else:
                value = self._get_terminal_value(node.board)
                eval_time = 0
            
            # Backpropagation
            while node is not None:
                node.visit_count += 1
                node.total_value += value
                value = -value
                node = node.parent
            
            if (i + 1) % 10 == 0:
                total_time = time.time() - start_time
                print(f"[DEBUG] Simulation {i+1}/{max_iterations} - Total: {total_time:.2f}s, Avg sim: {total_time/(i+1)*100:.1f}ms")

        legal_moves = list(initial_board.legal_moves)
        if not root.children:
            default_move = random.choice(legal_moves) if legal_moves else None
            uniform_policy = {m.uci(): 1.0 / len(legal_moves) for m in legal_moves} if legal_moves else {}
            total_time = time.time() - start_time
            print(f"[DEBUG] MCTS search completed in {total_time:.2f}s")
            return default_move, uniform_policy

        visit_dict = {m.uci(): float(root.children[m].visit_count) if m in root.children else 0.0 for m in legal_moves}
        best_move = max(root.children.keys(), key=lambda m: root.children[m].visit_count)
        total_time = time.time() - start_time
        print(f"[DEBUG] MCTS search completed in {total_time:.2f}s")
        return best_move, visit_dict
"""
Unified MCTS Engine with Policy Head Integration.

This module implements an MCTS engine that uses the unified model's policy head
for move prioritization, significantly accelerating the search process.
"""

import math
import random
import chess
import numpy as np
import time
from typing import Dict, Optional, Tuple
from models.unified_chess_nets import ChessCoreNet, UnifiedMoveClassifierNet, CLASS_NAMES


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search (MCTS) tree.

    Each node represents a chess board position and stores information about
    simulations that have passed through it.
    """
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
        """Calculate the UCB1 score for a move."""
        child = self.children.get(move)
        p_score = self.priors.get(move, 0.0)
        child_visits = child.visit_count if child else 0
        q_score = -child.value if child else 0.0
        u_score = cpuct * p_score * math.sqrt(self.visit_count) / (1 + child_visits)
        return q_score + u_score


class UnifiedMCTS:
    """
    Monte Carlo Tree Search (MCTS) engine optimized for chess move classification.

    This class implements the MCTS algorithm enhanced with:
    1. **Unified Model Policy**: Uses the unified model's policy head for move priors
    2. **Move Prioritization**: Only evaluates the top N moves based on heuristics
    3. **Caching**: Stores board evaluations to avoid redundant computations

    The MCTS algorithm follows four phases:
    1. **Selection**: Traverse the tree using UCB1 to select the most promising node
    2. **Expansion**: Add a new child node
    3. **Evaluation**: Use the unified model to evaluate the new position
    4. **Backpropagation**: Propagate the evaluation value back up the tree
    """
    def __init__(self, unified_model: UnifiedMoveClassifierNet, cpuct: float = 2.0,
                 max_simulations: int = 20, top_moves_ratio: float = 0.3, policy_output_dim: int = 64):
        """
        Initialize the MCTS engine.

        Args:
            unified_model: UnifiedMoveClassifierNet instance for policy and value predictions
            cpuct: UCB1 exploration constant (default: 2.0)
            max_simulations: Maximum number of simulations per search (default: 20)
            top_moves_ratio: Fraction of legal moves to evaluate (default: 0.3)
            policy_output_dim: Dimension of policy output (default: 64)
        """
        self.model = unified_model
        self.cpuct = cpuct
        self.max_simulations = max_simulations
        self.top_moves_ratio = top_moves_ratio
        self.class_priority = {
            "Best": 1.0, "Excellent": 0.8, "Good": 0.5,
            "Inaccuracy": 0.2, "Mistake": 0.05, "Blunder": 0.001
        }
        self.board_cache: Dict[str, Tuple[int, float]] = {}
        self.move_cache: Dict[Tuple[str, str], Tuple[str, float, float]] = {}
        self.policy_output_dim = policy_output_dim

    def _get_terminal_value(self, board: chess.Board) -> float:
        """Get the evaluation value for a terminal game state."""
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 1.0 if board.turn == chess.WHITE else -1.0
            if result == "0-1":
                return 1.0 if board.turn == chess.BLACK else -1.0
        return 0.0

    def _get_fen(self, board: chess.Board) -> str:
        return board.fen()

    def _evaluate_and_expand_node(self, node: MCTSNode) -> float:
        """
        Evaluate a node and expand it with children.

        Args:
            node: The MCTS node to evaluate and expand

        Returns:
            Average evaluation value of the expanded children
        """
        # Check if game is over
        if node.board.is_game_over():
            return self._get_terminal_value(node.board)

        legal_moves = list(node.board.legal_moves)
        if not legal_moves:
            node.is_expanded = True
            return self._get_terminal_value(node.board)

        # Get FEN for caching
        board_fen = self._get_fen(node.board)

        # Check cache
        if board_fen in self.board_cache:
            cached_visits, cached_value = self.board_cache[board_fen]
            if cached_visits >= 3:
                node.visit_count = cached_visits
                node.total_value = cached_value
                node.is_expanded = True
                return cached_value

        # Move prioritization: only evaluate top N moves
        num_to_evaluate = max(1, int(len(legal_moves) * self.top_moves_ratio))

        # Sort moves by heuristic score
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

        move_scores.sort(key=lambda x: x[1], reverse=True)
        top_moves = [m for m, _ in move_scores[:num_to_evaluate]]

        if len(top_moves) < num_to_evaluate:
            top_moves = legal_moves

        # Get policy and values from unified model
        try:
            classes, confidences, values = self.model.classify_moves_batch(node.board, [m.uci() for m in top_moves])
        except Exception as e:
            print(f"Batch error, falling back to sequential: {e}")
            classes, confidences, values = [], [], []
            for move in top_moves:
                try:
                    san = node.board.san(move)
                    res, val = self.model.classify_move(node.board.fen(), san)
                    classes.append(res.classification)
                    confidences.append(res.confidence)
                    values.append(val)
                except Exception as e2:
                    print(f"Sequential error for move {move}: {e2}")
                    classes.append("Good")
                    confidences.append(0.5)
                    values.append(0.0)

        # Convert classifications to scores
        raw_scores = []
        leaf_values = []
        for cls, conf, val in zip(classes, confidences, values):
            score = self.class_priority.get(cls, 0.1) * conf
            raw_scores.append(score)
            leaf_values.append(val)

        if not raw_scores:
            raw_scores = [0.1] * len(top_moves)
            leaf_values = [0.0] * len(top_moves)

        # Softmax for policy
        exp_scores = np.exp(np.array(raw_scores) - np.max(np.array(raw_scores)))
        probabilities = exp_scores / exp_scores.sum()

        # Apply Dirichlet noise at root for exploration
        if node.parent is None:
            dirichlet_alpha = 0.03
            noise = np.random.dirichlet([dirichlet_alpha] * len(top_moves))
            probabilities = 0.75 * probabilities + 0.25 * noise

        # Store priors for UCB calculation
        for move, prob in zip(top_moves, probabilities):
            node.priors[move] = prob

        node.is_expanded = True

        # Cache the board evaluation
        self.board_cache[board_fen] = (len(top_moves), float(np.mean(leaf_values)))

        return float(np.mean(leaf_values))

    def search(self, initial_board: chess.Board, num_simulations: int = None) -> Tuple[Optional[chess.Move], Dict[str, float]]:
        """
        Run MCTS search to find the best move.

        Args:
            initial_board: Starting chess board position
            num_simulations: Number of simulations to run (uses max_simulations if None)

        Returns:
            Tuple of (best_move, visit_dict):
            - best_move: The best move found (or None if game over)
            - visit_dict: Dictionary mapping move UCI to visit count
        """
        start_time = time.time()

        if initial_board.is_game_over():
            return None, {}

        root = MCTSNode(board=initial_board.copy())

        timeout_seconds = 5.0
        max_iterations = num_simulations or self.max_simulations

        for i in range(max_iterations):
            if time.time() - start_time > timeout_seconds:
                break

            node = root

            # SELECTION PHASE
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

            # EXPANSION & EVALUATION PHASE
            if not node.is_expanded:
                value = self._evaluate_and_expand_node(node)
            else:
                value = self._get_terminal_value(node.board)

            # BACKPROPAGATION PHASE
            while node is not None:
                node.visit_count += 1
                node.total_value += value
                value = -value
                node = node.parent

        # Get the best move from root's children
        legal_moves = list(initial_board.legal_moves)
        if not root.children:
            default_move = random.choice(legal_moves) if legal_moves else None
            uniform_policy = {m.uci(): 1.0 / len(legal_moves) for m in legal_moves} if legal_moves else {}
            total_time = time.time() - start_time
            return default_move, uniform_policy

        # Build visit dictionary for all legal moves
        visit_dict = {m.uci(): float(root.children[m].visit_count) if m in root.children else 0.0 for m in legal_moves}
        best_move = max(root.children.keys(), key=lambda m: root.children[m].visit_count)

        total_time = time.time() - start_time

        return best_move, visit_dict

    def get_policy(self, board: chess.Board, top_k: int = 64) -> Dict[str, float]:
        """
        Get policy probabilities for top K moves from the unified model.

        Args:
            board: Chess board position
            top_k: Number of top moves to return (default: 64)

        Returns:
            Dictionary mapping move UCI to probability
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}

        # Get policy from model
        policy_logits = self.model.get_policy(board, top_k)

        policy = {}
        for move in legal_moves:
            policy[move.uci()] = policy_logits[move.to_square].item()

        # Normalize
        total = sum(policy.values())
        if total > 0:
            policy = {k: v / total for k, v in policy.items()}

        return policy

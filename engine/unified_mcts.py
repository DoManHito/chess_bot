import math
import random
import chess
import numpy as np
import time
from typing import Dict, Optional, Tuple
from models.unified_chess_nets import UnifiedMoveClassifierNet, CLASS_NAMES
from classifiers.move_classifier import MoveData

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
        """Calculate the UCB1 score for a move."""
        child = self.children.get(move)
        p_score = self.priors.get(move, 0.0)
        child_visits = child.visit_count if child else 0
        q_score = -child.value if child else 0.0
        u_score = cpuct * p_score * math.sqrt(self.visit_count) / (1 + child_visits)
        return q_score + u_score


class UnifiedMCTS:
    def __init__(self, unified_model: UnifiedMoveClassifierNet, cpuct: float = 2.0,
                 max_simulations: int = 20, top_moves_ratio: float = 0.3, policy_output_dim: int = 64):
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

        board_fen = node.board.fen()
        turn_num = node.board.fullmove_number
        turn_label = "White" if node.board.turn == chess.WHITE else "Black"

        batch_moves = [
            MoveData(
                board_fen=board_fen,
                move_san=node.board.san(m),
                turn_num=turn_num,
                turn_label=turn_label
            )
            for m in top_moves
        ]

        try:
            batch_results = self.model.classify_moves_batch(batch_moves)
            
            classes = [r.classification for r in batch_results]
            confidences = [r.confidence for r in batch_results]
            values = [r.evaluation for r in batch_results]
        except Exception as e:
            print(f"Batch error, falling back to sequential: {e}")
            classes, confidences, values = [], [], []
            for move in top_moves:
                try:
                    san = node.board.san(move)
                    res = self.model.classify_move(
                        board_fen, 
                        san, 
                        turn_num=turn_num, 
                        turn_label=turn_label
                    )
                    classes.append(res.classification)
                    confidences.append(res.confidence)
                    values.append(res.evaluation)
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

    def search(self, initial_board: chess.Board, num_simulations: int = None, temperature: float = 0.0) -> Tuple[Optional[chess.Move], Dict[str, float]]:
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

        legal_moves = list(initial_board.legal_moves)
        if not root.children:
            default_move = random.choice(legal_moves) if legal_moves else None
            uniform_policy = {m.uci(): 1.0 / len(legal_moves) for m in legal_moves} if legal_moves else {}
            return default_move, uniform_policy

        visit_dict = {m.uci(): float(root.children[m].visit_count) if m in root.children else 0.0 for m in legal_moves}

        children_moves = list(root.children.keys())
        visit_counts = np.array([float(root.children[m].visit_count) for m in children_moves], dtype=np.float32)

        if temperature <= 0.0:
            best_move = children_moves[np.argmax(visit_counts)]
        else:
            power_counts = visit_counts ** (1.0 / temperature)
            sum_power = np.sum(power_counts)
            
            if sum_power > 0:
                probabilities = power_counts / sum_power
                chosen_idx = np.random.choice(len(children_moves), p=probabilities)
                best_move = children_moves[chosen_idx]
            else:
                best_move = max(root.children.keys(), key=lambda m: root.children[m].visit_count)

        total_time = time.time() - start_time
        return best_move, visit_dict

    def get_policy(self, board: chess.Board, top_k: int = 64) -> Dict[str, float]:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}

        policy_logits = self.model.get_policy(board, top_k)

        policy = {}
        for idx, move in enumerate(legal_moves):
            policy[move.uci()] = policy_logits[idx].item()

        total = sum(policy.values())
        if total > 0:
            policy = {k: v / total for k, v in policy.items()}

        return policy

import math
import random
import chess
import numpy as np
from typing import Dict, Optional, Tuple
from functools import lru_cache
import time


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search (MCTS) tree.

    Each node represents a chess board position and stores information about
    simulations that have passed through it. The node structure enables the
    MCTS algorithm to efficiently explore and exploit the game tree.

    Attributes:
        board: Chess.Board object representing the position
        parent: Parent node in the tree (None for root)
        move: The move that led to this position (None for root)
        is_expanded: Whether the node has been expanded (children added)
        children: Dictionary mapping moves to child nodes
        visit_count: Number of times this node has been visited
        total_value: Sum of values from all simulations through this node
        priors: Dictionary of move probabilities (from neural network policy)
    """
    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None, move: Optional[chess.Move] = None):
        """
        Initialize an MCTS node.

        Args:
            board: Chess board state at this node
            parent: Parent node in the tree (None for root)
            move: Move that led to this position (None for root)
        """
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
        """
        Get the average value of this node.

        Returns the average value from all simulations that passed through
        this node. Returns 0.0 if the node has never been visited.

        Returns:
            Average value (total_value / visit_count)
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def get_ucb_score(self, move: chess.Move, cpuct: float = 1.5) -> float:
        """
        Calculate the UCB1 score for a move.

        The Upper Confidence Bound (UCB1) formula balances exploration and
        exploitation. It consists of two components:
        - Q-score: The average value of the child node (exploitation)
        - U-score: Exploration bonus based on visit counts

        Args:
            move: The move to evaluate
            cpuct: Exploration constant (default: 1.5)

        Returns:
            UCB1 score for the move. Higher scores indicate better moves.
        """
        child = self.children.get(move)
        # Prior score from neural network policy
        p_score = self.priors.get(move, 0.0)
        # Child's visit count (0 if child doesn't exist)
        child_visits = child.visit_count if child else 0
        # Q-score: negative because we want to maximize from current player's perspective
        q_score = -child.value if child else 0.0
        # U-score: exploration bonus
        u_score = cpuct * p_score * math.sqrt(self.visit_count) / (1 + child_visits)
        return q_score + u_score


class MoveClassifierMCTS:
    """
    Monte Carlo Tree Search (MCTS) engine optimized for chess move classification.

    This class implements the MCTS algorithm enhanced with:
    1. **Neural Network Guidance**: Uses a trained classifier to provide move priors
       and evaluation values, reducing the search space.
    2. **Move Prioritization**: Only evaluates the top N moves based on heuristics
       (captures, checks), significantly speeding up search.
    3. **Caching**: Stores board evaluations to avoid redundant computations.
    4. **Dirichlet Noise**: Adds exploration noise at the root to prevent premature
       convergence to suboptimal moves.

    The MCTS algorithm follows four phases:
    1. **Selection**: Traverse the tree using UCB1 to select the most promising node
    2. **Expansion**: Add a new child node (expand the tree)
    3. **Evaluation**: Use the neural network to evaluate the new position
    4. **Backpropagation**: Propagate the evaluation value back up the tree

    Args:
        classifier: MoveClassifier instance for neural network guidance
        cpuct: UCB1 exploration constant (default: 2.0)
        max_simulations: Maximum number of simulations per search (default: 20)
        top_moves_ratio: Fraction of legal moves to evaluate (default: 0.3)

    Attributes:
        classifier: The neural network classifier
        cpuct: UCB1 exploration constant
        max_simulations: Maximum simulations per search
        top_moves_ratio: Ratio of moves to evaluate
        class_priority: Priority weights for each move classification
        board_cache: Cache of board evaluations (FEN -> visits, value)
        move_cache: Cache of move evaluations
    """
    def __init__(self, classifier, cpuct: float = 2.0, max_simulations: int = 20, top_moves_ratio: float = 0.3):
        """
        Initialize the MCTS engine.

        Args:
            classifier: MoveClassifier instance for neural network guidance
            cpuct: UCB1 exploration constant (higher = more exploration)
            max_simulations: Maximum number of simulations per search
            top_moves_ratio: Fraction of legal moves to evaluate (e.g., 0.3 = 30%)
        """
        self.classifier = classifier
        self.cpuct = cpuct
        self.max_simulations = max_simulations
        self.top_moves_ratio = top_moves_ratio
        # Priority weights for move classifications (higher = better moves)
        self.class_priority = {
            "Best": 1.0, "Excellent": 0.8, "Good": 0.5,
            "Inaccuracy": 0.2, "Mistake": 0.05, "Blunder": 0.001
        }
        # Board state cache: FEN -> (visit_count, total_value)
        # Used to avoid re-evaluating the same position
        self.board_cache: Dict[str, Tuple[int, float]] = {}
        # Move evaluation cache: (FEN, move_san) -> (class, confidence, value)
        self.move_cache: Dict[Tuple[str, str], Tuple[str, float, float]] = {}

    def _get_terminal_value(self, board: chess.Board) -> float:
        """
        Get the evaluation value for a terminal game state.

        Args:
            board: Chess board state

        Returns:
            Evaluation value:
            - 1.0 if White won (and it's White's turn)
            - -1.0 if White won (and it's Black's turn)
            - 1.0 if Black won (and it's Black's turn)
            - -1.0 if Black won (and it's White's turn)
            - 0.0 for draws
        """
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 1.0 if board.turn == chess.WHITE else -1.0
            if result == "0-1":
                return 1.0 if board.turn == chess.BLACK else -1.0
        return 0.0

    def _get_fen(self, board: chess.Board) -> str:
        """
        Get the FEN string for a board position.

        Args:
            board: Chess board state

        Returns:
            FEN string representing the board position
        """
        return board.fen()

    def _evaluate_and_expand_node(self, node: MCTSNode) -> float:
        """
        Evaluate a node and expand it with children.

        This method:
        1. Checks if the game is over (terminal state)
        2. Uses caching to avoid re-evaluating positions
        3. Prioritizes moves using heuristics (captures, checks)
        4. Evaluates only the top N moves to save computation
        5. Uses the neural network to get move priors and values
        6. Applies Dirichlet noise at the root for exploration
        7. Caches the board evaluation for future use

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
        # This dramatically reduces computation by focusing on promising moves
        num_to_evaluate = max(1, int(len(legal_moves) * self.top_moves_ratio))

        # Sort moves by heuristic score (prioritize captures and checks)
        move_scores = []
        for m in legal_moves:
            score = 0.0
            move_san = node.board.san(m)

            # Prioritize captures: higher value pieces = higher score
            captured = node.board.piece_at(m.to_square)
            if captured:
                piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
                score += piece_values.get(captured.piece_type, 0) * 10

            # Prioritize checks: giving check is strategically important
            board_copy = node.board.copy()
            board_copy.push(m)
            if board_copy.is_check():
                score += 5.0

            move_scores.append((m, score))

        # Sort by score descending (best moves first)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        top_moves = [m for m, _ in move_scores[:num_to_evaluate]]

        # If we have fewer moves than we need to evaluate, evaluate all
        if len(top_moves) < num_to_evaluate:
            top_moves = legal_moves

        moves_san = [node.board.san(m) for m in top_moves]

        # Get neural network evaluation for all top moves
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

        # Convert classifications to scores using priority weights
        raw_scores = []
        leaf_values = []
        for cls, conf, val in zip(classes, confidences, values):
            # Score = priority weight * confidence
            score = self.class_priority.get(cls, 0.1) * conf
            raw_scores.append(score)
            leaf_values.append(val)

        # Fallback if no scores were generated
        if not raw_scores:
            raw_scores = [0.1] * len(top_moves)
            leaf_values = [0.0] * len(top_moves)

        # Convert to probabilities using softmax
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        probabilities = exp_scores / exp_scores.sum()

        # Apply Dirichlet noise at the root for exploration
        # This prevents premature convergence to suboptimal moves
        if node.parent is None:
            dirichlet_alpha = 0.03
            noise = np.random.dirichlet([dirichlet_alpha] * len(top_moves))
            probabilities = 0.75 * probabilities + 0.25 * noise

        # Store priors for UCB calculation
        for move, prob in zip(top_moves, probabilities):
            node.priors[move] = prob

        node.is_expanded = True

        # Cache the board evaluation for future use
        self.board_cache[board_fen] = (len(top_moves), float(np.mean(leaf_values)))

        return float(np.mean(leaf_values))

    def search(self, initial_board: chess.Board, num_simulations: int = None) -> Tuple[Optional[chess.Move], Dict[str, float]]:
        """
        Run MCTS search to find the best move.

        This method implements the complete MCTS algorithm:
        1. **Selection**: Traverse the tree using UCB1 until reaching an unexpanded node
        2. **Expansion**: Add a new child node
        3. **Evaluation**: Evaluate the new position using the neural network
        4. **Backpropagation**: Propagate the value back up the tree

        Args:
            initial_board: Starting chess board position
            num_simulations: Number of simulations to run (uses max_simulations if None)

        Returns:
            Tuple of (best_move, visit_dict):
            - best_move: The best move found (or None if game over)
            - visit_dict: Dictionary mapping move UCI to visit count
        """
        start_time = time.time()
        print(f"\n[DEBUG] MCTS search started with {num_simulations or self.max_simulations} simulations")
        print(f"[DEBUG] Legal moves count: {len(list(initial_board.legal_moves))}")

        # Check if game is already over
        if initial_board.is_game_over():
            return None, {}

        root = MCTSNode(board=initial_board.copy())

        # Timeout mechanism to prevent infinite loops
        timeout_seconds = 5.0
        max_iterations = num_simulations or self.max_simulations

        for i in range(max_iterations):
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                print(f"[DEBUG] Timeout reached after {i+1} simulations")
                break

            node = root
            # === SELECTION PHASE ===
            selection_start = time.time()
            # Traverse the tree using UCB1 until reaching an unexpanded node
            while node.is_expanded and not node.board.is_game_over():
                legal_moves = list(node.board.legal_moves)
                if not legal_moves:
                    break
                # Select move with highest UCB score
                best_move = max(legal_moves, key=lambda m: node.get_ucb_score(m, self.cpuct))
                if best_move not in node.children:
                    # Expand: create a new child node
                    next_board = node.board.copy()
                    next_board.push(best_move)
                    node.children[best_move] = MCTSNode(board=next_board, parent=node, move=best_move)
                node = node.children[best_move]
            selection_time = time.time() - selection_start

            # === EXPANSION & EVALUATION PHASE ===
            if not node.is_expanded:
                eval_start = time.time()
                value = self._evaluate_and_expand_node(node)
                eval_time = time.time() - eval_start
            else:
                value = self._get_terminal_value(node.board)
                eval_time = 0

            # === BACKPROPAGATION PHASE ===
            # Propagate the value back up the tree
            while node is not None:
                node.visit_count += 1
                node.total_value += value
                # Alternate sign for opponent's perspective
                value = -value
                node = node.parent

            # Progress reporting
            if (i + 1) % 10 == 0:
                total_time = time.time() - start_time
                print(f"[DEBUG] Simulation {i+1}/{max_iterations} - Total: {total_time:.2f}s, Avg sim: {total_time/(i+1)*100:.1f}ms")

        # Get the best move from the root's children
        legal_moves = list(initial_board.legal_moves)
        if not root.children:
            # No children created, return random move
            default_move = random.choice(legal_moves) if legal_moves else None
            uniform_policy = {m.uci(): 1.0 / len(legal_moves) for m in legal_moves} if legal_moves else {}
            total_time = time.time() - start_time
            print(f"[DEBUG] MCTS search completed in {total_time:.2f}s")
            return default_move, uniform_policy

        # Build visit dictionary for all legal moves
        visit_dict = {m.uci(): float(root.children[m].visit_count) if m in root.children else 0.0 for m in legal_moves}
        # Select the move with most visits
        best_move = max(root.children.keys(), key=lambda m: root.children[m].visit_count)
        total_time = time.time() - start_time
        print(f"[DEBUG] MCTS search completed in {total_time:.2f}s")
        return best_move, visit_dict
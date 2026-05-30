import math
import random
import chess
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from classifiers.move_classifier import MoveClassifier

class MCTSNode:
    """Узел дерева MCTS."""
    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None, move: Optional[chess.Move] = None):
        self.board = board
        self.parent = parent
        self.move = move  # Ход, который привёл в эту позицию
        
        self.is_expanded = False
        self.children: Dict[chess.Move, 'MCTSNode'] = {}
        
        # Статистика MCTS
        self.visit_count = 0
        self.total_value = 0.0  # Суммарная оценка (Value)
        
        # Априорные вероятности из классификатора (Policy/Prior)
        self.priors: Dict[chess.Move, float] = {}

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def get_ucb_score(self, move: chess.Move, cpuct: float = 1.5) -> float:
        """Расчет метрики UCB (Upper Confidence Bound) для выбора хода."""
        child = self.children.get(move)
        p_score = self.priors.get(move, 0.0)
        
        if child:
            q_score = child.value
            u_score = cpuct * p_score * math.sqrt(self.visit_count) / (1 + child.visit_count)
            return q_score + u_score
        else:
            # Если узел еще не посещался, у него высокий потенциал за счет Prior
            return cpuct * p_score * math.sqrt(self.visit_count + 1e-6)


class MoveClassifierMCTS:
    """Движок MCTS, интегрированный с вашей нейросетью."""
    def __init__(self, classifier: MoveClassifier, cpuct: float = 1.5):
        self.classifier = classifier
        self.cpuct = cpuct
        
        # Веса для типов ходов (превращаем классификацию в "Policy" вероятности)
        self.class_priority = {
            "Best": 1.0,
            "Excellent": 0.8,
            "Good": 0.5,
            "Inaccuracy": 0.2,
            "Mistake": 0.05,
            "Blunder": 0.001
        }

    def _evaluate_node_priors(self, node: MCTSNode):
        """Оценивает все легальные ходы из узла с помощью нейросети (заполнение Priors)."""
        legal_moves = list(node.board.legal_moves)
        if not legal_moves:
            return

        # Нам нужно превратить классы нейросети в распределение вероятностей (Priors)
        raw_scores = []
        
        # Делаем предсказания (в идеале тут сделать батчинг, но для начала — быстрый цикл)
        for move in legal_moves:
            move_san = node.board.san(move)
            
            # Виртуальный ход
            node.board.push(move)
            fen_after = node.board.fen()
            node.board.pop()
            
            # Инференс нашей сети
            res = self.classifier.classify_move(node.board.fen(), move_san)
            
            # Берем базовый вес класса и умножаем на уверенность сети
            score = self.class_priority.get(res.classification, 0.1) * res.confidence
            raw_scores.append(score)
            
        # Нормализуем через Softmax (чтобы сумма вероятностей была равна 1.0)
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        probabilities = exp_scores / exp_scores.sum()
        
        for move, prob in zip(legal_moves, probabilities):
            node.priors[move] = prob

    def search(self, initial_board: chess.Board, num_simulations: int = 100) -> chess.Move:
        """Главный цикл поиска лучшего хода."""
        if initial_board.is_game_over():
            return None
        
        root = MCTSNode(board=initial_board.copy())
        self._evaluate_node_priors(root)
        root.is_expanded = True
        
        for _ in range(num_simulations):
            node = root
            
            # 1. СЕЛЕКЦИЯ (Selection) - Спускаемся до листа по UCB
            while node.is_expanded and node.board.legal_moves:
                legal_moves = list(node.board.legal_moves)
                
                # Если ходы еще не оценены сетью для этого узла, оцениваем
                if not node.priors:
                    self._evaluate_node_priors(node)
                    
                best_move = max(legal_moves, key=lambda m: node.get_ucb_score(m, self.cpuct))
                
                if best_move not in node.children:
                    # Создаем новый узел-листок
                    next_board = node.board.copy()
                    next_board.push(best_move)
                    node.children[best_move] = MCTSNode(board=next_board, parent=node, move=best_move)
                
                node = node.children[best_move]
            
            # 2. ЭКСПАНСИЯ (Expansion) и СИМУЛЯЦИЯ (Simulation)
            # Так как у нас пока нет "Value" головы (которая говорит +0.5 или -0.7), 
            # мы делаем классический rollout (случайную доигровку) или берем оценку материала.
            value = self._rollout(node.board)
            
            # Если это был ход черных, инвертируем оценку для бэкапа
            if node.board.turn != initial_board.turn:
                value = -value
                
            # 3. БЭКАП (Backpropagation) - Идем вверх и обновляем статистику
            while node is not None:
                node.visit_count += 1
                node.total_value += value
                value = -value  # Меняем знак на каждом уровне дерева (минимаксная логика)
                node = node.parent
                
        # Выбираем ход, который посетили чаще всего (самый надежный)
        best_move = max(root.children.keys(), key=lambda m: root.children[m].visit_count)
        return best_move

    def _rollout(self, board: chess.Board) -> float:
        """Быстрая эвристическая оценка позиции (материальный баланс).
        Возвращает значение от -1.0 (выигрыш черных) до 1.0 (выигрыш белых)."""
        if board.is_game_over():
            result = board.result()
            if result == "1-0": return 1.0
            if result == "0-1": return -1.0
            return 0.0

        # Считаем материал на доске
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
                    
        # Нормализуем разницу в интервал [-1, 1]
        diff = white_material - black_material
        # Предел в 15 пешек преимущества (все что выше — абсолютная победа)
        value = math.tanh(diff / 15.0) 
        return value
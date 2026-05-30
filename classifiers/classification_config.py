"""Classification threshold configuration for move evaluation."""

from dataclasses import dataclass
from typing import List

@dataclass
class ClassificationThreshold:
    """Threshold configuration for move classification."""
    name: str
    min_evaluation: float
    max_evaluation: float

# Новые, адекватные шахматной логике пороги (в пешках):
# Best: потеря от 0.0 до 0.02 (идеальный или почти идеальный ход)
# Excellent: потеря от 0.02 до 0.15 (очень хороший ход)
# Good: потеря от 0.15 до 0.40 (нормальный игровой ход)
# Inaccuracy: потеря от 0.40 до 0.80 (сомнительный ход / неточность)
# Mistake: потеря от 0.80 до 1.50 (явная ошибка)
# Blunder: потеря выше 1.50 (зевок фигуры / мата)
THRESHOLDS: List[ClassificationThreshold] = [
    ClassificationThreshold(name="Best", min_evaluation=0.0, max_evaluation=0.02),
    ClassificationThreshold(name="Excellent", min_evaluation=0.02, max_evaluation=0.15),
    ClassificationThreshold(name="Good", min_evaluation=0.15, max_evaluation=0.40),
    ClassificationThreshold(name="Inaccuracy", min_evaluation=0.40, max_evaluation=0.80),
    ClassificationThreshold(name="Mistake", min_evaluation=0.80, max_evaluation=1.50),
    ClassificationThreshold(name="Blunder", min_evaluation=1.50, max_evaluation=100.0),
]

NEGATIVE_THRESHOLD = -0.02  # Оставляем для отсечения ходов, которые улучшили оценку

CLASS_NAMES = ["Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"]
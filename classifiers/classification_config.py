"""Classification threshold configuration for move evaluation."""

from dataclasses import dataclass
from typing import List

@dataclass
class ClassificationThreshold:
    """Threshold configuration for move classification."""
    name: str
    min_evaluation: float
    max_evaluation: float

THRESHOLDS: List[ClassificationThreshold] = [
    ClassificationThreshold(name="Blunder", min_evaluation=-100.0, max_evaluation=-0.80),
    ClassificationThreshold(name="Mistake", min_evaluation=-0.80, max_evaluation=-0.40),
    ClassificationThreshold(name="Inaccuracy", min_evaluation=-0.40, max_evaluation=-0.15),
    ClassificationThreshold(name="Good", min_evaluation=-0.15, max_evaluation=0.15),
    ClassificationThreshold(name="Excellent", min_evaluation=0.15, max_evaluation=0.80),
    ClassificationThreshold(name="Best", min_evaluation=0.80, max_evaluation=100.0),
]

NEGATIVE_THRESHOLD = -0.02

CLASS_NAMES = ["Blunder", "Mistake", "Inaccuracy", "Good", "Excellent", "Best"]
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
    ClassificationThreshold(name="Best", min_evaluation=0.0, max_evaluation=0.02),
    ClassificationThreshold(name="Excellent", min_evaluation=0.02, max_evaluation=0.15),
    ClassificationThreshold(name="Good", min_evaluation=0.15, max_evaluation=0.40),
    ClassificationThreshold(name="Inaccuracy", min_evaluation=0.40, max_evaluation=0.80),
    ClassificationThreshold(name="Mistake", min_evaluation=0.80, max_evaluation=1.50),
    ClassificationThreshold(name="Blunder", min_evaluation=1.50, max_evaluation=100.0),
]

NEGATIVE_THRESHOLD = -0.02 

CLASS_NAMES = ["Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"]
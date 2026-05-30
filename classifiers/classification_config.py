"""Classification threshold configuration for move evaluation."""

from dataclasses import dataclass
from typing import List

@dataclass
class ClassificationThreshold:
    """Threshold configuration for move classification."""
    name: str
    min_evaluation: float
    max_evaluation: float

# Best: Loss from 0.0 to 0.02 (perfect or nearly perfect move)
# Excellent: Loss from 0.02 to 0.15 (very good move)
# Good: Loss from 0.15 to 0.40 (normal game move)
# Inaccuracy: Loss from 0.40 to 0.80 (questionable move/inaccuracy)
# Mistake: Loss from 0.80 to 1.50 (obvious error)
# Blunder: Loss above 1.50 (blunder of a piece/mate)
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
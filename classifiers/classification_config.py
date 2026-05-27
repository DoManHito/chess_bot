"""Classification threshold configuration for move evaluation."""

from dataclasses import dataclass
from typing import List


@dataclass
class ClassificationThreshold:
    """Threshold configuration for move classification."""
    name: str
    min_evaluation: float
    max_evaluation: float


# Define classification thresholds
# Best: 0.00 - 0.00 (perfect move)
# Excellent: 0.00 - 0.02
# Good: 0.02 - 0.05
# Inaccuracy: 0.05 - 0.10
# Mistake: 0.10 - 0.20
# Blunder: 0.20 - 1.00
THRESHOLDS: List[ClassificationThreshold] = [
    ClassificationThreshold(name="Best", min_evaluation=0.0, max_evaluation=0.0),
    ClassificationThreshold(name="Excellent", min_evaluation=0.0, max_evaluation=0.02),
    ClassificationThreshold(name="Good", min_evaluation=0.02, max_evaluation=0.05),
    ClassificationThreshold(name="Inaccuracy", min_evaluation=0.05, max_evaluation=0.10),
    ClassificationThreshold(name="Mistake", min_evaluation=0.10, max_evaluation=0.20),
    ClassificationThreshold(name="Blunder", min_evaluation=0.20, max_evaluation=1.0),
]

# Handle negative evaluations (moves worse than best)
NEGATIVE_THRESHOLD = -0.02  # Any negative evaluation is worse than best

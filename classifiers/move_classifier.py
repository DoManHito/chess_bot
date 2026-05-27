"""Move classification logic based on evaluation thresholds."""

from dataclasses import dataclass
from typing import List, Optional

from .classification_config import THRESHOLDS, NEGATIVE_THRESHOLD


@dataclass
class MoveData:
    """Data structure for a single move to classify."""
    evaluation: float
    turn_num: int
    turn_label: str


@dataclass
class MoveClassificationResult:
    """Result of move classification with confidence score."""
    evaluation: float
    turn_num: int
    turn_label: str
    classification: str
    confidence: float

    def __str__(self) -> str:
        """Return string representation of classification result."""
        return (
            f"Turn {self.turn_num} ({self.turn_label}): "
            f"{self.classification} (eval={self.evaluation:.3f}, conf={self.confidence:.2f})"
        )


class MoveClassifier:
    """Classifier for move evaluations."""

    def __init__(self) -> None:
        """Initialize the move classifier."""
        self._thresholds = THRESHOLDS

    def classify_move(
        self,
        evaluation: float,
        turn_num: int,
        turn_label: str
    ) -> MoveClassificationResult:
        """Classify a single move based on evaluation score.

        Args:
            evaluation: The evaluation score for the move.
            turn_num: The turn number.
            turn_label: The label for the turn.

        Returns:
            MoveClassificationResult with classification and confidence.
        """
        # Handle missing evaluation data
        if evaluation is None:
            return MoveClassificationResult(
                evaluation=0.0,
                turn_num=turn_num,
                turn_label=turn_label,
                classification="Unknown",
                confidence=0.0
            )

        # Handle negative evaluations (moves worse than best)
        if evaluation < 0:
            # Negative evaluations are worse than best
            # Use absolute value for classification, but mark as negative
            abs_eval = abs(evaluation)
            classification = self._classify_by_threshold(abs_eval)
            confidence = min(1.0, abs(evaluation) / 0.2)  # Scale confidence for negative evals
            return MoveClassificationResult(
                evaluation=evaluation,
                turn_num=turn_num,
                turn_label=turn_label,
                classification=classification,
                confidence=confidence
            )

        # Classify positive evaluations using thresholds
        classification = self._classify_by_threshold(evaluation)
        confidence = min(1.0, evaluation / 0.2)  # Scale confidence

        return MoveClassificationResult(
            evaluation=evaluation,
            turn_num=turn_num,
            turn_label=turn_label,
            classification=classification,
            confidence=confidence
        )

    def _classify_by_threshold(self, evaluation: float) -> str:
        """Classify an evaluation score using the threshold list.

        Args:
            evaluation: The evaluation score to classify.

        Returns:
            The classification name.
        """
        for threshold in self._thresholds:
            if threshold.min_evaluation <= evaluation <= threshold.max_evaluation:
                return threshold.name

        # Fallback for edge cases
        if evaluation > 1.0:
            return "Blunder"
        return "Unknown"

    def classify_moves(
        self,
        moves: List[MoveData]
    ) -> List[MoveClassificationResult]:
        """Batch classify multiple moves.

        Args:
            moves: List of MoveData objects to classify.

        Returns:
            List of MoveClassificationResult objects.
        """
        if not moves:
            return []

        return [self.classify_move(move.evaluation, move.turn_num, move.turn_label) for move in moves]

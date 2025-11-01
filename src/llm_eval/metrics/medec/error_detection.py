"""Error detection metric for MEDEC evaluation."""

from typing import List, Dict, Any


from ..base import BaseMetric
from ...data.medec.samples import MEDECSample
from ...parsers.medec.error_detection import ErrorDetectionParser


class ErrorDetectionMetric(BaseMetric):
    """Metric for evaluating error detection performance."""

    def __init__(
        self, metric_name: str = "error_detection", parser: ErrorDetectionParser = None
    ):
        if parser is None:
            parser = ErrorDetectionParser()
        super().__init__(metric_name, parser)

    def _get_reference_value(self, sample: MEDECSample) -> int:
        """Get reference error flag from MEDEC sample."""
        if not isinstance(sample, MEDECSample):
            raise ValueError(f"Expected MEDECSample, got {type(sample)}")
        return sample.error_flag

    def compute_metric(
        self, predictions: List[int], references: List[int]
    ) -> Dict[str, Any]:
        """Compute error detection metrics with both aggregated and per-item results."""
        if len(predictions) != len(references):
            raise ValueError(
                f"Predictions and references must have same length: {len(predictions)} vs {len(references)}"
            )

        if not predictions:
            return {
                "aggregated": {
                    f"{self.metric_name}_accuracy": 0.0,
                    f"{self.metric_name}_precision": 0.0,
                    f"{self.metric_name}_recall": 0.0,
                    f"{self.metric_name}_f1": 0.0,
                },
                "per_item": [],
            }

        # Calculate confusion matrix - treat -1 as incorrect prediction
        tp = sum(1 for p, r in zip(predictions, references) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, references) if p == 1 and r == 0)
        tn = sum(1 for p, r in zip(predictions, references) if p == 0 and r == 0)
        fn = sum(
            1
            for p, r in zip(predictions, references)
            if (p == 0 and r == 1) or (p == -1 and r == 1)
        )
        # Parse failures (-1) are treated as false negatives when reference is 1, or as false positives when reference is 0
        fp += sum(1 for p, r in zip(predictions, references) if p == -1 and r == 0)

        total = len(predictions)

        # Calculate aggregated metrics
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        aggregated_scores = {
            f"{self.metric_name}_accuracy": accuracy,
            f"{self.metric_name}_precision": precision,
            f"{self.metric_name}_recall": recall,
            f"{self.metric_name}_f1": f1,
            f"{self.metric_name}_tp": tp,
            f"{self.metric_name}_fp": fp,
            f"{self.metric_name}_tn": tn,
            f"{self.metric_name}_fn": fn,
        }

        # Calculate per-item scores
        per_item_scores = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # Parse failures (-1) are always incorrect
            is_correct = pred == ref if pred != -1 else False
            is_tp = pred == 1 and ref == 1
            is_fp = pred == 1 and ref == 0 or (pred == -1 and ref == 0)
            is_tn = pred == 0 and ref == 0
            is_fn = pred == 0 and ref == 1 or (pred == -1 and ref == 1)

            per_item_scores.append(
                {
                    "sample_index": i,
                    "prediction": pred,
                    "reference": ref,
                    "is_correct": is_correct,
                    "is_tp": is_tp,
                    "is_fp": is_fp,
                    "is_tn": is_tn,
                    "is_fn": is_fn,
                }
            )

        return {"aggregated": aggregated_scores, "per_item": per_item_scores}

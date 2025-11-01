"""Sentence extraction metric for MEDEC evaluation."""

from typing import List, Dict, Any


from ..base import BaseMetric
from ...data.medec.samples import MEDECSample
from ...parsers.medec.sentence_extraction import SentenceExtractionParser


class SentenceExtractionMetric(BaseMetric):
    """Metric for evaluating sentence extraction performance."""

    def __init__(
        self,
        metric_name: str = "sentence_extraction",
        parser: SentenceExtractionParser = None,
    ):
        if parser is None:
            parser = SentenceExtractionParser()
        super().__init__(metric_name, parser)

    def _get_reference_value(self, sample: MEDECSample) -> int:
        """Get reference sentence ID from MEDEC sample."""
        if not isinstance(sample, MEDECSample):
            raise ValueError(f"Expected MEDECSample, got {type(sample)}")

        # error_sentence_id is already an integer (or None)
        if sample.error_sentence_id is not None:
            return sample.error_sentence_id
        else:
            return 0  # No error sentence

    def compute_metric(
        self, predictions: List[int], references: List[int]
    ) -> Dict[str, Any]:
        """Compute sentence extraction metrics with both aggregated and per-item results."""
        if len(predictions) != len(references):
            raise ValueError(
                f"Predictions and references must have same length: {len(predictions)} vs {len(references)}"
            )

        if not predictions:
            return {
                "aggregated": {
                    f"{self.metric_name}_accuracy": 0.0,
                    f"{self.metric_name}_exact_match": 0.0,
                },
                "per_item": [],
            }

        # Calculate exact match accuracy - parse failures (-1) are always incorrect
        exact_matches = sum(
            1 for p, r in zip(predictions, references) if p == r and p != -1
        )
        exact_match_rate = exact_matches / len(predictions)

        # Calculate accuracy (considering only samples with errors)
        error_samples = [(p, r) for p, r in zip(predictions, references) if r > 0]

        if error_samples:
            # Parse failures (-1) in error samples are always incorrect
            error_matches = sum(1 for p, r in error_samples if p == r and p != -1)
            error_accuracy = error_matches / len(error_samples)
        else:
            error_accuracy = 0.0

        aggregated_scores = {
            f"{self.metric_name}_accuracy": exact_match_rate,
            f"{self.metric_name}_exact_match": exact_match_rate,
            f"{self.metric_name}_error_accuracy": error_accuracy,
            f"{self.metric_name}_total_samples": len(predictions),
            f"{self.metric_name}_error_samples": len(error_samples),
            f"{self.metric_name}_exact_matches": exact_matches,
        }

        # Calculate per-item scores
        per_item_scores = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # Parse failures (-1) are always incorrect
            is_exact_match = pred == ref if pred != -1 else False
            has_error = ref > 0
            is_correct_on_error = has_error and (pred == ref) and pred != -1

            per_item_scores.append(
                {
                    "sample_index": i,
                    "prediction": pred,
                    "reference": ref,
                    "is_exact_match": is_exact_match,
                    "has_error": has_error,
                    "is_correct_on_error": is_correct_on_error,
                }
            )

        return {"aggregated": aggregated_scores, "per_item": per_item_scores}

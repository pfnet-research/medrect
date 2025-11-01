"""Metrics for MEDEC error type classification evaluation."""

from typing import List, Dict, Any
from collections import Counter
from loguru import logger

from ..base import BaseMetric
from ...data.medec.samples import MEDECSample
from ...parsers.medec_error_type.error_type_parser import ErrorTypeParser


class ErrorTypeMetrics(BaseMetric):
    """Metric for evaluating MEDEC error type classification results."""

    def __init__(
        self, metric_name: str = "error_type_metrics", parser: ErrorTypeParser = None
    ):
        if parser is None:
            parser = ErrorTypeParser()
        super().__init__(metric_name, parser)

    def _get_reference_value(self, sample: MEDECSample) -> Dict[str, Any]:
        """Get reference value for error type classification."""
        if not isinstance(sample, MEDECSample):
            raise ValueError(f"Expected MEDECSample, got {type(sample)}")

        return {
            "sample_id": sample.sample_id,
            "original_error_type": sample.error_type,
            "error_flag": sample.error_flag,
            "has_error": sample.has_error(),
        }

    def compute_metric(
        self, predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute error type classification metrics."""
        if len(predictions) != len(references):
            raise ValueError(
                f"Predictions and references must have same length: {len(predictions)} vs {len(references)}"
            )

        if not predictions:
            return {
                "aggregated": {f"{self.metric_name}_total_samples": 0.0},
                "per_item": [],
            }

        total_samples = len(predictions)
        parse_errors = sum(1 for pred in predictions if pred.get("parse_error", False))
        valid_predictions = [
            pred for pred in predictions if not pred.get("parse_error", False)
        ]
        valid_references = [
            ref
            for i, ref in enumerate(references)
            if not predictions[i].get("parse_error", False)
        ]
        valid_count = len(valid_predictions)

        if valid_count == 0:
            logger.warning("No valid predictions found for error type metrics")
            aggregated_scores = {
                f"{self.metric_name}_total_samples": float(total_samples),
                f"{self.metric_name}_parse_errors": float(parse_errors),
                f"{self.metric_name}_valid_samples": 0.0,
                f"{self.metric_name}_parse_error_rate": float(parse_errors)
                / total_samples
                if total_samples > 0
                else 0.0,
            }
            return {"aggregated": aggregated_scores, "per_item": []}

        # Collect error type predictions and references
        predicted_error_types = [pred["error_type"] for pred in valid_predictions]
        reference_error_flags = [ref["error_flag"] for ref in valid_references]

        # Count error type distributions
        error_type_distribution = Counter(predicted_error_types)

        # Count samples by error flag
        correct_samples = sum(1 for flag in reference_error_flags if flag == 0)
        error_samples = sum(1 for flag in reference_error_flags if flag == 1)

        # Count "none" predictions for correct samples vs error samples
        none_predictions = sum(
            1 for pred_type in predicted_error_types if pred_type == "none"
        )
        none_for_correct = sum(
            1
            for pred, ref in zip(valid_predictions, valid_references)
            if pred["error_type"] == "none" and ref["error_flag"] == 0
        )
        none_for_error = sum(
            1
            for pred, ref in zip(valid_predictions, valid_references)
            if pred["error_type"] == "none" and ref["error_flag"] == 1
        )

        # Count non-none predictions for error samples
        classified_errors = sum(
            1
            for pred, ref in zip(valid_predictions, valid_references)
            if pred["error_type"] != "none" and ref["error_flag"] == 1
        )

        # Count confidence levels
        confidence_distribution = Counter(
            pred["confidence"] for pred in valid_predictions
        )

        # Calculate accuracy of none vs non-none classification
        correct_none_classification = none_for_correct + classified_errors
        classification_accuracy = (
            correct_none_classification / valid_count if valid_count > 0 else 0.0
        )

        # Calculate error type coverage (non-none types assigned to error samples)
        error_type_coverage = (
            classified_errors / error_samples if error_samples > 0 else 0.0
        )

        # Calculate false positive rate (non-none assigned to correct samples)
        false_positive_rate = (
            (error_samples - classified_errors) / correct_samples
            if correct_samples > 0
            else 0.0
        )

        # Prepare aggregated scores
        aggregated_scores = {
            # Overall statistics
            f"{self.metric_name}_total_samples": float(total_samples),
            f"{self.metric_name}_parse_errors": float(parse_errors),
            f"{self.metric_name}_valid_samples": float(valid_count),
            f"{self.metric_name}_parse_error_rate": float(parse_errors) / total_samples
            if total_samples > 0
            else 0.0,
            # Sample distribution
            f"{self.metric_name}_correct_samples": float(correct_samples),
            f"{self.metric_name}_error_samples": float(error_samples),
            # Classification accuracy
            f"{self.metric_name}_classification_accuracy": classification_accuracy,
            f"{self.metric_name}_error_type_coverage": error_type_coverage,
            f"{self.metric_name}_false_positive_rate": false_positive_rate,
            # Prediction counts
            f"{self.metric_name}_none_predictions": float(none_predictions),
            f"{self.metric_name}_none_for_correct": float(none_for_correct),
            f"{self.metric_name}_none_for_error": float(none_for_error),
            f"{self.metric_name}_classified_errors": float(classified_errors),
            # Confidence distribution
            f"{self.metric_name}_high_confidence": float(
                confidence_distribution.get("high", 0)
            ),
            f"{self.metric_name}_medium_confidence": float(
                confidence_distribution.get("medium", 0)
            ),
            f"{self.metric_name}_low_confidence": float(
                confidence_distribution.get("low", 0)
            ),
        }

        # Add error type distribution
        for error_type, count in error_type_distribution.items():
            aggregated_scores[f"{self.metric_name}_{error_type}_count"] = float(count)
            aggregated_scores[f"{self.metric_name}_{error_type}_rate"] = (
                float(count) / valid_count if valid_count > 0 else 0.0
            )

        # Prepare per-item scores
        per_item_scores = []
        for i, (pred, ref) in enumerate(zip(valid_predictions, valid_references)):
            item_score = {
                "sample_id": ref["sample_id"],
                "predicted_error_type": pred["error_type"],
                "confidence": pred["confidence"],
                "original_error_type": ref.get("original_error_type", "unknown"),
                "error_flag": ref["error_flag"],
                "classification_correct": (
                    pred["error_type"] == "none" and ref["error_flag"] == 0
                )
                or (pred["error_type"] != "none" and ref["error_flag"] == 1),
                "explanation": pred.get("explanation", pred.get("reasoning", "")),
            }
            per_item_scores.append(item_score)

        return {"aggregated": aggregated_scores, "per_item": per_item_scores}

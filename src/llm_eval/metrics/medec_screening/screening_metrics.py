"""Screening metrics for MEDEC evaluation."""

from typing import List, Dict, Any
from collections import Counter

from loguru import logger

from ..base import BaseMetric
from ...data.medec.samples import MEDECSample
from ...parsers.medec_screening.screening_judge import ScreeningJudgeParser


class ScreeningMetrics(BaseMetric):
    """Metric for evaluating MEDEC quality screening results."""

    def __init__(
        self,
        metric_name: str = "screening_metrics",
        parser: ScreeningJudgeParser = None,
    ):
        if parser is None:
            parser = ScreeningJudgeParser()
        super().__init__(metric_name, parser)

    def _get_reference_value(self, sample: MEDECSample) -> Dict[str, Any]:
        """Get reference value - for screening, we don't have ground truth, so return sample info."""
        if not isinstance(sample, MEDECSample):
            raise ValueError(f"Expected MEDECSample, got {type(sample)}")

        return {
            "sample_id": sample.sample_id,
            "error_flag": sample.error_flag,
            "has_metadata": hasattr(sample, "metadata") and sample.metadata is not None,
        }

    def compute_metric(
        self, predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute screening metrics with both aggregated and per-item results."""
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
        valid_count = len(valid_predictions)

        if valid_count == 0:
            logger.warning("No valid predictions found for screening metrics")
            aggregated_scores = {
                f"{self.metric_name}_total_samples": float(total_samples),
                f"{self.metric_name}_parse_errors": float(parse_errors),
                f"{self.metric_name}_valid_samples": 0.0,
                f"{self.metric_name}_overall_valid_rate": 0.0,
            }
        else:
            # Count individual issues (strict - will raise KeyError if field missing)
            ambiguous_error_count = sum(
                1 for pred in valid_predictions if pred["ambiguous_error"] > 0
            )
            multiple_errors_count = sum(
                1 for pred in valid_predictions if pred["multiple_errors"] > 0
            )
            numerical_error_count = sum(
                1 for pred in valid_predictions if pred["numerical_error"] > 0
            )

            # Count Japanese-specific criteria (may not exist for English templates)
            extra_elements_count = sum(
                1 for pred in valid_predictions if pred.get("extra_elements", 0) > 0
            )
            synthesis_consistency_error_count = sum(
                1
                for pred in valid_predictions
                if pred.get("synthesis_consistency_error", 0) > 0
            )

            # Count English-specific criteria (may not exist for Japanese templates)
            unrealistic_scenario_count = sum(
                1
                for pred in valid_predictions
                if pred.get("unrealistic_scenario", 0) > 0
            )
            inconsistent_context_count = sum(
                1
                for pred in valid_predictions
                if pred.get("inconsistent_context", 0) > 0
            )

            # Count samples with no issues (valid for benchmark)
            valid_for_benchmark = sum(
                1 for pred in valid_predictions if pred["is_valid"]
            )

            # Calculate issue distribution
            total_issues = sum(
                pred["total_issues"]
                for pred in valid_predictions
                if pred["total_issues"] >= 0
            )
            avg_issues_per_sample = total_issues / valid_count if valid_count > 0 else 0

            # Count samples by number of issues
            issue_distribution = Counter(
                pred["total_issues"]
                for pred in valid_predictions
                if pred["total_issues"] >= 0
            )

            # Calculate aggregated rates
            aggregated_scores = {
                # Overall statistics
                f"{self.metric_name}_total_samples": float(total_samples),
                f"{self.metric_name}_parse_errors": float(parse_errors),
                f"{self.metric_name}_valid_samples": float(valid_count),
                f"{self.metric_name}_parse_error_rate": float(parse_errors)
                / total_samples
                if total_samples > 0
                else 0.0,
                # Quality assessment results
                f"{self.metric_name}_valid_for_benchmark": float(valid_for_benchmark),
                f"{self.metric_name}_overall_valid_rate": float(valid_for_benchmark)
                / valid_count
                if valid_count > 0
                else 0.0,
                f"{self.metric_name}_avg_issues_per_sample": avg_issues_per_sample,
                f"{self.metric_name}_total_issues": float(total_issues),
                # Individual issue rates (Japanese criteria)
                f"{self.metric_name}_ambiguous_error_rate": float(ambiguous_error_count)
                / valid_count
                if valid_count > 0
                else 0.0,
                f"{self.metric_name}_extra_elements_rate": float(extra_elements_count)
                / valid_count
                if valid_count > 0
                else 0.0,
                f"{self.metric_name}_multiple_errors_rate": float(multiple_errors_count)
                / valid_count
                if valid_count > 0
                else 0.0,
                f"{self.metric_name}_numerical_error_rate": float(numerical_error_count)
                / valid_count
                if valid_count > 0
                else 0.0,
                f"{self.metric_name}_synthesis_consistency_error_rate": float(
                    synthesis_consistency_error_count
                )
                / valid_count
                if valid_count > 0
                else 0.0,
                # Individual issue rates (English criteria)
                f"{self.metric_name}_unrealistic_scenario_rate": float(
                    unrealistic_scenario_count
                )
                / valid_count
                if valid_count > 0
                else 0.0,
                f"{self.metric_name}_inconsistent_context_rate": float(
                    inconsistent_context_count
                )
                / valid_count
                if valid_count > 0
                else 0.0,
                # Issue counts (Japanese criteria)
                f"{self.metric_name}_ambiguous_error_count": float(
                    ambiguous_error_count
                ),
                f"{self.metric_name}_extra_elements_count": float(extra_elements_count),
                f"{self.metric_name}_multiple_errors_count": float(
                    multiple_errors_count
                ),
                f"{self.metric_name}_numerical_error_count": float(
                    numerical_error_count
                ),
                f"{self.metric_name}_synthesis_consistency_error_count": float(
                    synthesis_consistency_error_count
                ),
                # Issue counts (English criteria)
                f"{self.metric_name}_unrealistic_scenario_count": float(
                    unrealistic_scenario_count
                ),
                f"{self.metric_name}_inconsistent_context_count": float(
                    inconsistent_context_count
                ),
            }

            # Add issue distribution statistics
            for issue_count in range(6):  # 0-5 issues
                count = issue_distribution.get(issue_count, 0)
                aggregated_scores[
                    f"{self.metric_name}_samples_with_{issue_count}_issues"
                ] = float(count)
                aggregated_scores[
                    f"{self.metric_name}_samples_with_{issue_count}_issues_rate"
                ] = float(count) / valid_count if valid_count > 0 else 0.0

        # Calculate per-item scores
        per_item_scores = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # Extract screening results
            parse_error = pred.get("parse_error", False)

            item_score = {
                "sample_index": i,
                "sample_id": ref.get("sample_id", f"sample_{i}"),
                "parse_error": parse_error,
            }

            if not parse_error:
                # Add individual issue flags (strict for core fields, get() only for optional ones)
                item_score.update(
                    {
                        # Core criteria (must exist)
                        "ambiguous_error": pred["ambiguous_error"],
                        "multiple_errors": pred["multiple_errors"],
                        "numerical_error": pred["numerical_error"],
                        "total_issues": pred["total_issues"],
                        "is_valid": pred["is_valid"],
                        "explanation": pred["explanation"],
                        # Language-specific criteria (may not exist)
                        "extra_elements": pred.get("extra_elements", 0),
                        "synthesis_consistency_error": pred.get(
                            "synthesis_consistency_error", 0
                        ),
                        "unrealistic_scenario": pred.get("unrealistic_scenario", 0),
                        "inconsistent_context": pred.get("inconsistent_context", 0),
                        # Optional metadata
                        "detected_language": pred.get("detected_language", "unknown"),
                    }
                )

            per_item_scores.append(item_score)

        return {"aggregated": aggregated_scores, "per_item": per_item_scores}

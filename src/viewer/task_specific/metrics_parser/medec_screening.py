"""MEDEC Screening-specific metrics parser for quality assessment."""

from typing import Dict, List, Any

from .base import BaseMetricsParser


class MEDECScreeningMetricsParser(BaseMetricsParser):
    """MEDEC Screening-specific metrics parser for quality assessment."""

    def parse_summary_metrics(self, summary: Dict[str, Any]) -> Dict[str, float]:
        """Parse MEDEC Screening metrics from predictions.json."""
        if "metric_breakdowns" in summary:
            return self._parse_screening_metrics(summary["metric_breakdowns"])

        return {}

    def _parse_screening_metrics(
        self, metric_breakdowns: Dict[str, Any]
    ) -> Dict[str, float]:
        """Parse MEDEC Screening metrics from predictions.json format."""
        parsed = {}

        # Screening judge metrics
        if "screening_judge" in metric_breakdowns:
            sj = metric_breakdowns["screening_judge"]
            for key, value in sj.items():
                if isinstance(value, (int, float)):
                    parsed[key] = value

        return parsed

    def get_metric_display_names(self) -> Dict[str, str]:
        """Get human-readable names for MEDEC Screening metrics."""
        return {
            "screening_metrics_overall_valid_rate": "Overall Valid Rate",
            "screening_metrics_parse_error_rate": "Parse Error Rate",
            "screening_metrics_avg_issues_per_sample": "Avg Issues per Sample",
            # Core criteria (always present)
            "screening_metrics_ambiguous_error_rate": "Ambiguous Error Rate",
            "screening_metrics_multiple_errors_rate": "Multiple Errors Rate",
            "screening_metrics_numerical_error_rate": "Numerical Error Rate",
            # Japanese-specific criteria
            "screening_metrics_extra_elements_rate": "Extra Elements Rate",
            "screening_metrics_synthesis_consistency_error_rate": "Synthesis Consistency Error Rate",
            # English-specific criteria
            "screening_metrics_unrealistic_scenario_rate": "Unrealistic Scenario Rate",
            "screening_metrics_inconsistent_context_rate": "Inconsistent Context Rate",
            # Sample count metrics
            "screening_metrics_total_samples": "Total Samples",
            "screening_metrics_valid_samples": "Valid Samples",
            "screening_metrics_valid_for_benchmark": "Valid for Benchmark",
            "screening_metrics_total_issues": "Total Issues Found",
        }

    def get_primary_metrics(self) -> List[str]:
        """Get list of primary metrics to highlight."""
        return self._get_default_primary_metrics()

    def _get_default_primary_metrics(self) -> List[str]:
        """Get default primary MEDEC Screening metrics to highlight."""
        return [
            "screening_metrics_overall_valid_rate",
            "screening_metrics_parse_error_rate",
            "screening_metrics_avg_issues_per_sample",
        ]

    def get_overview_metrics(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get MEDEC Screening metrics to display in overview tab."""
        overview_metrics = []

        # Parse the metrics
        parsed_metrics = self.parse_summary_metrics(metrics)

        # Overall quality metrics
        if "screening_metrics_overall_valid_rate" in parsed_metrics:
            overview_metrics.append(
                {
                    "label": "Overall Valid Rate",
                    "value": f"{parsed_metrics['screening_metrics_overall_valid_rate']:.3f}",
                    "category": "quality",
                }
            )

        if "screening_metrics_parse_error_rate" in parsed_metrics:
            overview_metrics.append(
                {
                    "label": "Parse Error Rate",
                    "value": f"{parsed_metrics['screening_metrics_parse_error_rate']:.3f}",
                    "category": "quality",
                }
            )

        if "screening_metrics_avg_issues_per_sample" in parsed_metrics:
            overview_metrics.append(
                {
                    "label": "Avg Issues per Sample",
                    "value": f"{parsed_metrics['screening_metrics_avg_issues_per_sample']:.3f}",
                    "category": "quality",
                }
            )

        # Error type breakdown - show all error types (prioritize core criteria)
        error_rate_metrics = [
            # Core criteria (always present)
            ("screening_metrics_ambiguous_error_rate", "Ambiguous Error Rate"),
            ("screening_metrics_multiple_errors_rate", "Multiple Errors Rate"),
            ("screening_metrics_numerical_error_rate", "Numerical Error Rate"),
            # Japanese-specific criteria
            ("screening_metrics_extra_elements_rate", "Extra Elements Rate"),
            (
                "screening_metrics_synthesis_consistency_error_rate",
                "Synthesis Consistency Error Rate",
            ),
            # English-specific criteria
            (
                "screening_metrics_unrealistic_scenario_rate",
                "Unrealistic Scenario Rate",
            ),
            (
                "screening_metrics_inconsistent_context_rate",
                "Inconsistent Context Rate",
            ),
        ]

        # Add error type metrics that have non-zero values
        for metric_key, metric_label in error_rate_metrics:
            if metric_key in parsed_metrics and parsed_metrics[metric_key] > 0:
                overview_metrics.append(
                    {
                        "label": metric_label,
                        "value": f"{parsed_metrics[metric_key]:.3f}",
                        "category": "error_types",
                    }
                )

        return overview_metrics

    def calculate_total_samples(self, metrics: Dict[str, Any]) -> int:
        """Calculate total sample count from MEDEC Screening metrics."""
        if "metric_breakdowns" in metrics:
            metric_breakdowns = metrics["metric_breakdowns"]
            if "screening_judge" in metric_breakdowns:
                sj = metric_breakdowns["screening_judge"]
                return int(sj.get("screening_metrics_total_samples", 0))
        return 0

    def get_total_attempted_samples(self, metrics: Dict[str, Any]) -> int:
        """Get total attempted samples including parsing failures."""
        # For screening task, total samples = attempted samples
        return self.calculate_total_samples(metrics)

    def get_metrics_priority_order(self) -> List[str]:
        """Get MEDEC Screening-specific metric priority order for charts."""
        return [
            "Overall Valid Rate",
            "Parse Error Rate",
            "Avg Issues per Sample",
            "Synthesis Consistency Error Rate",
            "Multiple Errors Rate",
            "Extra Elements Rate",
            "Ambiguous Error Rate",
            "Numerical Error Rate",
        ]

    def get_overview_display_config(self) -> Dict[str, Any]:
        """Get MEDEC Screening-specific overview display configuration."""
        return {
            "max_metrics": 6,  # Show more metrics as they're all relevant for quality assessment
            "show_only_top_per_category": False,  # Show all metrics
            "include_sample_counts": True,  # Include sample counts
        }

    def get_comparison_chart_config(self) -> Dict[str, Any]:
        """Get MEDEC Screening-specific comparison chart configuration."""
        return {
            "max_charts": None,  # Show all available metrics
            "chart_height": 500,  # Standard chart height
            "columns_layout": "auto",  # Dynamic column layout
        }

    def get_sample_analysis_sort_options(self) -> List[Dict[str, str]]:
        """Get MEDEC Screening-specific sort options for sample analysis."""
        return [
            {"label": "Original Order (Default)", "key": "original_order"},
            {"label": "Sample ID (A-Z)", "key": "sample_id"},
            {"label": "Problem Validity (High to Low)", "key": "validity_desc"},
            {"label": "Problem Validity (Low to High)", "key": "validity_asc"},
            {"label": "Total Issues (High to Low)", "key": "total_issues_desc"},
            {"label": "Total Issues (Low to High)", "key": "total_issues_asc"},
        ]

    def get_metrics_by_category(
        self, parsed_metrics: Dict[str, float]
    ) -> Dict[str, List[str]]:
        """Get MEDEC Screening metrics grouped by category."""
        display_names = self.get_metric_display_names()
        categories = {}

        # Quality metrics
        quality_metrics = []
        quality_keys = [
            "screening_metrics_overall_valid_rate",
            "screening_metrics_parse_error_rate",
            "screening_metrics_avg_issues_per_sample",
        ]
        for key in quality_keys:
            if key in parsed_metrics:
                display_name = display_names.get(key, key)
                quality_metrics.append(display_name)
        if quality_metrics:
            categories["Quality Assessment"] = quality_metrics

        # Error type metrics (organized by core vs language-specific)
        error_type_metrics = []
        # Core error criteria (always present)
        core_error_keys = [
            "screening_metrics_ambiguous_error_rate",
            "screening_metrics_multiple_errors_rate",
            "screening_metrics_numerical_error_rate",
        ]
        # Language-specific error criteria
        language_specific_error_keys = [
            "screening_metrics_extra_elements_rate",
            "screening_metrics_synthesis_consistency_error_rate",
            "screening_metrics_unrealistic_scenario_rate",
            "screening_metrics_inconsistent_context_rate",
        ]
        # Combine all error keys, prioritizing core ones
        error_keys = core_error_keys + language_specific_error_keys
        for key in error_keys:
            if key in parsed_metrics and parsed_metrics[key] > 0:
                display_name = display_names.get(key, key)
                error_type_metrics.append(display_name)
        if error_type_metrics:
            categories["Error Types"] = error_type_metrics

        # Sample count metrics
        count_metrics = []
        count_keys = [
            "screening_metrics_total_samples",
            "screening_metrics_valid_samples",
            "screening_metrics_valid_for_benchmark",
            "screening_metrics_total_issues",
        ]
        for key in count_keys:
            if key in parsed_metrics:
                display_name = display_names.get(key, key)
                count_metrics.append(display_name)
        if count_metrics:
            categories["Sample Counts"] = count_metrics

        return categories

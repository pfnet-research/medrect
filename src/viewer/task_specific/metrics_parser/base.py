"""Base metrics parser for benchmark viewer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Any, Optional

if TYPE_CHECKING:
    import pandas as pd


class BaseMetricsParser(ABC):
    """Base class for task-specific metrics parsing."""

    def __init__(self, task_config: Optional[Any] = None):
        self.task_config = task_config

    @abstractmethod
    def parse_summary_metrics(self, summary: Dict[str, Any]) -> Dict[str, float]:
        """Parse metrics from summary.json into display format."""
        pass

    @abstractmethod
    def get_metric_display_names(self) -> Dict[str, str]:
        """Get human-readable names for metrics."""
        pass

    def get_primary_metrics(self) -> List[str]:
        """Get list of primary metrics to highlight."""
        if self.task_config and self.task_config.metrics:
            return [
                m["name"] for m in self.task_config.metrics[:3]
            ]  # First 3 as primary
        return self._get_default_primary_metrics()

    @abstractmethod
    def _get_default_primary_metrics(self) -> List[str]:
        """Get default primary metrics when no config available."""
        pass

    @abstractmethod
    def get_overview_metrics(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get metrics to display in overview tab.

        Returns list of dicts with keys:
        - label: Display label for metric
        - value: Formatted metric value
        - category: Optional category for grouping
        """
        pass

    @abstractmethod
    def calculate_total_samples(self, metrics: Dict[str, Any]) -> int:
        """Calculate total sample count from metrics.

        Args:
            metrics: Raw metrics dictionary from summary.json

        Returns:
            Total number of samples processed
        """
        pass

    def get_metrics_priority_order(self) -> List[str]:
        """Get prioritized list of metric display names for charts.

        Returns:
            List of metric display names in priority order for chart display.
            Empty list means use default alphabetical order.
        """
        return []  # Default: no specific priority order

    def get_overview_display_config(self) -> Dict[str, Any]:
        """Get display configuration for overview tab.

        Returns:
            Configuration dict with keys:
            - max_metrics: Maximum metrics to show (None for no limit)
            - show_only_top_per_category: Show only top metric per category
            - include_sample_counts: Whether to show sample count columns
        """
        return {
            "max_metrics": 3,  # Default: show up to 3 metrics
            "show_only_top_per_category": False,  # Default: show all metrics
            "include_sample_counts": True,  # Default: include sample counts
        }

    def get_comparison_chart_config(self) -> Dict[str, Any]:
        """Get chart configuration for comparison tab.

        Returns:
            Configuration dict with keys:
            - max_charts: Maximum charts to show (None for no limit)
            - chart_height: Height of individual charts in pixels
            - columns_layout: 'auto' for dynamic or specific number
        """
        return {
            "max_charts": 3,  # Default: show up to 3 charts
            "chart_height": 500,  # Default chart height
            "columns_layout": "auto",  # Default: dynamic layout
        }

    def get_sample_analysis_sort_options(self) -> List[Dict[str, str]]:
        """Get sort options for sample analysis tab based on overview metrics.

        Returns:
            List of sort options with 'label' and 'key' fields.
        """
        # Default implementation - override in task-specific parsers
        return [
            {"label": "Original Order (Default)", "key": "original_order"},
            {"label": "Sample ID (A-Z)", "key": "sample_id"},
        ]

    def get_metrics_by_category(
        self, parsed_metrics: Dict[str, float]
    ) -> Dict[str, List[str]]:
        """Get metrics grouped by category for comparison tab.

        Args:
            parsed_metrics: Dictionary of parsed metric values

        Returns:
            Dictionary mapping category names to lists of metric display names
        """
        # Default implementation - override in task-specific parsers
        display_names = self.get_metric_display_names()
        available_metrics = [
            display_names.get(key, key) for key in parsed_metrics.keys()
        ]
        return {"All Metrics": available_metrics}

    def render_category_plots(
        self, category_name: str, df: "pd.DataFrame", selected_models: List[str]
    ) -> None:
        """Render custom plots for specific categories.

        Base implementation does nothing. Override in task-specific parsers to add custom plots.

        Args:
            category_name: Name of the metric category
            df: DataFrame containing metrics for all models
            selected_models: List of selected model names
        """
        pass

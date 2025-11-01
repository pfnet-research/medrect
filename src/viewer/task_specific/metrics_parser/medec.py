"""MEDEC-specific metrics parser."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Any

if TYPE_CHECKING:
    import pandas as pd

from .base import BaseMetricsParser


class MEDECMetricsParser(BaseMetricsParser):
    """MEDEC-specific metrics parser."""

    def parse_summary_metrics(self, summary: Dict[str, Any]) -> Dict[str, float]:
        """Parse MEDEC metrics from predictions.json."""
        if "metric_breakdowns" in summary:
            return self._parse_predictions_metrics(summary["metric_breakdowns"])

        return {}

    def _parse_predictions_metrics(
        self, metric_breakdowns: Dict[str, Any]
    ) -> Dict[str, float]:
        """Parse MEDEC metrics from predictions.json format."""
        parsed = {}

        # Define exclusion patterns for different metric categories
        exclusion_patterns = {
            "error_detection": ["_tp", "_fp", "_tn", "_fn"],
            "sentence_extraction": ["_samples", "_matches"],
            "error_correction": ["_pairs", "_samples"],
            "error_correction_with_bert": ["_pairs", "_samples"],
            "error_correction_full": ["_samples", "_pairs"],
        }

        # Process all metric categories
        for category, exclusions in exclusion_patterns.items():
            if category in metric_breakdowns:
                metrics = metric_breakdowns[category]
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not any(
                        key.endswith(suffix) for suffix in exclusions
                    ):
                        parsed[key] = value

        return parsed

    @property
    def _metric_priorities(self) -> Dict[str, List[str]]:
        """Define metric priority orders by category."""
        return {
            "error_detection": [
                "error_detection_f1",
                "error_detection_precision",
                "error_detection_recall",
                "error_detection_accuracy",
            ],
            "sentence_extraction": [
                "sentence_extraction_error_accuracy",
                "sentence_extraction_accuracy",
            ],
            "error_correction": [
                "error_correction_average_score",
                "error_correction_composite_avg",
                "error_correction_rouge_1_f",
                "error_correction_bleurt",
                "error_correction_bertscore_f1",
            ],
        }

    def get_metric_display_names(self) -> Dict[str, str]:
        """Get human-readable names for MEDEC metrics."""
        return {
            "error_detection_accuracy": "ED Acc.",
            "error_detection_f1": "ED F1",
            "error_detection_precision": "ED Precision",
            "error_detection_recall": "ED Recall",
            "sentence_extraction_accuracy": "SE All-Cases Acc.",
            "sentence_extraction_exact_match": "SE Exact Match",
            "sentence_extraction_error_accuracy": "SE Error-Only Acc.",
            "error_correction_rouge_1_f": "EC ROUGE-1",
            "error_correction_bleurt": "EC BLEURT",
            "error_correction_bertscore_f1": "EC BERTScore",
            "error_correction_average_score": "EC Avg.",
            "error_correction_composite_avg": "EC Composite Avg.",
        }

    def get_primary_metrics(self) -> List[str]:
        """Get list of primary metrics to highlight."""
        # Override base class to return actual metric names
        return self._get_default_primary_metrics()

    def _get_default_primary_metrics(self) -> List[str]:
        """Get default primary MEDEC metrics to highlight."""
        metrics = []
        for category_metrics in self._metric_priorities.values():
            metrics.extend(category_metrics)
        return metrics

    def get_overview_metrics(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get MEDEC metrics to display in overview tab."""
        overview_metrics = []

        # Handle both new predictions.json format and legacy summary.json format
        parsed_metrics = self.parse_summary_metrics(metrics)

        # Error detection metrics
        if "error_detection_f1" in parsed_metrics:
            overview_metrics.append(
                {
                    "label": "Error Detection F1",
                    "value": f"{parsed_metrics['error_detection_f1']:.3f}",
                    "category": "error_detection",
                }
            )

        # Sentence extraction metrics
        if "sentence_extraction_error_accuracy" in parsed_metrics:
            overview_metrics.append(
                {
                    "label": "Sentence Extraction Acc",
                    "value": f"{parsed_metrics['sentence_extraction_error_accuracy']:.3f}",
                    "category": "sentence_extraction",
                }
            )

        # Error correction metrics - show only the top priority metric (for overview)
        display_names = self.get_metric_display_names()
        for metric_key in self._metric_priorities["error_correction"]:
            if metric_key in parsed_metrics:
                # Use full form for overview display
                if metric_key == "error_correction_average_score":
                    label = "Error Correction Avg."
                elif metric_key == "error_correction_composite_avg":
                    label = "Error Correction Composite Avg."
                elif metric_key == "error_correction_rouge_1_f":
                    label = "Error Correction ROUGE-1"
                elif metric_key == "error_correction_bleurt":
                    label = "Error Correction BLEURT"
                elif metric_key == "error_correction_bertscore_f1":
                    label = "Error Correction BERTScore"
                else:
                    label = display_names.get(metric_key, metric_key)

                overview_metrics.append(
                    {
                        "label": label,
                        "value": f"{parsed_metrics[metric_key]:.3f}",
                        "category": "error_correction",
                    }
                )
                break  # Only add the first (highest priority) available metric

        return overview_metrics

    def calculate_total_samples(self, metrics: Dict[str, Any]) -> int:
        """Calculate total sample count from MEDEC metrics."""
        if "metric_breakdowns" in metrics:
            metric_breakdowns = metrics["metric_breakdowns"]
            if "error_detection" in metric_breakdowns:
                ed = metric_breakdowns["error_detection"]
                return (
                    ed.get("error_detection_tp", 0)
                    + ed.get("error_detection_fp", 0)
                    + ed.get("error_detection_tn", 0)
                    + ed.get("error_detection_fn", 0)
                )
            # Fallback to sentence extraction total
            if "sentence_extraction" in metric_breakdowns:
                return metric_breakdowns["sentence_extraction"].get(
                    "sentence_extraction_total_samples", 0
                )
        return 0

    def get_total_attempted_samples(self, metrics: Dict[str, Any]) -> int:
        """Get total attempted samples including parsing failures."""
        if "metric_breakdowns" in metrics:
            metric_breakdowns = metrics["metric_breakdowns"]
            # Try to get from sentence_extraction first (includes all attempts)
            if "sentence_extraction" in metric_breakdowns:
                total = metric_breakdowns["sentence_extraction"].get(
                    "sentence_extraction_total_samples", 0
                )
                if total > 0:
                    return total

            # Fallback to error_correction total pairs
            if "error_correction" in metric_breakdowns:
                total = metric_breakdowns["error_correction"].get(
                    "error_correction_total_pairs", 0
                )
                if total > 0:
                    return total

        # Last resort: use processed samples
        return self.calculate_total_samples(metrics)

    def get_overview_display_config(self) -> Dict[str, Any]:
        """Get MEDEC-specific overview display configuration."""
        return {
            "max_metrics": 3,  # Error detection, sentence extraction, error correction
            "show_only_top_per_category": True,  # Show only top metric per category
            "include_sample_counts": True,  # Include processed/attempted sample counts
        }

    def get_comparison_chart_config(self) -> Dict[str, Any]:
        """Get MEDEC-specific comparison chart configuration."""
        return {
            "max_charts": None,  # Show all available metrics
            "chart_height": 500,  # Standard chart height
            "columns_layout": "auto",  # Dynamic column layout
        }

    def get_sample_analysis_sort_options(self) -> List[Dict[str, str]]:
        """Get MEDEC-specific sort options for sample analysis based on overview metrics."""
        return [
            {"label": "Original Order (Default)", "key": "original_order"},
            {"label": "Sample ID (A-Z)", "key": "sample_id"},
            {
                "label": "Error Detection F1 (High to Low)",
                "key": "error_detection_f1_desc",
            },
            {
                "label": "Error Detection F1 (Low to High)",
                "key": "error_detection_f1_asc",
            },
            {
                "label": "Sentence Extraction Acc (High to Low)",
                "key": "sentence_extraction_error_accuracy_desc",
            },
            {
                "label": "Sentence Extraction Acc (Low to High)",
                "key": "sentence_extraction_error_accuracy_asc",
            },
            {
                "label": "Error Correction Score (High to Low)",
                "key": "error_correction_top_metric_desc",
            },
            {
                "label": "Error Correction Score (Low to High)",
                "key": "error_correction_top_metric_asc",
            },
        ]

    def get_metrics_by_category(
        self, parsed_metrics: Dict[str, float]
    ) -> Dict[str, List[str]]:
        """Get MEDEC metrics grouped by category."""
        display_names = self.get_metric_display_names()
        categories = {}

        # Map category names to display names
        category_display_names = {
            "error_detection": "Error Detection",
            "sentence_extraction": "Sentence Extraction",
            "error_correction": "Error Correction",
        }

        # Process each category using priority order
        for category, metric_keys in self._metric_priorities.items():
            category_metrics = []
            for key in metric_keys:
                if key in parsed_metrics:
                    display_name = display_names.get(key, key)
                    category_metrics.append(display_name)

            if category_metrics:
                display_category_name = category_display_names.get(category, category)
                categories[display_category_name] = category_metrics

        return categories

    def render_category_plots(
        self, category_name: str, df: "pd.DataFrame", selected_models: List[str]
    ) -> None:
        """Render MEDEC-specific custom plots for categories."""
        if category_name == "Error Detection":
            self._render_precision_recall_plot(df, selected_models)

    def _render_precision_recall_plot(
        self, df: "pd.DataFrame", selected_models: List[str]
    ) -> None:
        """Render 2D precision-recall scatter plot for MEDEC error detection."""
        import streamlit as st
        import plotly.graph_objects as go

        precision_col = "ED Precision"
        recall_col = "ED Recall"

        # Check if both precision and recall columns exist
        if precision_col not in df.columns or recall_col not in df.columns:
            return

        # Create scatter plot
        fig = go.Figure()

        # Add scatter points for each model
        for _, row in df.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row[recall_col]],
                    y=[row[precision_col]],
                    mode="markers+text",
                    name=row["Model"],
                    text=[row["Model"]],
                    textposition="top center",
                    marker=dict(size=12),
                    showlegend=False,  # No legend needed - model names are shown on points
                    hovertemplate=f"<b>{row['Model']}</b><br>Precision: {row[precision_col]:.3f}<br>Recall: {row[recall_col]:.3f}<extra></extra>",
                )
            )

        # Update layout to match other charts
        fig.update_layout(
            title="Precision vs Recall",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=500,  # Match config chart_height
            hovermode="closest",
            showlegend=False,  # No legend needed
            xaxis=dict(
                range=[0, 1.05], showgrid=True, gridcolor="lightgray", automargin=True
            ),
            yaxis=dict(range=[0, 1.05], showgrid=True, gridcolor="lightgray"),
            plot_bgcolor="white",
            margin=dict(l=50, r=50, t=50, b=100),  # Match bottom margin from bar charts
        )

        # Add diagonal line for reference
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line=dict(color="gray", width=1, dash="dash"),
        )

        config = {"width": "stretch"}
        st.plotly_chart(fig, config=config)

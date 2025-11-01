"""Metrics comparison tab component for benchmark viewer."""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Any, Dict, List

from ..task_specific.data_loader.base import BaseDataLoader
from ..task_specific.metrics_parser.base import BaseMetricsParser
from ..task_specific.response_formatter.base import BaseResponseFormatter


class MetricsComparisonTab:
    """Metrics comparison tab for model performance analysis."""

    def __init__(
        self,
        loader: BaseDataLoader,
        parser: BaseMetricsParser,
        formatter: BaseResponseFormatter,
        base_path: Path,
    ):
        self.loader = loader
        self.parser = parser
        self.formatter = formatter
        self.base_path = base_path

    def render(
        self,
        dataset: Dict[str, Any],
        task: str,
        dataset_name: str,
        template_names: List[str],
        selected_models: List[str],
    ):
        """Render the metrics comparison tab content."""
        st.header("📈 Metrics")

        # Note: dataset parameter available for future use

        # Calculate total number of experiments
        total_experiments = len(selected_models) * len(template_names)
        if total_experiments < 2:
            st.warning(
                "Please select at least 2 experiments (model-template combinations) for comparison"
            )
            return

        # Prepare comparison data for all model-template combinations
        comparison_data = self._prepare_experiment_comparison_data(
            selected_models, template_names, task, dataset_name
        )

        if comparison_data:
            df = pd.DataFrame(comparison_data)

            # Display as table
            config = {"width": "stretch"}
            st.dataframe(df, **config)

            # Bar charts for each metric
            self._render_experiment_charts(
                df, selected_models, template_names, task, dataset_name
            )
        else:
            st.warning("No comparison data available")

    def _prepare_comparison_data(
        self,
        selected_models: List[str],
        task: str,
        dataset_name: str,
        template_name: str,
    ) -> List[Dict[str, Any]]:
        """Prepare comparison data for selected models."""
        comparison_data = []

        for model in selected_models:
            model_results = self.loader.load_model_results(
                task, dataset_name, model, template_name
            )
            if model_results and "summary" in model_results:
                parsed_metrics = self.parser.parse_summary_metrics(
                    model_results["summary"]
                )

                # Get primary metrics for display
                row_data = {"Model": model}
                display_names = self.parser.get_metric_display_names()

                for metric in self.parser.get_primary_metrics():
                    if metric in parsed_metrics:
                        display_name = display_names.get(metric, metric)
                        row_data[display_name] = parsed_metrics[metric]

                comparison_data.append(row_data)

        return comparison_data

    def _prepare_experiment_comparison_data(
        self,
        selected_models: List[str],
        template_names: List[str],
        task: str,
        dataset_name: str,
    ) -> List[Dict[str, Any]]:
        """Prepare comparison data for all model-template combinations."""
        comparison_data = []

        for model in selected_models:
            for template_name in template_names:
                model_results = self.loader.load_model_results(
                    task, dataset_name, model, template_name
                )
                if model_results and "summary" in model_results:
                    parsed_metrics = self.parser.parse_summary_metrics(
                        model_results["summary"]
                    )

                    # Create experiment name
                    experiment_name = f"{model} ({template_name})"

                    # Get primary metrics for display
                    row_data = {
                        "Experiment": experiment_name,
                        "Model": model,
                        "Template": template_name,
                    }
                    display_names = self.parser.get_metric_display_names()

                    for metric in self.parser.get_primary_metrics():
                        if metric in parsed_metrics:
                            display_name = display_names.get(metric, metric)
                            row_data[display_name] = parsed_metrics[metric]

                    comparison_data.append(row_data)

        return comparison_data

    def _render_experiment_charts(
        self,
        df: pd.DataFrame,
        selected_models: List[str],
        template_names: List[str],
        task: str = "",
        dataset_name: str = "",
    ):
        """Render bar charts for experiment comparison organized by category."""
        st.subheader("Experiment-wise Comparison")

        # Get the first experiment's metrics to determine available metrics
        if len(df) == 0:
            return

        # Get sample parsed metrics to categorize
        first_experiment = df.iloc[0]
        first_model = first_experiment["Model"]
        first_template = first_experiment["Template"]
        model_results = self.loader.load_model_results(
            task, dataset_name, first_model, first_template
        )

        if not model_results or "summary" not in model_results:
            return

        parsed_metrics = self.parser.parse_summary_metrics(model_results["summary"])
        categories = self.parser.get_metrics_by_category(parsed_metrics)
        config = self.parser.get_comparison_chart_config()
        chart_height = config.get("chart_height", 500)

        # Render metrics by category
        for category_name, category_metrics in categories.items():
            # Filter metrics that exist in the dataframe
            available_metrics = [
                metric for metric in category_metrics if metric in df.columns
            ]

            if not available_metrics:
                continue

            st.markdown(f"### {category_name}")

            # Render custom 2D plots if supported by parser
            self.parser.render_category_plots(
                category_name,
                df,
                [f"{m} ({t})" for m in selected_models for t in template_names],
            )

            # Always use 3 columns maximum with wrapping
            metrics_chunks = [
                available_metrics[i : i + 3]
                for i in range(0, len(available_metrics), 3)
            ]

            for chunk in metrics_chunks:
                cols = st.columns(3)  # Always create 3 columns

                for idx, metric in enumerate(chunk):
                    with cols[idx]:
                        # Sort by metric value (descending) for this specific metric
                        sorted_df = df.sort_values(by=metric, ascending=False)

                        # Create a more detailed chart with title
                        st.markdown(f"**{metric}**")

                        # Use plotly for better control over the chart
                        fig = px.bar(
                            sorted_df,
                            x="Experiment",
                            y=metric,
                            text=metric,
                            title=None,  # Title already shown above
                        )

                        # Format the chart using configuration
                        fig.update_traces(
                            texttemplate="%{text:.3f}", textposition="outside"
                        )
                        fig.update_layout(
                            showlegend=False,
                            xaxis_tickangle=-45,  # Rotate labels for better visibility
                            height=chart_height,
                            margin=dict(
                                b=100
                            ),  # More bottom margin for experiment names
                            xaxis=dict(
                                tickmode="linear",
                                automargin=True,  # Auto-adjust margin for long labels
                            ),
                            yaxis=dict(title="", rangemode="tozero"),
                        )

                        config = {"width": "stretch"}
                        st.plotly_chart(fig, config=config)

    def _render_metric_charts(
        self,
        df: pd.DataFrame,
        selected_models: List[str],
        task: str,
        dataset_name: str,
        template_name: str,
    ):
        """Render bar charts for metrics comparison organized by category."""
        st.subheader("Metric-wise Comparison")

        # Get the first model's metrics to determine available metrics
        if len(df) == 0:
            return

        # Get sample parsed metrics to categorize
        first_model = df.iloc[0]["Model"]
        model_results = self.loader.load_model_results(
            task, dataset_name, first_model, template_name
        )

        if not model_results or "summary" not in model_results:
            return

        parsed_metrics = self.parser.parse_summary_metrics(model_results["summary"])
        categories = self.parser.get_metrics_by_category(parsed_metrics)
        config = self.parser.get_comparison_chart_config()
        chart_height = config.get("chart_height", 500)

        # Render metrics by category
        for category_name, category_metrics in categories.items():
            # Filter metrics that exist in the dataframe
            available_metrics = [
                metric for metric in category_metrics if metric in df.columns
            ]

            if not available_metrics:
                continue

            st.markdown(f"### {category_name}")

            # Render custom 2D plots if supported by parser
            self.parser.render_category_plots(category_name, df, selected_models)

            # Always use 3 columns maximum with wrapping
            metrics_chunks = [
                available_metrics[i : i + 3]
                for i in range(0, len(available_metrics), 3)
            ]

            for chunk in metrics_chunks:
                cols = st.columns(3)  # Always create 3 columns

                for idx, metric in enumerate(chunk):
                    with cols[idx]:
                        # Sort by metric value (descending) for this specific metric
                        sorted_df = df.sort_values(by=metric, ascending=False)

                        # Create a more detailed chart with title
                        st.markdown(f"**{metric}**")

                        # Use plotly for better control over the chart
                        fig = px.bar(
                            sorted_df,
                            x="Model",
                            y=metric,
                            text=metric,
                            title=None,  # Title already shown above
                        )

                        # Format the chart using configuration
                        fig.update_traces(
                            texttemplate="%{text:.3f}", textposition="outside"
                        )
                        fig.update_layout(
                            showlegend=False,
                            xaxis_tickangle=-45,  # Rotate labels for better visibility
                            height=chart_height,
                            margin=dict(b=100),  # More bottom margin for model names
                            xaxis=dict(
                                tickmode="linear",
                                automargin=True,  # Auto-adjust margin for long labels
                            ),
                            yaxis=dict(title="", rangemode="tozero"),
                        )

                        config = {"width": "stretch"}
                        st.plotly_chart(fig, config=config)

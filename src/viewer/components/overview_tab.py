"""Overview tab component for benchmark viewer."""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

from ..task_specific.data_loader.base import BaseDataLoader
from ..task_specific.metrics_parser.base import BaseMetricsParser
from ..task_specific.response_formatter.base import BaseResponseFormatter


class OverviewTab:
    """Overview tab showing dataset and model summaries."""

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
        """Render the overview tab content."""
        st.header("📂 Dataset Overview")

        # Basic metrics
        self._render_basic_metrics(dataset, selected_models)

        # Distribution visualization
        self._render_distribution(dataset)

        # Model results summary
        if selected_models:
            self._render_model_summary(
                selected_models, task, dataset_name, template_names, dataset
            )

    def _render_basic_metrics(
        self, dataset: Dict[str, Any], selected_models: List[str]
    ):
        """Render basic dataset metrics."""
        # Get task-specific statistics
        stats = self.loader.get_dataset_statistics(dataset)

        # Sort by order and fill up to 4 columns
        stats.sort(key=lambda x: x.get("order", 999))

        # Add generic stats if needed
        while len(stats) < 4:
            if len(stats) == 1:
                stats.append(
                    {
                        "label": "Available Models",
                        "value": len(selected_models),
                        "order": 2,
                    }
                )
            elif len(stats) == 2:
                stats.append(
                    {
                        "label": "Selected Models",
                        "value": len(selected_models),
                        "order": 3,
                    }
                )
            else:
                stats.append({"label": "Metrics Available", "value": "3", "order": 4})

        # Display in columns
        cols = st.columns(4)
        for i, stat in enumerate(stats[:4]):
            with cols[i]:
                st.metric(stat["label"], stat["value"])

    def _render_distribution(self, dataset: Dict[str, Any]):
        """Render distribution visualization if available."""
        distribution = self.loader.get_dataset_distribution(dataset)
        if distribution:
            st.subheader(distribution["title"])
            df = pd.DataFrame(
                {
                    "Category": list(distribution["data"].keys()),
                    "Count": list(distribution["data"].values()),
                }
            )
            st.bar_chart(data=df, x="Category", y="Count")

    def _render_model_summary(
        self,
        selected_models: List[str],
        task: str,
        dataset_name: str,
        template_names: List[str],
        dataset: Dict[str, Any],
    ):
        """Render model results summary."""
        st.subheader("🤖 Selected Experiments Summary")

        # Get dataset sample count for comparison
        dataset_sample_count = len(dataset.get("samples", []))

        # Create all model-template combinations
        experiments = []
        for model in selected_models:
            for template_name in template_names:
                # Check if experiment has results
                model_results = self.loader.load_model_results(
                    task, dataset_name, model, template_name
                )
                has_results = bool(model_results and "summary" in model_results)

                experiment_name = f"{model} ({template_name})"
                experiments.append(
                    {
                        "name": experiment_name,
                        "model": model,
                        "template": template_name,
                        "results": model_results,
                        "has_results": has_results,
                    }
                )

        # Display each experiment
        for experiment in experiments:
            with st.expander(
                f"🧪 {experiment['name']}", expanded=experiment["has_results"]
            ):
                if experiment["has_results"]:
                    self._render_model_metrics(
                        experiment["results"]["summary"], dataset_sample_count
                    )
                else:
                    st.warning(f"No results found for {experiment['name']}")

    def _render_model_metrics(self, summary: Dict[str, Any], dataset_sample_count: int):
        """Render metrics for a single model."""
        # Pass entire summary to parser - it knows how to handle both summary.json and predictions.json formats
        overview_metrics = self.parser.get_overview_metrics(summary)

        # Calculate total samples using task-specific logic
        model_sample_count = self.parser.calculate_total_samples(summary)

        # Get total attempted samples if available
        total_attempted = None
        if hasattr(self.parser, "get_total_attempted_samples"):
            total_attempted = self.parser.get_total_attempted_samples(summary)

        # Get display configuration from task-specific parser
        config = self.parser.get_overview_display_config()
        num_metrics = len(overview_metrics)
        has_attempted = total_attempted is not None

        if num_metrics >= 3:
            # Use configuration to determine metrics to display
            max_metrics = config.get("max_metrics", 3)
            metrics_to_show = (
                min(num_metrics, max_metrics) if max_metrics else num_metrics
            )

            # Use configuration for sample count inclusion
            include_sample_counts = config.get("include_sample_counts", True)

            if include_sample_counts:
                if has_attempted:
                    # Show metrics + processed samples + attempted samples
                    total_cols = metrics_to_show + 2
                    cols = st.columns(total_cols)
                else:
                    # Show metrics + processed samples only
                    total_cols = metrics_to_show + 1
                    cols = st.columns(total_cols)
            else:
                # Show only metrics, no sample counts
                cols = st.columns(metrics_to_show)

            # Display metrics dynamically
            for i in range(metrics_to_show):
                with cols[i]:
                    if i < len(overview_metrics):
                        st.metric(
                            overview_metrics[i]["label"], overview_metrics[i]["value"]
                        )

            # Show sample counts only if configured to do so
            if include_sample_counts:
                # Show processed samples
                with cols[metrics_to_show]:
                    if (
                        model_sample_count != dataset_sample_count
                        and model_sample_count > 0
                    ):
                        delta_value = model_sample_count - dataset_sample_count
                        st.metric(
                            "Processed",
                            f"{model_sample_count}",
                            delta=f"{delta_value:+d} from dataset",
                            delta_color="normal",
                        )
                    else:
                        st.metric("Processed", model_sample_count)

                # Show attempted samples if different from processed
                if has_attempted:
                    with cols[metrics_to_show + 1]:
                        if (
                            total_attempted != dataset_sample_count
                            and total_attempted > 0
                        ):
                            delta_value = total_attempted - dataset_sample_count
                            st.metric(
                                "Total Attempts",
                                f"{total_attempted}",
                                delta=f"{delta_value:+d} from dataset",
                                delta_color="normal",
                            )
                        else:
                            st.metric("Total Attempts", total_attempted)
        else:
            # Fallback for fewer metrics
            if has_attempted:
                cols = st.columns(len(overview_metrics) + 2)
            else:
                cols = st.columns(len(overview_metrics) + 1)

            for i, metric in enumerate(overview_metrics):
                with cols[i]:
                    st.metric(metric["label"], metric["value"])

            # Show processed samples
            with cols[len(overview_metrics)]:
                if (
                    model_sample_count != dataset_sample_count
                    and model_sample_count > 0
                ):
                    delta_value = model_sample_count - dataset_sample_count
                    st.metric(
                        "Processed",
                        f"{model_sample_count}",
                        delta=f"{delta_value:+d} from dataset",
                        delta_color="normal",
                    )
                else:
                    st.metric("Processed", model_sample_count)

            # Show attempted samples if different
            if has_attempted:
                with cols[-1]:
                    if total_attempted != dataset_sample_count and total_attempted > 0:
                        delta_value = total_attempted - dataset_sample_count
                        st.metric(
                            "Total Attempts",
                            f"{total_attempted}",
                            delta=f"{delta_value:+d} from dataset",
                            delta_color="normal",
                        )
                    else:
                        st.metric("Total Attempts", total_attempted)

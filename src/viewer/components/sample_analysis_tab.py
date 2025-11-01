"""Sample analysis tab component for benchmark viewer."""

import streamlit as st
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..task_specific.data_loader.base import BaseDataLoader
from ..task_specific.metrics_parser.base import BaseMetricsParser
from ..task_specific.response_formatter.base import BaseResponseFormatter


class SampleAnalysisTab:
    """Sample-by-sample analysis tab."""

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
        """Render the sample analysis tab content."""
        st.header("🔍 Sample Analysis")

        # Sample selector
        samples = dataset.get("samples", [])
        if not samples:
            st.warning("No samples available for analysis")
            return

        # Get filter options from task-specific loader
        filter_options = self.loader.get_filter_options(samples)
        # Get sort options from task-specific parser (based on overview metrics)
        sort_options = self.parser.get_sample_analysis_sort_options()

        # Filter and sort controls
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            # Binary status filter (most left) - configurable by task
            if filter_options.get("binary_status_options"):
                binary_options = filter_options["binary_status_options"]
                selected_binary_status = st.multiselect(
                    filter_options.get("binary_status_label", "Show samples"),
                    options=binary_options,
                    default=binary_options,
                    key="binary_status_filter",
                )
            else:
                selected_binary_status = []

        with col2:
            # Category filter (e.g., error types)
            if filter_options.get("categories"):
                selected_categories = st.multiselect(
                    f"Filter by {filter_options.get('category_label', 'Category')}",
                    options=filter_options["categories"],
                    default=filter_options["categories"],
                    key="category_filter",
                )
            else:
                selected_categories = []

        with col3:
            # Sort options
            sort_labels = [opt["label"] for opt in sort_options]
            sort_option = st.selectbox(
                "Sort by", options=sort_labels, key="sort_option"
            )
            # Get the corresponding sort key
            sort_key = next(
                (opt["key"] for opt in sort_options if opt["label"] == sort_option),
                "original_order",
            )

        # Apply filters using task-specific logic
        filters = {
            "categories": selected_categories
            if filter_options.get("categories")
            else [],
            "binary_status": selected_binary_status
            if filter_options.get("binary_status_options")
            else [],
        }
        filtered_samples = self.loader.apply_filters(samples, filters)

        # Apply sorting using task-specific logic with error handling
        # For now, use first template for sorting (could be enhanced later)
        primary_template = template_names[0] if template_names else ""
        try:
            filtered_samples = self._cached_sort_samples(
                filtered_samples,
                sort_key,
                task,
                dataset_name,
                primary_template,
                tuple(selected_models),  # Convert to tuple for hashability
            )
        except Exception as e:
            # If sorting fails, use unsorted samples
            st.warning(f"Sorting failed, using default order: {str(e)}")
            # filtered_samples remains as is

        if not filtered_samples:
            st.warning("No samples match the selected filters")
            return

        # Sample selector with filtered samples
        # Note: Initial slider interaction may cause a page reload (Streamlit limitation)
        sample_idx = (
            st.slider(
                "Select sample",
                min_value=1,
                max_value=len(filtered_samples),
                value=1,
                format="Sample %d of " + str(len(filtered_samples)),
                key="analysis_sample_slider",
            )
            - 1
        )  # Convert back to 0-indexed for internal use

        sample_data = (
            filtered_samples[sample_idx] if sample_idx < len(filtered_samples) else {}
        )

        if sample_data:
            # Display sample info - pass filtered_samples for optimization
            self._render_sample_info(
                sample_data,
                sample_idx,
                task,
                dataset_name,
                template_names,
                selected_models,
                filtered_samples,
            )

            # Display sample content
            self._render_sample_content(sample_data)

            # Display ground truth
            self._render_ground_truth(sample_data)

            # Display model responses
            if selected_models:
                self._render_model_predictions(
                    selected_models, task, dataset_name, template_names, sample_data
                )

    def _cached_sort_samples(
        self,
        samples: List[Dict[str, Any]],
        sort_key: str,
        task: str,
        dataset_name: str,
        template_name: str,
        selected_models: Tuple[str, ...],
    ) -> List[Dict[str, Any]]:
        """Sort samples without caching to avoid reload issues."""
        # Convert tuple back to list for the loader
        selected_models_list = list(selected_models)

        return self.loader.sort_samples(
            samples,
            sort_key,
            task=task,
            dataset_name=dataset_name,
            template_name=template_name,
            selected_models=selected_models_list,
        )

    def _render_sample_info(
        self,
        sample_data: Dict[str, Any],
        sample_idx: int,
        task: str,
        dataset_name: str,
        template_names: List[str],
        selected_models: List[str],
        all_samples: List[Dict[str, Any]] = None,
    ):
        """Render sample information.

        Args:
            sample_data: Current sample data
            sample_idx: Current sample index
            task: Task name
            dataset_name: Dataset name
            selected_models: List of selected models
            all_samples: Optional list of all samples for optimization
        """
        st.subheader(f"Sample {sample_idx + 1}: {sample_data.get('sample_id', 'N/A')}")

        # Get task-specific summary fields with error handling
        # Use first template for now (could be enhanced to aggregate across templates)
        primary_template = template_names[0] if template_names else ""
        try:
            summary_fields = self.loader.get_sample_summary_fields(
                sample_data,
                task=task,
                dataset_name=dataset_name,
                template_name=primary_template,
                selected_models=selected_models,
                all_samples=all_samples,  # Pass all samples for optimization
            )
            summary_fields.sort(key=lambda x: x.get("order", 999))
        except Exception:
            # Fallback to basic fields if accuracy calculation fails - minimal error display
            st.warning("⚠️ Accuracy data unavailable")

            # Use basic fields only
            summary_fields = self.loader.get_sample_summary_fields(sample_data)
            summary_fields.sort(key=lambda x: x.get("order", 999))

        # Group fields by category and display
        if summary_fields:
            # Check if any field has category information
            has_categories = any(field.get("category") for field in summary_fields)

            if has_categories:
                # Group fields by category
                from collections import defaultdict

                categories = defaultdict(list)
                for field in summary_fields:
                    category = field.get("category", "other")
                    categories[category].append(field)

                # Display each category in its own row
                category_order = [
                    "basic",
                    "metrics",
                    "screening",
                    "screening_details",
                    "other",
                ]  # Define display order
                for category in category_order:
                    if category in categories:
                        category_fields = sorted(
                            categories[category], key=lambda x: x.get("order", 999)
                        )
                        if category_fields:
                            # Display fields in this category
                            num_cols = len(category_fields)
                            cols = st.columns(num_cols)
                            for i, field in enumerate(category_fields):
                                with cols[i]:
                                    st.write(f"**{field['label']}**: {field['value']}")

                            # Add small spacing between categories (except after the last one)
                            if category != category_order[-1] and any(
                                cat in categories
                                for cat in category_order[
                                    category_order.index(category) + 1 :
                                ]
                            ):
                                st.write("")  # Add spacing
            else:
                # Fallback to original layout if no category information
                num_cols = min(6, len(summary_fields))
                cols = st.columns(num_cols)
                for i, field in enumerate(summary_fields[:num_cols]):
                    with cols[i]:
                        st.write(f"**{field['label']}**: {field['value']}")

                # Show remaining fields in second row if needed
                if len(summary_fields) > 6:
                    remaining_fields = summary_fields[6:]
                    num_cols_2 = min(6, len(remaining_fields))
                    cols_2 = st.columns(num_cols_2)
                    for i, field in enumerate(remaining_fields[:num_cols_2]):
                        with cols_2[i]:
                            st.write(f"**{field['label']}**: {field['value']}")

    def _render_sample_content(self, sample_data: Dict[str, Any]):
        """Render sample content."""
        with st.expander("📝 Sample Content", expanded=True):
            display_fields = self.loader.get_sample_display_fields(sample_data)
            question_text = display_fields.get(
                "Question", sample_data.get("question", "")
            )
            # Use custom CSS to make disabled text area text black
            st.markdown(
                """
                <style>
                .stTextArea textarea:disabled {
                    color: #262730 !important;
                    -webkit-text-fill-color: #262730 !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Calculate dynamic height based on content
            line_count = question_text.count("\n") + 1
            # Minimum height of 100px, with 20px per line, maximum of 500px
            dynamic_height = min(max(line_count * 20 + 40, 100), 500)

            st.text_area(
                "Sample Content",
                value=question_text,
                height=dynamic_height,
                disabled=True,
                label_visibility="collapsed",
            )

        # Show extra content if available
        extra_content = self.loader.get_sample_extra_content(sample_data)
        if extra_content:
            with st.expander(
                extra_content["title"], expanded=extra_content.get("expanded", False)
            ):
                # Check for special content types
                if extra_content.get("type") == "jmle_question":
                    # Special rendering for JMLE questions
                    jmle_data = extra_content["content"]

                    # Question ID and text
                    st.write(f"**Question ID**: {jmle_data.get('sample_id', 'N/A')}")

                    if "question" in jmle_data:
                        # CSS already applied above for all disabled text areas
                        st.text_area(
                            "Question",
                            value=jmle_data["question"],
                            height=150,
                            disabled=True,
                            label_visibility="visible",
                        )

                    # Choices
                    if "choices" in jmle_data:
                        st.write("**Choices:**")
                        for choice_key, choice_text in jmle_data["choices"].items():
                            is_answer = (
                                "answer" in jmle_data
                                and choice_key in jmle_data["answer"]
                            )
                            if is_answer:
                                st.write(f"✅ **{choice_key}**: {choice_text}")
                            else:
                                st.write(f"　 {choice_key}: {choice_text}")
                else:
                    # Generic content display
                    st.write(extra_content["content"])

    def _render_ground_truth(self, sample_data: Dict[str, Any]):
        """Render ground truth information."""
        with st.expander("✅ Ground Truth", expanded=True):
            # Use formatter's display method if available
            if hasattr(self.formatter, "format_ground_truth_display"):
                display_items = self.formatter.format_ground_truth_display(sample_data)
                for item in display_items:
                    st.write(f"**{item['label']}**: {item['value']}")
            else:
                # Generic display for unknown tasks
                truth = self.formatter.format_ground_truth(sample_data)
                for key, value in truth.items():
                    if value is not None and value != "":
                        label = key.replace("_", " ").title()
                        st.write(f"**{label}**: {value}")

    def _render_model_predictions(
        self,
        selected_models: List[str],
        task: str,
        dataset_name: str,
        template_names: List[str],
        sample_data: Dict[str, Any],
    ):
        """Render model predictions comparison."""
        st.subheader("Experiment Predictions Comparison")

        # Create all model-template combinations
        experiments = []
        for model in selected_models:
            for template_name in template_names:
                # Batch load all model results first for efficiency
                model_results_map = self._get_cached_model_results(
                    tuple([model]), task, dataset_name, template_name
                )

                # Use cached results
                model_results = model_results_map.get(model, {})
                has_results = bool(model_results and "predictions" in model_results)

                if has_results:
                    predictions = model_results["predictions"].get("samples", [])
                    prediction = self._find_prediction(
                        predictions, sample_data, model_results, task
                    )
                    has_prediction = bool(prediction)
                else:
                    prediction = None
                    has_prediction = False

                experiment_name = f"{model} ({template_name})"
                experiments.append(
                    {
                        "name": experiment_name,
                        "model": model,
                        "template": template_name,
                        "prediction": prediction,
                        "has_prediction": has_prediction,
                        "has_results": has_results,
                    }
                )

        # Display each experiment
        for experiment in experiments:
            with st.expander(
                f"🧪 {experiment['name']}", expanded=experiment["has_prediction"]
            ):
                if experiment["has_results"] and experiment["has_prediction"]:
                    self._render_single_prediction(
                        experiment["prediction"],
                        sample_data,
                        experiment["model"],
                        task,
                        dataset_name,
                        experiment["template"],
                        experiment["name"],
                    )
                elif experiment["has_results"]:
                    st.warning("No prediction found for this sample")
                else:
                    st.warning(f"No results found for {experiment['name']}")

    def _find_prediction(
        self,
        predictions: List[Dict],
        sample_data: Dict[str, Any],
        model_results: Dict[str, Any] = None,
        task: str = None,
    ) -> Dict[str, Any]:
        """Find matching prediction for sample."""
        # For screening tasks, look in detailed_metrics instead of samples
        if (
            task == "medec_screening"
            and model_results
            and "predictions" in model_results
        ):
            predictions_data = model_results["predictions"]
            if (
                "detailed_metrics" in predictions_data
                and "screening_judge" in predictions_data["detailed_metrics"]
            ):
                detailed_metrics = predictions_data["detailed_metrics"][
                    "screening_judge"
                ]
                for pred in detailed_metrics:
                    if pred.get("sample_id") == sample_data["sample_id"]:
                        return pred

        # Default logic for other tasks
        for pred in predictions:
            if pred["sample_id"] == sample_data["sample_id"]:
                return pred
        return {}

    def _get_cached_model_results(
        self,
        selected_models: Tuple[str, ...],
        task: str,
        dataset_name: str,
        template_name: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Get cached model results with session state optimization.

        Args:
            selected_models: Tuple of model names
            task: Task name
            dataset_name: Dataset name
            template_name: Template name

        Returns:
            Dictionary mapping model name to results
        """
        cache_key = f"batch_model_results_{task}_{dataset_name}_{template_name}_{'-'.join(sorted(selected_models))}"

        # Check if we have this batch cached in session state
        if cache_key in st.session_state:
            return st.session_state[cache_key]

        # Load results for all models
        results_map = {}
        for model in selected_models:
            try:
                # Use the loader's built-in caching (now with session state support)
                model_results = self.loader.load_model_results(
                    task, dataset_name, model, template_name
                )
                if model_results:
                    results_map[model] = model_results
            except Exception:
                # Skip models that fail to load
                pass

        # Cache the batch results
        st.session_state[cache_key] = results_map
        return results_map

    def _render_single_prediction(
        self,
        prediction: Dict[str, Any],
        sample_data: Dict[str, Any],
        model: str = None,
        task: str = None,
        dataset_name: str = None,
        template_name: str = None,
        experiment_name: str = None,
    ):
        """Render a single model prediction."""
        # Format and compare prediction
        comparison = self.formatter.compare_prediction(prediction, sample_data)

        # Get composite score for this sample and model (if MEDEC task)
        composite_score = None
        if (
            task == "medec"
            and model
            and hasattr(self.loader, "_get_composite_score_for_sample")
        ):
            sample_id = sample_data.get("sample_id")
            if sample_id:
                composite_score = self.loader._get_composite_score_for_sample(
                    sample_id, task, dataset_name, template_name, [model]
                )

        # Use formatter's comparison display if available
        if hasattr(self.formatter, "format_comparison_display"):
            # Pass composite_score if the formatter supports it
            import inspect

            if (
                "composite_score"
                in inspect.signature(
                    self.formatter.format_comparison_display
                ).parameters
            ):
                display_items = self.formatter.format_comparison_display(
                    comparison, composite_score=composite_score
                )
            else:
                display_items = self.formatter.format_comparison_display(comparison)

            # Show metrics in columns
            if display_items.get("metrics"):
                cols = st.columns(len(display_items["metrics"]))
                for i, metric in enumerate(display_items["metrics"]):
                    with cols[i]:
                        # Special handling for metrics that show scores instead of correct/incorrect
                        if metric.get("show_score") and metric["correct"] is None:
                            st.info(f"{metric['label']}")
                            if metric.get("caption"):
                                st.caption(metric["caption"])
                        elif metric["correct"]:
                            st.success(f"{metric['label']}: ✅")
                            if metric.get("caption"):
                                st.caption(metric["caption"])
                        else:
                            st.error(f"{metric['label']}: ❌")
                            if metric.get("caption"):
                                st.caption(metric["caption"])

            # Show detailed predictions
            if display_items.get("details"):
                for detail in display_items["details"]:
                    st.write(f"**{detail['label']}**: {detail['value']}")
        else:
            # Generic comparison display
            for key, value in comparison.items():
                if value is not None and value != "":
                    label = key.replace("_", " ").title()
                    st.write(f"**{label}**: {value}")

        # Show raw response if available
        if model and task and dataset_name:
            self._render_raw_response(
                sample_data, model, task, dataset_name, template_name, experiment_name
            )

    def _render_raw_response(
        self,
        sample_data: Dict[str, Any],
        model: str,
        task: str,
        dataset_name: str,
        template_name: str,
        experiment_name: str = None,
    ):
        """Render raw response content and reasoning with toggle."""
        # Try to get cached model results
        cache_key = (
            f"single_model_results_{task}_{dataset_name}_{template_name}_{model}"
        )

        # Check session state cache first
        if cache_key in st.session_state:
            model_results = st.session_state[cache_key]
        else:
            # Load and cache model results
            try:
                model_results = self.loader.load_model_results(
                    task, dataset_name, model, template_name
                )
                st.session_state[cache_key] = model_results
            except Exception:
                return

        if not model_results or "raw_responses" not in model_results:
            return

        raw_responses = model_results["raw_responses"]

        # Find the raw response for this sample
        sample_raw_response = None
        if "interactions" in raw_responses:
            for interaction in raw_responses["interactions"]:
                # Check if interaction has sample_data with matching sample_id
                if (
                    "sample_data" in interaction
                    and interaction["sample_data"].get("sample_id")
                    == sample_data["sample_id"]
                ):
                    sample_raw_response = interaction.get("raw_response", {})
                    break

        if not sample_raw_response:
            return

        # Show content if available
        if "content" in sample_raw_response:
            with st.expander("📝 Raw Content", expanded=False):
                content = sample_raw_response["content"]
                # Calculate dynamic height
                line_count = content.count("\n") + 1
                dynamic_height = min(max(line_count * 20 + 40, 100), 400)
                # Create unique key using experiment name or fallback to model+template
                key_suffix = (
                    experiment_name.replace(" ", "_").replace("(", "").replace(")", "")
                    if experiment_name
                    else f"{model}_{template_name}"
                )
                st.text_area(
                    "Content",
                    value=content,
                    height=dynamic_height,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"raw_content_{key_suffix}_{sample_data['sample_id']}",
                )

        # Show reasoning if available
        if "reasoning" in sample_raw_response:
            with st.expander("🧠 Raw Reasoning", expanded=False):
                reasoning = sample_raw_response["reasoning"]
                # Calculate dynamic height
                line_count = reasoning.count("\n") + 1
                dynamic_height = min(max(line_count * 20 + 40, 100), 400)
                # Create unique key using experiment name or fallback to model+template
                key_suffix = (
                    experiment_name.replace(" ", "_").replace("(", "").replace(")", "")
                    if experiment_name
                    else f"{model}_{template_name}"
                )
                st.text_area(
                    "Reasoning",
                    value=reasoning,
                    height=dynamic_height,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"raw_reasoning_{key_suffix}_{sample_data['sample_id']}",
                )

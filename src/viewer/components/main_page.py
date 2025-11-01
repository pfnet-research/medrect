"""Main page component for benchmark viewer."""

import streamlit as st
from pathlib import Path
from typing import List

from ..task_registry import get_task_registry
from .overview_tab import OverviewTab
from .sample_analysis_tab import SampleAnalysisTab
from .metrics_comparison_tab import MetricsComparisonTab


class MainPage:
    """Main page controller for the benchmark viewer."""

    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent.parent

    def render(self):
        """Render the main page."""
        st.title("🔬 LLM Evaluation Benchmark Viewer")

        # Initialize components
        registry = get_task_registry(self.base_path)
        available_tasks = registry.get_available_tasks()

        if not available_tasks:
            st.error("No evaluation results found. Please run evaluations first.")
            st.stop()

        # Task, dataset, and template selection sidebar
        selected_task, selected_dataset, selected_templates = self._render_sidebar(
            available_tasks, registry
        )

        # Get task components
        try:
            components = registry.get_task_components(selected_task)
            loader, parser, formatter = (
                components  # Unpack assuming 3 components for now
            )
        except ValueError as e:
            st.error(f"Error loading task components: {e}")
            st.stop()

        # Load dataset
        dataset = self._load_dataset(loader, selected_dataset)

        # Model selection
        available_models = registry.get_task_models(selected_task)

        # Check which models have valid results
        valid_models = self._get_valid_models(
            available_models,
            registry,
            selected_task,
            selected_dataset,
            selected_templates,
        )

        selected_models = self._render_model_selection(available_models, valid_models)

        # Preload model data for better performance
        if selected_models:
            self._preload_model_data(
                loader,
                selected_task,
                selected_dataset,
                selected_templates,
                selected_models,
            )

        # Render tabs
        self._render_tabs(
            loader,
            parser,
            formatter,
            dataset,
            selected_task,
            selected_dataset,
            selected_templates,
            selected_models,
        )

    def _render_sidebar(self, available_tasks, registry):
        """Render task, dataset, and template selection sidebar."""
        st.sidebar.header("🎯 Task Selection")

        selected_task = st.sidebar.selectbox("Select Task", available_tasks, index=0)

        # Show compact task status info right after task selection
        self._show_compact_task_status(registry)

        st.sidebar.header("📂 Dataset Selection")
        available_datasets = registry.get_task_datasets(selected_task)

        selected_dataset = st.sidebar.selectbox(
            "Select Dataset",
            available_datasets,
            index=0 if available_datasets else None,
        )

        if not selected_dataset:
            st.error("No datasets available")
            st.stop()

        # Show dataset status inline - simplified approach
        try:
            components = registry.get_task_components(selected_task)
            loader, _, _ = components
            dataset_for_status = self._load_dataset(loader, selected_dataset)
            self._show_dataset_status(dataset_for_status)
        except Exception:
            # If loading fails, show simple fallback
            st.sidebar.markdown("*📂 Dataset selected*")

        available_templates = registry.get_task_templates(
            selected_task, selected_dataset
        )

        if not available_templates:
            st.error("No templates available for this task/dataset combination")
            st.stop()

        selected_templates = self._render_template_selection(available_templates)

        return selected_task, selected_dataset, selected_templates

    def _show_compact_task_status(self, registry):
        """Show compact task status in sidebar."""
        task_status = registry.get_task_status()

        # Categorize tasks by status
        ready_tasks = []
        data_only_tasks = []
        implementation_only_tasks = []
        configured_only_tasks = []

        for task_name, status in task_status.items():
            if status["overall_status"] == "ready":
                ready_tasks.append(task_name)
            elif status["overall_status"] == "data_only":
                data_only_tasks.append(task_name)
            elif status["overall_status"] == "implementation_only":
                implementation_only_tasks.append(task_name)
            else:
                configured_only_tasks.append(task_name)

        # Show compact status
        total_tasks = len(task_status)
        ready_count = len(ready_tasks)

        # Always show status
        with st.sidebar.expander(
            f"🎯 {ready_count}/{total_tasks} ready", expanded=False
        ):
            if ready_tasks:
                st.write(f"✅ **Ready**: {', '.join(ready_tasks)}")

            if data_only_tasks:
                st.write(f"⚠️ **Missing implementation**: {', '.join(data_only_tasks)}")

            if implementation_only_tasks:
                st.write(
                    f"🔧 **Missing results**: {', '.join(implementation_only_tasks)}"
                )

            if configured_only_tasks:
                st.write(
                    f"⚙️ **Missing implementation & results**: {', '.join(configured_only_tasks)}"
                )

    def _load_dataset(self, loader, selected_dataset):
        """Load and cache dataset."""

        @st.cache_data
        def load_dataset(dataset_name):
            try:
                return loader.load_dataset(dataset_name)
            except FileNotFoundError:
                return {"samples": [], "metadata": {"source": "results_only"}}

        dataset = load_dataset(selected_dataset)

        return dataset

    def _show_dataset_status(self, dataset):
        """Show dataset status in sidebar."""
        if dataset.get("metadata", {}).get("source") == "results_only":
            st.sidebar.markdown("*📂 Results only*")
        else:
            sample_count = len(dataset.get("samples", []))
            st.sidebar.markdown(f"*📂 {sample_count} samples*")

    def _get_valid_models(
        self, available_models, registry, task, dataset_name, template_names
    ):
        """Get list of models that have valid results for at least one template."""
        if not available_models:
            return []

        loader, _, _ = registry.get_task_components(task)
        valid_models = []

        for model in available_models:
            has_valid_results = False
            for template_name in template_names:
                try:
                    results = loader.load_model_results(
                        task, dataset_name, model, template_name
                    )
                    if results and "summary" in results:
                        has_valid_results = True
                        break
                except Exception:
                    continue

            if has_valid_results:
                valid_models.append(model)

        return valid_models

    def _render_model_selection(self, available_models, valid_models):
        """Render model selection in sidebar."""
        if not valid_models:
            st.sidebar.warning("No model results available")
            st.stop()

        # Initialize checkbox states individually
        for model in valid_models:
            checkbox_key = f"model_cb_{model}"
            if checkbox_key not in st.session_state:
                st.session_state[checkbox_key] = True  # Default to selected

        # Count current selections for header
        selected_count = sum(
            1
            for model in valid_models
            if st.session_state.get(f"model_cb_{model}", False)
        )

        # Header with count
        st.sidebar.header(f"🤖 Model Selection ({selected_count}/{len(valid_models)})")

        # Quick action buttons with callbacks
        col1, col2 = st.sidebar.columns(2)
        with col1:
            config = {"key": "select_all", "width": "stretch"}
            if st.button("Select All", **config):
                for model in valid_models:
                    st.session_state[f"model_cb_{model}"] = True
                st.rerun()
        with col2:
            config = {"key": "clear_all", "width": "stretch"}
            if st.button("Clear All", **config):
                for model in valid_models:
                    st.session_state[f"model_cb_{model}"] = False
                st.rerun()

        # Render checkboxes using session state keys directly
        selected_models = []
        for model in valid_models:
            checkbox_key = f"model_cb_{model}"
            if st.sidebar.checkbox(model, key=checkbox_key):
                selected_models.append(model)

        return selected_models

    def _render_template_selection(self, available_templates):
        """Render template selection with checkboxes."""
        # Initialize checkbox states individually (default to first template only)
        for i, template in enumerate(available_templates):
            checkbox_key = f"template_cb_{template}"
            if checkbox_key not in st.session_state:
                st.session_state[checkbox_key] = (
                    i == 0
                )  # Only first template selected by default

        # Count current selections for header
        selected_count = sum(
            1
            for template in available_templates
            if st.session_state.get(f"template_cb_{template}", False)
        )

        # Header with count
        st.sidebar.header(
            f"📝 Template Selection ({selected_count}/{len(available_templates)})"
        )

        # Quick action buttons
        col1, col2 = st.sidebar.columns(2)
        with col1:
            config = {"key": "select_all_templates", "width": "stretch"}
            if st.button("Select All", **config):
                for template in available_templates:
                    st.session_state[f"template_cb_{template}"] = True
                st.rerun()
        with col2:
            config = {"key": "clear_all_templates", "width": "stretch"}
            if st.button("Clear All", **config):
                for template in available_templates:
                    st.session_state[f"template_cb_{template}"] = False
                st.rerun()

        # Render checkboxes using session state keys directly
        selected_templates = []
        for template in available_templates:
            checkbox_key = f"template_cb_{template}"
            if st.sidebar.checkbox(template, key=checkbox_key):
                selected_templates.append(template)

        # Ensure at least one template is selected
        if not selected_templates:
            st.sidebar.warning("Please select at least one template")
            st.stop()

        return selected_templates

    def _preload_model_data(
        self,
        loader,
        task: str,
        dataset_name: str,
        template_names: List[str],
        selected_models: List[str],
    ):
        """Preload model data for better performance during sample analysis."""
        preload_key = f"preloaded_{task}_{dataset_name}_{'-'.join(sorted(template_names))}_{'-'.join(sorted(selected_models))}"

        # Check if already preloaded for this combination
        if preload_key in st.session_state:
            return

        # Show loading progress
        total_combinations = len(selected_models) * len(template_names)
        with st.spinner(
            f"Loading data for {len(selected_models)} models and {len(template_names)} templates..."
        ):
            loaded_count = 0
            for model in selected_models:
                for template_name in template_names:
                    try:
                        # This will now be cached in session state via the BaseDataLoader
                        loader.load_model_results(
                            task, dataset_name, model, template_name
                        )
                        loaded_count += 1
                    except Exception:
                        continue

            # Mark as preloaded
            st.session_state[preload_key] = {
                "loaded_combinations": loaded_count,
                "total_combinations": total_combinations,
            }

    def _render_tabs(
        self,
        loader,
        parser,
        formatter,
        dataset,
        task,
        dataset_name,
        template_names,
        models,
    ):
        """Render main content tabs."""
        # Create tabs - Streamlit automatically maintains tab state
        tabs = st.tabs(["📊 Overview", "🔍 Analysis", "📈 Metrics"])

        # Render each tab's content
        with tabs[0]:
            overview = OverviewTab(loader, parser, formatter, self.base_path)
            overview.render(dataset, task, dataset_name, template_names, models)

        with tabs[1]:
            sample_analysis = SampleAnalysisTab(
                loader, parser, formatter, self.base_path
            )
            sample_analysis.render(dataset, task, dataset_name, template_names, models)

        with tabs[2]:
            metrics_comparison = MetricsComparisonTab(
                loader, parser, formatter, self.base_path
            )
            metrics_comparison.render(
                dataset, task, dataset_name, template_names, models
            )

"""Base data loader for benchmark viewer."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

try:
    import streamlit as st

    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


class BaseDataLoader(ABC):
    """Base class for task-specific data loaders."""

    def __init__(self, base_path: Path, task_config: Optional[Any] = None):
        self.base_path = Path(base_path)
        self.task_config = task_config
        self.results_base = self.base_path / "results"
        self.data_base = self.base_path / (
            task_config.data_dir if task_config else "data"
        )
        self._model_results_cache = {}  # Cache for model results
        self._dataset_cache = {}  # Cache for dataset data

    @abstractmethod
    def get_task_name(self) -> str:
        """Return the name of the task."""
        pass

    @abstractmethod
    def load_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Load dataset for viewing."""
        pass

    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets for this task from config."""
        if self.task_config:
            return list(self.task_config.datasets.keys())
        return self._discover_datasets()

    @abstractmethod
    def _discover_datasets(self) -> List[str]:
        """Discover datasets when no config is available."""
        pass

    @abstractmethod
    def get_sample_display_fields(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Get fields to display for a sample."""
        pass

    def get_filter_options(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get available filter options for samples.

        Returns dict with:
        - categories: List of category values for filtering
        - has_special_filter: Whether to show special filters (e.g., error-only)
        """
        return {"categories": [], "has_special_filter": False}

    def get_sort_options(self) -> List[Dict[str, str]]:
        """Get available sort options.

        Returns list of dicts with:
        - label: Display label for the option
        - key: Sort key to use
        """
        return [
            {"label": "Original Order (Default)", "key": "original_order"},
            {"label": "Sample ID (A-Z)", "key": "sample_id"},
        ]

    def apply_filters(
        self, samples: List[Dict[str, Any]], filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply filters to samples.

        Args:
            samples: List of samples to filter
            filters: Dict containing filter settings

        Returns:
            Filtered list of samples
        """
        return samples

    def sort_samples(
        self, samples: List[Dict[str, Any]], sort_key: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """Sort samples by specified key.

        Args:
            samples: List of samples to sort
            sort_key: Key to sort by
            **kwargs: Additional parameters (task, dataset_name, selected_models, etc.)

        Returns:
            Sorted list of samples
        """
        if sort_key == "sample_id":
            return sorted(samples, key=lambda x: x.get("sample_id", ""))
        elif sort_key == "original_order":
            return samples  # Keep original order
        return samples

    def get_dataset_statistics(self, dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get dataset statistics to display.

        Returns list of dicts with:
        - label: Display label
        - value: Statistic value
        - order: Display order (1-4)
        """
        total = len(dataset.get("samples", []))
        return [{"label": "Total Samples", "value": total, "order": 1}]

    def get_dataset_distribution(
        self, dataset: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get distribution data for visualization.

        Returns dict with:
        - title: Chart title
        - data: Dict of label -> count
        Returns None if no distribution to show.
        """
        return None

    def get_sample_summary_fields(
        self, sample: Dict[str, Any], **kwargs
    ) -> List[Dict[str, str]]:
        """Get summary fields to display for a sample.

        Args:
            sample: Sample data
            **kwargs: Additional parameters (task, dataset_name, selected_models, etc.)

        Returns list of dicts with:
        - label: Field label
        - value: Field value
        - order: Display order
        """
        return [{"label": "ID", "value": str(sample.get("sample_id", "")), "order": 1}]

    def get_sample_extra_content(
        self, sample: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get extra content to display for a sample.

        Returns dict with:
        - title: Expander title
        - content: Content to display
        - expanded: Whether to expand by default
        Returns None if no extra content.
        """
        return None

    def load_model_results(
        self, task: str, dataset: str, model: str, template: str = "0_shot_ja"
    ) -> Dict[str, Any]:
        """Load results for a specific model and template."""
        cache_key = f"{task}:{dataset}:{model}:{template}"

        # Check Streamlit session state cache first if available
        if HAS_STREAMLIT and hasattr(st, "session_state"):
            session_cache_key = f"model_results_cache_{cache_key}"
            if session_cache_key in st.session_state:
                return st.session_state[session_cache_key]

        # Check instance cache
        if cache_key in self._model_results_cache:
            return self._model_results_cache[cache_key]

        # Try multiple possible base paths
        possible_base_paths = [
            self.results_base,  # base_path/results
        ]

        result_data = {}

        # Try each possible base path
        for base_path in possible_base_paths:
            model_path = base_path / task / dataset / model / template

            # Find the latest timestamp directory
            if model_path.exists():
                timestamp_dirs = [d for d in model_path.iterdir() if d.is_dir()]
                if timestamp_dirs:
                    latest_dir = max(timestamp_dirs, key=lambda x: x.name)

                    # Load predictions
                    predictions_file = latest_dir / "predictions.json"
                    if predictions_file.exists():
                        with open(predictions_file, "r", encoding="utf-8") as f:
                            result_data["predictions"] = json.load(f)

                    # Load raw responses
                    raw_responses_file = latest_dir / "raw_responses.json"
                    if raw_responses_file.exists():
                        with open(raw_responses_file, "r", encoding="utf-8") as f:
                            result_data["raw_responses"] = json.load(f)

                    # Use predictions.json as summary source
                    if "predictions" in result_data:
                        result_data["summary"] = result_data["predictions"]

                    # Found results, no need to try other paths
                    break

        # Cache the result in both instance and session state
        self._model_results_cache[cache_key] = result_data

        # Also cache in Streamlit session state if available
        if HAS_STREAMLIT and hasattr(st, "session_state"):
            session_cache_key = f"model_results_cache_{cache_key}"
            st.session_state[session_cache_key] = result_data

        return result_data

"""MEDEC-specific data loader."""

from typing import Dict, List, Any, Optional
import json
from collections import defaultdict

from .base import BaseDataLoader

try:
    import streamlit as st

    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


class MEDECDataLoader(BaseDataLoader):
    """MEDEC task data loader."""

    def __init__(self, base_path, task_config):
        super().__init__(base_path, task_config)
        self._accuracy_cache = {}  # Cache for precomputed accuracies

    def get_task_name(self) -> str:
        return "medec"

    def load_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Load MEDEC dataset for viewing."""
        # Check session state cache first
        if HAS_STREAMLIT and hasattr(st, "session_state"):
            session_cache_key = f"medec_dataset_{dataset_name}"
            if session_cache_key in st.session_state:
                return st.session_state[session_cache_key]

        # Check instance cache
        if hasattr(self, "_dataset_cache") and dataset_name in self._dataset_cache:
            return self._dataset_cache[dataset_name]

        # Use config to get dataset path
        dataset_info = self.task_config.datasets[dataset_name]
        dataset_path = self.base_path / self.task_config.data_dir / dataset_info["file"]

        if not dataset_path.exists():
            # Try to find in results directory structure
            results_datasets = self._discover_datasets()
            if dataset_name in results_datasets:
                # Return empty dataset structure for results-only datasets
                return {"samples": [], "metadata": {"source": "results_only"}}
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Normalize to consistent format
        if isinstance(data, list):
            samples = []
            for item in data:
                sample = {
                    "sample_id": item["sample_id"],
                    "question": item["sentences"],
                    "has_error": item["error_flag"] == 1,
                    "error_type": item["error_type"],
                    "error_sentence": item["error_sentence"],  # Can be None
                    "error_sentence_id": item[
                        "error_sentence_id"
                    ],  # Add error sentence ID
                    "corrected_sentence": item["corrected_sentence"],  # Can be None
                    "metadata": item["metadata"],
                }
                samples.append(sample)
            result = {"samples": samples}
        else:
            result = data

        # Cache the result in both instance and session state
        if hasattr(self, "_dataset_cache"):
            self._dataset_cache[dataset_name] = result

        # Also cache in session state if available
        if HAS_STREAMLIT and hasattr(st, "session_state"):
            session_cache_key = f"medec_dataset_{dataset_name}"
            st.session_state[session_cache_key] = result

        return result

    def _discover_datasets(self) -> List[str]:
        """Discover datasets when no config is available."""
        datasets = set()

        # Check results directory
        medec_results = self.results_base / "medec"
        if medec_results.exists():
            for dataset_dir in medec_results.iterdir():
                if dataset_dir.is_dir() and not dataset_dir.name.startswith("."):
                    datasets.add(dataset_dir.name)

        return sorted(list(datasets))

    def get_sample_display_fields(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Get MEDEC-specific display fields."""
        display_fields = {
            "ID": sample["sample_id"],
            "Has Error": "Yes" if sample["has_error"] else "No",
            "Error Type": sample["error_type"] or "None",
            "Question": sample["question"],
            "Error Sentence": sample["error_sentence"] or "",
            "Corrected Sentence": sample["corrected_sentence"] or "",
            "Error Sentence ID": sample["error_sentence_id"],
        }

        return display_fields

    def analyze_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze MEDEC dataset statistics."""
        samples = dataset["samples"]

        # Error type distribution
        error_counts = defaultdict(int)
        for sample in samples:
            error_type = sample["error_type"] or "None"
            error_counts[error_type] += 1

        # Error presence
        has_error_count = sum(1 for s in samples if s["has_error"])

        return {
            "total_samples": len(samples),
            "error_types": dict(error_counts),
            "with_errors": has_error_count,
            "without_errors": len(samples) - has_error_count,
        }

    def get_filter_options(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get MEDEC-specific filter options."""
        error_types = list(set(s["error_type"] or "None" for s in samples))
        error_types.sort()

        return {
            "categories": error_types,
            "category_label": "Error Type",
            "binary_status_options": ["With Errors", "Without Errors"],
            "binary_status_label": "Show samples",
        }

    def get_sort_options(self) -> List[Dict[str, str]]:
        """Get MEDEC-specific sort options."""
        return [
            {"label": "Original Order (Default)", "key": "original_order"},
            {"label": "Sample ID (A-Z)", "key": "sample_id"},
            {
                "label": "Error Detection Accuracy (Low to High)",
                "key": "error_detection_asc",
            },
            {
                "label": "Error Detection Accuracy (High to Low)",
                "key": "error_detection_desc",
            },
            {
                "label": "Sentence Extraction Accuracy (Low to High)",
                "key": "sentence_extraction_asc",
            },
            {
                "label": "Sentence Extraction Accuracy (High to Low)",
                "key": "sentence_extraction_desc",
            },
        ]

    def apply_filters(
        self, samples: List[Dict[str, Any]], filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply MEDEC-specific filters to samples."""
        filtered = samples

        # Apply binary status filter (has precedence over special filter)
        if "binary_status" in filters and filters["binary_status"]:
            selected_status = filters["binary_status"]
            if len(selected_status) == 1:  # Only one option selected
                if "With Errors" in selected_status:
                    filtered = [s for s in filtered if s["has_error"]]
                elif "Without Errors" in selected_status:
                    filtered = [s for s in filtered if not s["has_error"]]
            # If both selected or none selected, show all (no filtering)

        # Apply category filter (error types)
        if "categories" in filters and filters["categories"]:
            filtered = [
                s
                for s in filtered
                if (s["error_type"] or "None") in filters["categories"]
            ]

        return filtered

    def sort_samples(
        self,
        samples: List[Dict[str, Any]],
        sort_key: str,
        task: str = None,
        dataset_name: str = None,
        template_name: str = None,
        selected_models: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """Sort MEDEC samples by specified key."""
        if sort_key == "original_order":
            return samples  # Keep original order
        elif sort_key == "sample_id":
            return sorted(samples, key=lambda x: x.get("sample_id", ""))
        elif sort_key in [
            "error_detection_asc",
            "error_detection_desc",
            "sentence_extraction_error_accuracy_asc",
            "sentence_extraction_error_accuracy_desc",
        ]:
            # Precompute accuracies for all samples if not cached
            cache_key = self._get_cache_key(
                task, dataset_name, template_name, selected_models, sort_key
            )
            if not self._is_accuracy_cached(cache_key):
                # Only compute the specific metric needed for sorting
                compute_error_detection = sort_key.startswith("error_detection")
                compute_sentence_extraction = sort_key.startswith("sentence_extraction")
                self._compute_accuracies_unified(
                    task=task,
                    dataset_name=dataset_name,
                    template_name=template_name,
                    selected_models=selected_models,
                    samples=samples,
                    compute_error_detection=compute_error_detection,
                    compute_sentence_extraction=compute_sentence_extraction,
                )

            # Use cached accuracies for sorting
            cached_accuracies = self._get_cached_accuracies(cache_key)
            samples_with_accuracy = []
            for sample in samples:
                sample_id = sample["sample_id"]
                accuracy = cached_accuracies.get(sample_id, 0.0)
                samples_with_accuracy.append((sample, accuracy))

            # Sort by accuracy
            reverse_order = sort_key.endswith("_desc")
            samples_with_accuracy.sort(
                key=lambda x: (x[1], x[0]["sample_id"]), reverse=reverse_order
            )

            return [item[0] for item in samples_with_accuracy]
        elif sort_key in [
            "error_correction_top_metric_asc",
            "error_correction_top_metric_desc",
        ]:
            # Sort by composite_score from detailed_metrics
            if not task or not dataset_name or not template_name or not selected_models:
                return samples

            # Load composite scores from all selected models
            samples_with_scores = []
            for sample in samples:
                sample_id = sample["sample_id"]
                composite_scores = []

                for model in selected_models:
                    try:
                        model_results = self.load_model_results(
                            task, dataset_name, model, template_name
                        )
                        predictions_data = model_results.get("predictions")
                        if (
                            predictions_data
                            and "detailed_metrics" in predictions_data
                            and "error_correction_full"
                            in predictions_data["detailed_metrics"]
                            and "samples" in predictions_data
                        ):
                            # Create mapping from sample_id to sample_index
                            model_samples = predictions_data["samples"]
                            sample_index_map = {
                                s["sample_id"]: idx
                                for idx, s in enumerate(model_samples)
                            }

                            # Get sample_index for this sample_id
                            sample_index = sample_index_map.get(sample_id)
                            if sample_index is not None:
                                ec_metrics = predictions_data["detailed_metrics"][
                                    "error_correction_full"
                                ]
                                # Find the sample's composite_score by sample_index
                                for ec_sample in ec_metrics:
                                    if ec_sample.get("sample_index") == sample_index:
                                        composite_scores.append(
                                            ec_sample.get("composite_score", 0.0)
                                        )
                                        break
                    except Exception:
                        continue

                # Use average composite score across models
                avg_score = (
                    sum(composite_scores) / len(composite_scores)
                    if composite_scores
                    else 0.0
                )
                samples_with_scores.append((sample, avg_score))

            # Sort by composite score
            reverse_order = sort_key.endswith("_desc")
            samples_with_scores.sort(
                key=lambda x: (x[1], x[0]["sample_id"]), reverse=reverse_order
            )

            return [item[0] for item in samples_with_scores]

        # Fallback to original order for unknown sort keys
        return samples

    def _get_cache_key(
        self,
        task: str,
        dataset_name: str,
        template_name: str,
        selected_models: List[str],
        sort_key: str,
    ) -> str:
        """Generate cache key for accuracy computation."""
        if not task or not dataset_name or not template_name or not selected_models:
            return ""
        models_str = "-".join(sorted(selected_models))
        return f"{task}:{dataset_name}:{template_name}:{models_str}:{sort_key.split('_')[0]}"  # error_detection or sentence_extraction

    def _get_session_cache_key(self, base_key: str) -> str:
        """Generate session state cache key."""
        return f"medec_accuracy_cache_{base_key}"

    def _get_accuracy_from_cache(
        self, cache_key: str, sample_id: str
    ) -> Optional[float]:
        """Get accuracy from cache (session state first, then instance)."""
        # Try session state first
        if HAS_STREAMLIT and hasattr(st, "session_state"):
            session_key = self._get_session_cache_key(cache_key)
            if session_key in st.session_state:
                return st.session_state[session_key].get(sample_id)

        # Fall back to instance cache
        if cache_key in self._accuracy_cache:
            return self._accuracy_cache[cache_key].get(sample_id)

        return None

    def _set_accuracy_cache(self, cache_key: str, accuracies: Dict[str, float]):
        """Set accuracy cache in both session state and instance."""
        # Set in instance cache
        self._accuracy_cache[cache_key] = accuracies

        # Set in session state if available
        if HAS_STREAMLIT and hasattr(st, "session_state"):
            session_key = self._get_session_cache_key(cache_key)
            st.session_state[session_key] = accuracies

    def _is_accuracy_cached(self, cache_key: str) -> bool:
        """Check if accuracy cache exists (session state first, then instance)."""
        # Check session state first
        if HAS_STREAMLIT and hasattr(st, "session_state"):
            session_key = self._get_session_cache_key(cache_key)
            if session_key in st.session_state:
                return True

        # Check instance cache
        return cache_key in self._accuracy_cache

    def _get_cached_accuracies(self, cache_key: str) -> Dict[str, float]:
        """Get cached accuracies (session state first, then instance)."""
        # Try session state first
        if HAS_STREAMLIT and hasattr(st, "session_state"):
            session_key = self._get_session_cache_key(cache_key)
            if session_key in st.session_state:
                return st.session_state[session_key]

        # Fall back to instance cache
        return self._accuracy_cache.get(cache_key, {})

    def _precompute_accuracies(
        self,
        samples: List[Dict[str, Any]],
        sort_key: str,
        task: str,
        dataset_name: str,
        template_name: str,
        selected_models: List[str],
        cache_key: str,
    ) -> None:
        """Precompute accuracies for all samples and cache them."""
        # Determine which metric types to compute
        compute_error_detection = sort_key.startswith("error_detection")
        compute_sentence_extraction = sort_key.startswith("sentence_extraction")

        # Use unified method
        self._compute_accuracies_unified(
            task=task,
            dataset_name=dataset_name,
            template_name=template_name,
            selected_models=selected_models,
            samples=samples,
            compute_error_detection=compute_error_detection,
            compute_sentence_extraction=compute_sentence_extraction,
        )

    def _calculate_sample_accuracy(
        self,
        sample: Dict[str, Any],
        sort_key: str,
        task: str,
        dataset_name: str,
        template_name: str,
        selected_models: List[str],
    ) -> float:
        """Calculate accuracy for a sample across selected models."""
        if not selected_models or not task or not dataset_name or not template_name:
            return 0.0

        # Check if we have cached accuracy for this combination
        cache_key = self._get_cache_key(
            task, dataset_name, template_name, selected_models, sort_key
        )
        cached_accuracy = self._get_accuracy_from_cache(cache_key, sample["sample_id"])
        if cached_accuracy is not None:
            return cached_accuracy

        # Fallback to individual calculation (shouldn't happen often)
        correct_count = 0
        total_count = 0

        for model in selected_models:
            try:
                model_results = self.load_model_results(
                    task, dataset_name, model, template_name
                )
                if not model_results or "predictions" not in model_results:
                    continue

                # Find prediction for this sample
                predictions = model_results["predictions"].get("samples", [])
                prediction = None
                for pred in predictions:
                    if pred["sample_id"] == sample["sample_id"]:
                        prediction = pred
                        break

                if not prediction:
                    continue

                # Calculate accuracy based on sort key
                if sort_key.startswith("error_detection"):
                    pred_has_error = prediction["predictions"]["errordetection"] == 1
                    actual_has_error = sample["has_error"]
                    if pred_has_error == actual_has_error:
                        correct_count += 1
                elif sort_key.startswith("sentence_extraction"):
                    # Only count if both have error (sentence extraction is only meaningful when there's an error)
                    if sample["has_error"]:
                        pred_sentence_idx = prediction["predictions"][
                            "sentenceextraction"
                        ]
                        actual_sentence_id = sample["error_sentence_id"]
                        if pred_sentence_idx == actual_sentence_id:
                            correct_count += 1

                total_count += 1

            except Exception:
                continue

        return correct_count / total_count if total_count > 0 else 0.0

    def get_sample_summary_extra(
        self, sample_data: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """Get extra info to display in sample summary."""
        if (
            sample_data["has_error"]
            and "error_sentence_id" in sample_data
            and sample_data["error_sentence_id"]
        ):
            return {
                "label": "Error Sentence ID",
                "value": str(sample_data["error_sentence_id"]),
            }
        return None

    def get_dataset_statistics(self, dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get MEDEC dataset statistics."""
        stats = self.analyze_dataset(dataset)
        return [
            {"label": "Total Samples", "value": stats["total_samples"], "order": 1},
            {"label": "Without Errors", "value": stats["without_errors"], "order": 2},
            {"label": "With Errors", "value": stats["with_errors"], "order": 3},
            {"label": "Error Types", "value": len(stats["error_types"]), "order": 4},
        ]

    def get_dataset_distribution(
        self, dataset: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get MEDEC error distribution."""
        stats = self.analyze_dataset(dataset)
        if stats["error_types"]:
            return {"title": "Error Type Distribution", "data": stats["error_types"]}
        return None

    def get_sample_summary_fields(
        self, sample: Dict[str, Any], **kwargs
    ) -> List[Dict[str, str]]:
        """Get MEDEC sample summary fields."""
        fields = [
            {
                "label": "ID",
                "value": str(sample["sample_id"]),
                "order": 1,
                "category": "basic",
            },
            {
                "label": "Has Error",
                "value": "Yes" if sample["has_error"] else "No",
                "order": 2,
                "category": "basic",
            },
            {
                "label": "Error Type",
                "value": sample["error_type"] or "None",
                "order": 3,
                "category": "basic",
            },
        ]
        if sample.get("has_error") and sample.get("error_sentence_id"):
            fields.append(
                {
                    "label": "Error Sentence ID",
                    "value": str(sample["error_sentence_id"]),
                    "order": 4,
                    "category": "basic",
                }
            )

        # Add accuracy fields if model data is provided
        task = kwargs.get("task")
        dataset_name = kwargs.get("dataset_name")
        template_name = kwargs.get("template_name")
        selected_models = kwargs.get("selected_models")
        samples = kwargs.get("all_samples")  # Pass all samples to avoid reloading

        if task and dataset_name and template_name and selected_models:
            try:
                # Ensure cache is populated for both metrics
                self._ensure_accuracy_cache(
                    sample, task, dataset_name, template_name, selected_models, samples
                )

                # Get cached accuracies
                error_detection_cache_key = self._get_cache_key(
                    task,
                    dataset_name,
                    template_name,
                    selected_models,
                    "error_detection_asc",
                )
                sentence_extraction_cache_key = self._get_cache_key(
                    task,
                    dataset_name,
                    template_name,
                    selected_models,
                    "sentence_extraction_asc",
                )

                sample_id = sample["sample_id"]

                # Error Detection Accuracy
                error_detection_accuracy = (
                    self._get_accuracy_from_cache(error_detection_cache_key, sample_id)
                    or 0.0
                )
                fields.append(
                    {
                        "label": "Error Detection Accuracy",
                        "value": f"{error_detection_accuracy:.1%}",
                        "order": 5,
                        "category": "metrics",
                    }
                )

                # Sentence Extraction Accuracy (only if sample has error)
                if sample["has_error"]:
                    sentence_extraction_accuracy = (
                        self._get_accuracy_from_cache(
                            sentence_extraction_cache_key, sample_id
                        )
                        or 0.0
                    )
                    fields.append(
                        {
                            "label": "Sentence Extraction Accuracy",
                            "value": f"{sentence_extraction_accuracy:.1%}",
                            "order": 6,
                            "category": "metrics",
                        }
                    )

                # Add Composite Score
                composite_score = self._get_composite_score_for_sample(
                    sample_id, task, dataset_name, template_name, selected_models
                )
                if composite_score is not None:
                    fields.append(
                        {
                            "label": "Composite Score",
                            "value": f"{composite_score:.3f}",
                            "order": 7,
                            "category": "metrics",
                        }
                    )
            except Exception:
                # Silently skip accuracy calculation if it fails
                pass

        return fields

    def _get_composite_score_for_sample(
        self,
        sample_id: str,
        task: str,
        dataset_name: str,
        template_name: str,
        selected_models: List[str],
    ) -> Optional[float]:
        """Get composite score for a sample across selected models."""
        if not task or not dataset_name or not template_name or not selected_models:
            return None

        composite_scores = []

        for model in selected_models:
            try:
                model_results = self.load_model_results(
                    task, dataset_name, model, template_name
                )
                # The actual data is in model_results['predictions'] (which contains the predictions.json content)
                predictions_data = model_results.get("predictions")

                if (
                    predictions_data
                    and "detailed_metrics" in predictions_data
                    and "error_correction_full" in predictions_data["detailed_metrics"]
                    and "samples" in predictions_data
                ):
                    # Create mapping from sample_id to sample_index
                    samples = predictions_data["samples"]
                    sample_index_map = {
                        sample["sample_id"]: idx for idx, sample in enumerate(samples)
                    }

                    # Get sample_index for this sample_id
                    sample_index = sample_index_map.get(sample_id)
                    if sample_index is not None:
                        ec_metrics = predictions_data["detailed_metrics"][
                            "error_correction_full"
                        ]
                        # Find the sample's composite_score by sample_index
                        for ec_sample in ec_metrics:
                            if ec_sample.get("sample_index") == sample_index:
                                composite_scores.append(
                                    ec_sample.get("composite_score", 0.0)
                                )
                                break
            except Exception:
                continue

        # Return average composite score across models
        return (
            sum(composite_scores) / len(composite_scores) if composite_scores else None
        )

    def _ensure_accuracy_cache(
        self,
        sample: Dict[str, Any],
        task: str,
        dataset_name: str,
        template_name: str,
        selected_models: List[str],
        samples: List[Dict[str, Any]] = None,
    ) -> None:
        """Ensure accuracy cache is populated for both error detection and sentence extraction.

        Args:
            sample: Current sample being viewed
            task: Task name
            dataset_name: Dataset name
            selected_models: List of selected model names
            samples: Optional list of all samples to avoid reloading dataset
        """
        try:
            error_detection_key = self._get_cache_key(
                task,
                dataset_name,
                template_name,
                selected_models,
                "error_detection_asc",
            )
            sentence_extraction_key = self._get_cache_key(
                task,
                dataset_name,
                template_name,
                selected_models,
                "sentence_extraction_asc",
            )

            # Check if both caches exist
            if not self._is_accuracy_cached(
                error_detection_key
            ) or not self._is_accuracy_cached(sentence_extraction_key):
                # Need to populate cache - pass samples to avoid reloading
                self._compute_accuracies_unified(
                    task=task,
                    dataset_name=dataset_name,
                    template_name=template_name,
                    selected_models=selected_models,
                    samples=samples,
                    compute_error_detection=True,
                    compute_sentence_extraction=True,
                )
        except Exception:
            # If cache population fails, set empty cache to prevent repeated attempts
            error_detection_key = self._get_cache_key(
                task,
                dataset_name,
                template_name,
                selected_models,
                "error_detection_asc",
            )
            sentence_extraction_key = self._get_cache_key(
                task,
                dataset_name,
                template_name,
                selected_models,
                "sentence_extraction_asc",
            )
            self._set_accuracy_cache(error_detection_key, {})
            self._set_accuracy_cache(sentence_extraction_key, {})

    def _compute_accuracies_unified(
        self,
        task: str,
        dataset_name: str,
        template_name: str,
        selected_models: List[str],
        samples: List[Dict[str, Any]] = None,
        compute_error_detection: bool = True,
        compute_sentence_extraction: bool = True,
    ) -> None:
        """Unified method to compute accuracies for all samples.

        Args:
            task: Task name
            dataset_name: Dataset name
            template_name: Template name
            selected_models: List of selected model names
            samples: Optional list of samples to avoid reloading dataset
            compute_error_detection: Whether to compute error detection accuracy
            compute_sentence_extraction: Whether to compute sentence extraction accuracy
        """
        if not selected_models or not task or not dataset_name or not template_name:
            # Set empty caches if needed
            if compute_error_detection:
                key = self._get_cache_key(
                    task,
                    dataset_name,
                    template_name,
                    selected_models,
                    "error_detection_asc",
                )
                self._set_accuracy_cache(key, {})
            if compute_sentence_extraction:
                key = self._get_cache_key(
                    task,
                    dataset_name,
                    template_name,
                    selected_models,
                    "sentence_extraction_asc",
                )
                self._set_accuracy_cache(key, {})
            return

        # Load all model results once
        all_model_predictions = {}
        for model in selected_models:
            try:
                model_results = self.load_model_results(
                    task, dataset_name, model, template_name
                )
                if model_results and "predictions" in model_results:
                    predictions = model_results["predictions"].get("samples", [])
                    # Create lookup dictionary for faster access
                    all_model_predictions[model] = {
                        pred["sample_id"]: pred for pred in predictions
                    }
            except Exception:
                continue

        # Use provided samples or load dataset
        if samples is None:
            dataset = self.load_dataset(dataset_name)
            samples = dataset.get("samples", [])

        # Initialize accuracy dictionaries
        error_detection_accuracies = {} if compute_error_detection else None
        sentence_extraction_accuracies = {} if compute_sentence_extraction else None

        for sample in samples:
            sample_id = sample["sample_id"]

            # Error detection counters
            error_detection_correct = 0
            error_detection_total = 0

            # Sentence extraction counters
            sentence_extraction_correct = 0
            sentence_extraction_total = 0

            for model in selected_models:
                if model not in all_model_predictions:
                    continue

                prediction = all_model_predictions[model].get(sample_id)
                if not prediction:
                    continue

                # Error Detection calculation
                if compute_error_detection:
                    pred_has_error = prediction["predictions"]["errordetection"] == 1
                    actual_has_error = sample["has_error"]
                    if pred_has_error == actual_has_error:
                        error_detection_correct += 1
                    error_detection_total += 1

                # Sentence Extraction calculation (only for samples with errors)
                if compute_sentence_extraction and sample["has_error"]:
                    pred_sentence_idx = prediction["predictions"].get(
                        "sentenceextraction"
                    )
                    actual_sentence_id = sample.get("error_sentence_id")
                    if pred_sentence_idx is not None and actual_sentence_id is not None:
                        if pred_sentence_idx == actual_sentence_id:
                            sentence_extraction_correct += 1
                        sentence_extraction_total += 1

            # Store accuracies
            if compute_error_detection:
                error_detection_accuracies[sample_id] = (
                    error_detection_correct / error_detection_total
                    if error_detection_total > 0
                    else 0.0
                )
            if compute_sentence_extraction:
                sentence_extraction_accuracies[sample_id] = (
                    sentence_extraction_correct / sentence_extraction_total
                    if sentence_extraction_total > 0
                    else 0.0
                )

        # Cache results
        if compute_error_detection:
            error_detection_key = self._get_cache_key(
                task,
                dataset_name,
                template_name,
                selected_models,
                "error_detection_asc",
            )
            self._set_accuracy_cache(error_detection_key, error_detection_accuracies)
        if compute_sentence_extraction:
            sentence_extraction_key = self._get_cache_key(
                task,
                dataset_name,
                template_name,
                selected_models,
                "sentence_extraction_asc",
            )
            self._set_accuracy_cache(
                sentence_extraction_key, sentence_extraction_accuracies
            )

    def get_sample_extra_content(
        self, sample: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get JMLE original question content."""
        if "metadata" in sample and "original_jmle_data" in sample["metadata"]:
            return {
                "title": "📚 Original JMLE Question",
                "content": sample["metadata"]["original_jmle_data"],
                "expanded": False,
                "type": "jmle_question",  # Special type for custom rendering
            }
        return None

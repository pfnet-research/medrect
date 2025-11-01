"""MEDEC Screening-specific data loader for quality assessment."""

from typing import Dict, List, Any, Optional
from collections import defaultdict

from .medec import MEDECDataLoader

try:
    import streamlit as st

    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


class MEDECScreeningDataLoader(MEDECDataLoader):
    """MEDEC Screening task data loader for quality assessment."""

    def get_task_name(self) -> str:
        return "medec_screening"

    def _discover_datasets(self) -> List[str]:
        """Discover datasets when no config is available."""
        datasets = set()

        # Check results directory
        medec_screening_results = self.results_base / "medec_screening"
        if medec_screening_results.exists():
            for dataset_dir in medec_screening_results.iterdir():
                if dataset_dir.is_dir() and not dataset_dir.name.startswith("."):
                    datasets.add(dataset_dir.name)

        return sorted(list(datasets))

    def get_filter_options(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get MEDEC Screening-specific filter options."""
        error_types = list(set(s["error_type"] or "None" for s in samples))
        error_types.sort()

        return {
            "categories": error_types,
            "category_label": "Error Type",
            "binary_status_options": ["Valid Problems", "Invalid Problems"],
            "binary_status_label": "Show problems",
        }

    def get_sort_options(self) -> List[Dict[str, str]]:
        """Get MEDEC Screening-specific sort options."""
        return [
            {"label": "Original Order (Default)", "key": "original_order"},
            {"label": "Sample ID (A-Z)", "key": "sample_id"},
            {"label": "Validity Score (High to Low)", "key": "validity_desc"},
            {"label": "Validity Score (Low to High)", "key": "validity_asc"},
            {"label": "Total Issues (High to Low)", "key": "total_issues_desc"},
            {"label": "Total Issues (Low to High)", "key": "total_issues_asc"},
        ]

    def apply_filters(
        self, samples: List[Dict[str, Any]], filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply MEDEC Screening-specific filters to samples."""
        filtered = samples

        # Apply binary status filter for problem validity
        if "binary_status" in filters and filters["binary_status"]:
            selected_status = filters["binary_status"]
            if len(selected_status) == 1:  # Only one option selected
                if "Valid Problems" in selected_status:
                    filtered = [s for s in filtered if s.get("is_valid", True)]
                elif "Invalid Problems" in selected_status:
                    filtered = [s for s in filtered if not s.get("is_valid", True)]
            # If both selected or none selected, show all (no filtering)

        # Apply category filter (error types) - inherited from parent
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
        """Sort MEDEC Screening samples by specified key."""
        if sort_key == "original_order":
            return samples  # Keep original order
        elif sort_key == "sample_id":
            return sorted(samples, key=lambda x: x.get("sample_id", ""))
        elif sort_key in ["validity_asc", "validity_desc"]:
            # Sort by validity-related metrics from model results
            cache_key = self._get_screening_cache_key(
                task, dataset_name, template_name, selected_models, "validity"
            )
            if not self._is_screening_cached(cache_key):
                self._compute_screening_metrics(
                    task=task,
                    dataset_name=dataset_name,
                    template_name=template_name,
                    selected_models=selected_models,
                    samples=samples,
                )

            cached_scores = self._get_cached_screening_scores(cache_key)
            samples_with_scores = []
            for sample in samples:
                sample_id = sample["sample_id"]
                score = cached_scores.get(sample_id, 0.0)
                samples_with_scores.append((sample, score))

            reverse_order = sort_key.endswith("_desc")
            samples_with_scores.sort(
                key=lambda x: (x[1], x[0]["sample_id"]), reverse=reverse_order
            )

            return [item[0] for item in samples_with_scores]
        elif sort_key in ["total_issues_asc", "total_issues_desc"]:
            # Sort by total issues count from model results
            cache_key = self._get_screening_cache_key(
                task, dataset_name, template_name, selected_models, "total_issues"
            )
            if not self._is_screening_cached(cache_key):
                self._compute_screening_metrics(
                    task=task,
                    dataset_name=dataset_name,
                    template_name=template_name,
                    selected_models=selected_models,
                    samples=samples,
                )

            cached_scores = self._get_cached_screening_scores(cache_key)
            samples_with_scores = []
            for sample in samples:
                sample_id = sample["sample_id"]
                score = cached_scores.get(sample_id, 0)
                samples_with_scores.append((sample, score))

            reverse_order = sort_key.endswith("_desc")
            samples_with_scores.sort(
                key=lambda x: (x[1], x[0]["sample_id"]), reverse=reverse_order
            )

            return [item[0] for item in samples_with_scores]

        return samples

    def _get_screening_cache_key(
        self,
        task: str,
        dataset_name: str,
        template_name: str,
        selected_models: List[str],
        metric_type: str,
    ) -> str:
        """Generate cache key for screening metrics."""
        if not task or not dataset_name or not template_name or not selected_models:
            return ""
        models_str = "-".join(sorted(selected_models))
        return f"{task}:{dataset_name}:{template_name}:{models_str}:{metric_type}"

    def _is_screening_cached(self, cache_key: str) -> bool:
        """Check if screening metrics are cached."""
        # Check session state first
        if HAS_STREAMLIT and hasattr(st, "session_state"):
            session_key = f"medec_screening_cache_{cache_key}"
            if session_key in st.session_state:
                return True

        # Check instance cache
        if not hasattr(self, "_screening_cache"):
            self._screening_cache = {}
        return cache_key in self._screening_cache

    def _get_cached_screening_scores(self, cache_key: str) -> Dict[str, float]:
        """Get cached screening scores."""
        # Try session state first
        if HAS_STREAMLIT and hasattr(st, "session_state"):
            session_key = f"medec_screening_cache_{cache_key}"
            if session_key in st.session_state:
                return st.session_state[session_key]

        # Fall back to instance cache
        if not hasattr(self, "_screening_cache"):
            self._screening_cache = {}
        return self._screening_cache.get(cache_key, {})

    def _set_screening_cache(self, cache_key: str, scores: Dict[str, float]):
        """Set screening scores cache."""
        # Set in instance cache
        if not hasattr(self, "_screening_cache"):
            self._screening_cache = {}
        self._screening_cache[cache_key] = scores

        # Set in session state if available
        if HAS_STREAMLIT and hasattr(st, "session_state"):
            session_key = f"medec_screening_cache_{cache_key}"
            st.session_state[session_key] = scores

    def _compute_screening_metrics(
        self,
        task: str,
        dataset_name: str,
        template_name: str,
        selected_models: List[str],
        samples: List[Dict[str, Any]] = None,
    ) -> None:
        """Compute screening metrics for samples."""
        if not selected_models or not task or not dataset_name or not template_name:
            return

        # Use provided samples or load dataset
        if samples is None:
            dataset = self.load_dataset(dataset_name)
            samples = dataset.get("samples", [])

        # Initialize caches
        validity_scores = {}
        total_issues_scores = {}

        # Load all model results
        for sample in samples:
            sample_id = sample["sample_id"]
            validity_values = []
            total_issues_values = []

            for model in selected_models:
                try:
                    model_results = self.load_model_results(
                        task, dataset_name, model, template_name
                    )
                    if not model_results or "predictions" not in model_results:
                        continue

                    # Look for detailed metrics
                    predictions_data = model_results["predictions"]
                    if (
                        "detailed_metrics" in predictions_data
                        and "screening_judge" in predictions_data["detailed_metrics"]
                    ):
                        detailed_metrics = predictions_data["detailed_metrics"][
                            "screening_judge"
                        ]

                        # Find the sample in detailed metrics
                        for sample_metric in detailed_metrics:
                            if sample_metric.get("sample_id") == sample_id:
                                # Get validity score (1 if valid, 0 if invalid)
                                validity_values.append(
                                    1.0 if sample_metric.get("is_valid", True) else 0.0
                                )
                                # Get total issues count
                                total_issues_values.append(
                                    sample_metric.get("total_issues", 0)
                                )
                                break
                except Exception:
                    continue

            # Calculate average scores across models
            validity_scores[sample_id] = (
                sum(validity_values) / len(validity_values) if validity_values else 0.0
            )
            total_issues_scores[sample_id] = (
                sum(total_issues_values) / len(total_issues_values)
                if total_issues_values
                else 0.0
            )

        # Cache results
        validity_key = self._get_screening_cache_key(
            task, dataset_name, template_name, selected_models, "validity"
        )
        total_issues_key = self._get_screening_cache_key(
            task, dataset_name, template_name, selected_models, "total_issues"
        )

        self._set_screening_cache(validity_key, validity_scores)
        self._set_screening_cache(total_issues_key, total_issues_scores)

    def get_sample_summary_fields(
        self, sample: Dict[str, Any], **kwargs
    ) -> List[Dict[str, str]]:
        """Get MEDEC Screening sample summary fields."""
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

        # Add screening-specific fields if model data is provided
        task = kwargs.get("task")
        dataset_name = kwargs.get("dataset_name")
        template_name = kwargs.get("template_name")
        selected_models = kwargs.get("selected_models")

        if task and dataset_name and template_name and selected_models:
            try:
                sample_id = sample["sample_id"]

                # Get individual evaluation scores for each model
                evaluation_scores = self._get_individual_evaluation_scores(
                    sample_id, task, dataset_name, template_name, selected_models
                )

                if evaluation_scores:
                    # Add overall validity
                    avg_validity = sum(
                        s.get("is_valid", 0) for s in evaluation_scores.values()
                    ) / len(evaluation_scores)
                    fields.append(
                        {
                            "label": "Problem Validity",
                            "value": f"{avg_validity:.1%}",
                            "order": 4,
                            "category": "screening",
                        }
                    )

                    # Add individual evaluation categories
                    # Core criteria (always present)
                    core_eval_categories = [
                        ("ambiguous_error", "Ambiguous Error"),
                        ("multiple_errors", "Multiple Errors"),
                        ("numerical_error", "Numerical Error"),
                    ]

                    # Language-specific criteria
                    language_eval_categories = [
                        ("extra_elements", "Extra Elements"),
                        ("synthesis_consistency_error", "Synthesis Consistency Error"),
                        ("unrealistic_scenario", "Unrealistic Scenario"),
                        ("inconsistent_context", "Inconsistent Context"),
                    ]

                    # Combine categories - prioritize core ones
                    eval_categories = core_eval_categories + language_eval_categories

                    order = 5
                    for eval_key, eval_label in eval_categories:
                        avg_score = sum(
                            s.get(eval_key, 0) for s in evaluation_scores.values()
                        ) / len(evaluation_scores)
                        fields.append(
                            {
                                "label": eval_label,
                                "value": f"{avg_score:.0f}",
                                "order": order,
                                "category": "screening_details",
                            }
                        )
                        order += 1

                    # Add total issues
                    avg_total_issues = sum(
                        s.get("total_issues", 0) for s in evaluation_scores.values()
                    ) / len(evaluation_scores)
                    fields.append(
                        {
                            "label": "Total Issues",
                            "value": f"{avg_total_issues:.1f}",
                            "order": order,
                            "category": "screening",
                        }
                    )
            except Exception:
                # Silently skip if screening metrics calculation fails
                pass

        return fields

    def _get_individual_evaluation_scores(
        self,
        sample_id: str,
        task: str,
        dataset_name: str,
        template_name: str,
        selected_models: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Get individual evaluation scores for a sample from all models."""
        evaluation_scores = {}

        for model in selected_models:
            try:
                model_results = self.load_model_results(
                    task, dataset_name, model, template_name
                )
                if not model_results or "predictions" not in model_results:
                    continue

                # Look for detailed metrics
                predictions_data = model_results["predictions"]
                if (
                    "detailed_metrics" in predictions_data
                    and "screening_judge" in predictions_data["detailed_metrics"]
                ):
                    detailed_metrics = predictions_data["detailed_metrics"][
                        "screening_judge"
                    ]

                    # Find the sample in detailed metrics
                    for sample_metric in detailed_metrics:
                        if sample_metric.get("sample_id") == sample_id:
                            evaluation_scores[model] = {
                                "is_valid": 1
                                if sample_metric.get("is_valid", True)
                                else 0,
                                # Core criteria (always present)
                                "ambiguous_error": sample_metric.get(
                                    "ambiguous_error", 0
                                ),
                                "multiple_errors": sample_metric.get(
                                    "multiple_errors", 0
                                ),
                                "numerical_error": sample_metric.get(
                                    "numerical_error", 0
                                ),
                                # Language-specific criteria (may not be present)
                                "extra_elements": sample_metric.get(
                                    "extra_elements", 0
                                ),
                                "synthesis_consistency_error": sample_metric.get(
                                    "synthesis_consistency_error", 0
                                ),
                                "unrealistic_scenario": sample_metric.get(
                                    "unrealistic_scenario", 0
                                ),
                                "inconsistent_context": sample_metric.get(
                                    "inconsistent_context", 0
                                ),
                                "total_issues": sample_metric.get("total_issues", 0),
                                "explanation": sample_metric.get("explanation", ""),
                            }
                            break
            except Exception:
                continue

        return evaluation_scores

    def analyze_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze MEDEC Screening dataset statistics."""
        samples = dataset["samples"]

        # Error type distribution (inherited from parent)
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

    def get_dataset_statistics(self, dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get MEDEC Screening dataset statistics."""
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
        """Get MEDEC Screening error distribution."""
        stats = self.analyze_dataset(dataset)
        if stats["error_types"]:
            return {"title": "Error Type Distribution", "data": stats["error_types"]}
        return None

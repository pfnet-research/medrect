"""Main OutputSaver class that provides a unified interface."""

from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from loguru import logger

from .io_handlers import save_json


class OutputSaver:
    """Main facade class for saving evaluation results in various formats."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self._session_timestamp = None
        self._session_context = None

    def get_session_timestamp(self) -> str:
        """Get the current session timestamp, creating one if it doesn't exist.

        Returns:
            Session timestamp string in format YYYYMMDD_HHMMSS
        """
        if self._session_timestamp is None:
            self._session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._session_timestamp

    def update_session_timestamp(self, timestamp: Optional[datetime] = None) -> None:
        """Update the session timestamp.

        Args:
            timestamp: Optional datetime to use. If None, uses current time.
        """
        if timestamp is None:
            timestamp = datetime.now()
        self._session_timestamp = timestamp.strftime("%Y%m%d_%H%M%S")

    def start_session(
        self,
        task_name: str,
        model_name: str,
        template_name: str,
        dataset_name: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Start a new session with experiment context.

        Args:
            task_name: Name of the task
            model_name: Name of the model
            template_name: Name of the template
            dataset_name: Name of the dataset
            timestamp: Optional timestamp to use. If None, uses current time.
        """
        if timestamp is None:
            timestamp = datetime.now()
        self._session_timestamp = timestamp.strftime("%Y%m%d_%H%M%S")
        self._session_context = {
            "task_name": task_name,
            "model_name": model_name,
            "template_name": template_name,
            "dataset_name": dataset_name,
        }

    def get_session_dir(self) -> Path:
        """Get the current session directory path.

        Returns:
            Path to the session directory

        Raises:
            ValueError: If session has not been started
        """
        if not self._session_context:
            raise ValueError("Session not started. Call start_session() first.")

        return (
            self.output_dir
            / self._session_context["task_name"]
            / self._session_context["dataset_name"]
            / self._session_context["model_name"]
            / self._session_context["template_name"]
            / self._session_timestamp
        )

    def save_raw_responses(
        self,
        samples: List[Any],
        messages_batch: List[List[Dict[str, str]]],
        outputs: List[Any],
        model_config: Dict[str, Any],
        dataset_config: Dict[str, Any],
    ) -> str:
        """Save raw responses to file."""
        save_dir = self.get_session_dir()
        save_dir.mkdir(parents=True, exist_ok=True)

        raw_path = save_dir / "raw_responses.json"
        # Prepare safe model config
        safe_model_config = self._prepare_safe_model_config(model_config)

        # Build raw responses data directly
        interactions = []

        for i, (sample, messages, output) in enumerate(
            zip(samples, messages_batch, outputs)
        ):
            # Convert sample to dict if it's a custom object
            if hasattr(sample, "__dict__"):
                sample_data = sample.__dict__
            elif hasattr(sample, "to_dict"):
                sample_data = sample.to_dict()
            else:
                sample_data = sample

            # Process output to handle different formats
            if isinstance(output, dict):
                # Preserve full dict structure (from existing raw_responses or model API)
                processed_output = output
            else:
                # Convert simple string/content to dict format
                processed_output = {"content": str(output)}

            interaction = {
                "interaction_index": str(i + 1).zfill(3),
                "sample_data": sample_data,
                "messages": messages,
                "raw_response": processed_output,
                "response_metadata": {
                    "model": self._session_context["model_name"],
                    "template": self._session_context["template_name"],
                    "timestamp": self._session_timestamp,
                },
            }
            interactions.append(interaction)

        raw_responses_data = {
            "metadata": {
                "task": self._session_context["task_name"],
                "model": self._session_context["model_name"],
                "template": self._session_context["template_name"],
                "dataset": self._session_context["dataset_name"],
                "timestamp": self._session_timestamp,
                "num_samples": len(interactions),
            },
            "model_config": safe_model_config,
            "dataset_config": dataset_config,
            "interactions": interactions,
        }

        saved_path = save_json(raw_responses_data, raw_path, add_timestamp=False)

        logger.info(f"Saved raw responses ({len(samples)} samples): {saved_path}")
        return saved_path

    def save_predictions(
        self,
        samples: List[Any],
        predictions: Dict[str, List[Any]],
        metric_results: Dict[str, Any],
        detailed_results: Dict[str, List[Dict[str, Any]]],
        metrics_config: Dict[str, Dict[str, Any]],
    ) -> str:
        """Save predictions and detailed analysis."""
        save_dir = self.get_session_dir()
        save_dir.mkdir(parents=True, exist_ok=True)

        # Convert samples to serializable format and add predictions
        serializable_samples = []
        for i, sample in enumerate(samples):
            if hasattr(sample, "__dict__"):
                sample_dict = sample.__dict__.copy()
            elif hasattr(sample, "to_dict"):
                sample_dict = sample.to_dict().copy()
            else:
                sample_dict = sample.copy() if isinstance(sample, dict) else sample

            # Add predictions for this sample (only if predictions exist)
            if predictions is not None:
                sample_predictions = {}
                for parser_name, parser_results in predictions.items():
                    if i < len(parser_results):
                        sample_predictions[parser_name] = parser_results[i]
                sample_dict["predictions"] = sample_predictions
            else:
                sample_dict["predictions"] = {}

            serializable_samples.append(sample_dict)

        # Construct predictions_data with optimized key ordering for usability
        predictions_data = {
            "metadata": {
                "timestamp": self._session_timestamp,
                "task": self._session_context["task_name"],
                "model": self._session_context["model_name"],
                "template": self._session_context["template_name"],
                "dataset": self._session_context["dataset_name"],
                "num_samples": len(samples),
            },
            "metric_breakdowns": metric_results,
            "metrics_config": metrics_config,
            "detailed_metrics": detailed_results,
            "samples": serializable_samples,
        }

        predictions_path = save_dir / "predictions.json"
        saved_path = save_json(predictions_data, predictions_path, add_timestamp=False)

        logger.info(
            f"Saved predictions and metrics ({len(samples)} samples): {saved_path}"
        )

        return saved_path

    def _prepare_safe_model_config(
        self, model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare model config with secret parameters excluded."""
        if hasattr(model_config, "get_safe_config"):
            return model_config.get_safe_config()
        else:
            # Fallback: manually exclude secret section
            return {k: v for k, v in model_config.items() if k != "secret"}

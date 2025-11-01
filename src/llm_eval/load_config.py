from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
import pprint
import copy

from loguru import logger


class ConfigLoader:
    """Manages configuration loading and default value resolution."""

    @staticmethod
    def load_batch_config(config_path: str) -> Dict[str, Any]:
        """Load batch configuration from YAML file."""
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded batch config from {config_path}")
        return config

    @staticmethod
    def load_task_configs(
        task_name: str,
        dataset_names: Optional[List[str]] = None,
        template_names: Optional[List[str]] = None,
        metric_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Load experiment-specific configuration for a single task with resolved defaults.

        Args:
            task_name: Single task name
            dataset_names: List of dataset names (uses defaults if None)
            template_names: List of template names (uses defaults if None)
            metric_names: List of metric names (uses defaults if None)

        Returns:
            dict: Experiment-specific configuration
        """
        # Load full YAML configuration for single task
        task_config_path = f"configs/tasks/{task_name}.yaml"
        with open(task_config_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f) or {}
        logger.info(f"Loaded task config from {task_config_path}")

        # Resolve defaults for this task
        default_dataset_names = dataset_names or [full_config["defaults"]["dataset"]]
        default_template_names = template_names or [full_config["defaults"]["template"]]
        default_metric_names = metric_names or full_config["defaults"]["metrics"]

        logger.info(
            f"Task {task_name} - datasets={default_dataset_names}, templates={default_template_names}, metrics={default_metric_names}"
        )

        # Build datasets section with all requested datasets
        datasets_section = {}
        data_dir = full_config["data"]["dir"]
        for dataset_name in default_dataset_names:
            dataset_config = full_config["data"]["datasets"][dataset_name]
            # Build full path by combining data_dir and dataset file
            from pathlib import Path

            dataset_path = str(Path(data_dir) / dataset_config["file"])
            datasets_section[dataset_name] = {
                "path": dataset_path,
                "loader": dataset_config["loader"],
            }

        # Filter selected metrics using new format
        selected_metrics = {}
        metrics_configs = full_config["metrics"]
        for metric_name in default_metric_names:
            if metric_name in metrics_configs:
                selected_metrics[metric_name] = metrics_configs[metric_name].copy()
            else:
                logger.warning(f"Metric '{metric_name}' not found in metrics")

        # Create configuration for this task
        task_config = {
            "task_name": task_name,
            "components": full_config["components"],
            "datasets": datasets_section,
            "templates": default_template_names,
            "metrics": selected_metrics,
        }

        # Log the structured result
        logger.info(
            f"Task {task_name} config loaded:\n{pprint.pformat(task_config, width=80, depth=3)}"
        )

        # Return the configuration directly
        return task_config

    @staticmethod
    def load_model_configs(model_names: List[str]) -> List[tuple[Dict[str, Any], str]]:
        """Load configurations for multiple models.

        Args:
            model_names: List of model names to load

        Returns:
            List[tuple]: List of (model_config, model_type) tuples
        """
        # Load configurations for all models
        models_dir = Path("configs/models")
        results = []

        for model_name in model_names:
            model_config = None
            model_type = None

            # Search for the model in all model config files
            for model_file in models_dir.glob("*.yaml"):
                with open(model_file, "r", encoding="utf-8") as f:
                    all_models = yaml.safe_load(f) or {}
                    if model_name in all_models:
                        model_config = all_models[model_name]
                        model_type = model_file.stem
                        logger.info(
                            f"Loaded model config for '{model_name}' from {model_file}"
                        )
                        break

            if model_config is None:
                raise ValueError(f"Model '{model_name}' not found in any config files")

            # Log the structured result with secret masking
            safe_config = ConfigLoader._mask_secrets_in_config(model_config)
            logger.info(
                f"Model config loaded:\n{pprint.pformat({'model_name': model_name, 'model_type': model_type, 'config': safe_config}, width=80, depth=3)}"
            )

            results.append((model_config, model_type))

        return results

    @staticmethod
    def _mask_secrets_in_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Mask secret section in config for safe logging."""
        safe_config = copy.deepcopy(config)

        # Simply mask the 'secret' section if it exists
        if "secret" in safe_config:
            safe_config["secret"] = "***MASKED***"

        return safe_config

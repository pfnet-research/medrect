"""Configuration loader that uses existing llm_eval config files."""

import yaml
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from loguru import logger


@dataclass
class TaskConfig:
    """Configuration for a specific task."""

    name: str
    display_name: str
    data_dir: str
    datasets: Dict[str, Dict[str, str]]  # dataset_name -> {file, loader}
    metrics: List[Dict[str, str]]  # [{name, parser}, ...]
    defaults: Dict[str, Any]


@dataclass
class ViewerConfig:
    """Runtime configuration for the viewer."""

    available_tasks: List[str]
    task_configs: Dict[str, TaskConfig]
    available_models: Dict[str, List[str]]  # task -> models
    available_datasets: Dict[str, List[str]]  # task -> datasets
    available_templates: Dict[str, List[str]]  # task -> templates
    output_dirs: Dict[str, List[str]]  # task -> list of output_dirs


class ConfigValidator:
    """Validates configuration data."""

    @staticmethod
    def validate_task_config(config: Dict[str, Any], task_name: str) -> List[str]:
        """Validate task configuration and return list of errors."""
        errors = []

        if not isinstance(config, dict):
            errors.append(f"Task {task_name}: Configuration must be a dictionary")
            return errors

        # Check required fields exist and have correct types
        data_section = config.get("data", {})
        if not isinstance(data_section, dict):
            errors.append(f"Task {task_name}: 'data' section must be a dictionary")

        # Handle metrics - must be dict format
        metrics_section = config.get("metrics", {})
        if not isinstance(metrics_section, dict):
            errors.append(f"Task {task_name}: 'metrics' section must be a dictionary")
        else:
            # Validate dict format
            for metric_name, metric_config in metrics_section.items():
                if not isinstance(metric_config, dict):
                    errors.append(
                        f"Task {task_name}: metric '{metric_name}' must be a dictionary"
                    )
                    continue
                # Validate required fields
                if "class" not in metric_config:
                    errors.append(
                        f"Task {task_name}: metric '{metric_name}' missing 'class' field"
                    )

        return errors

    @staticmethod
    def validate_batch_config(config: Dict[str, Any], filename: str) -> List[str]:
        """Validate batch configuration and return list of errors."""
        errors = []

        if not isinstance(config, dict):
            errors.append(f"Batch {filename}: Configuration must be a dictionary")
            return errors

        if "task" not in config:
            errors.append(f"Batch {filename}: Missing 'task' field")

        # Validate list fields
        for field in ["models", "datasets", "templates"]:
            if field in config and not isinstance(config[field], list):
                errors.append(f"Batch {filename}: '{field}' must be a list")

        return errors


class FileReader:
    """Handles file reading operations."""

    @staticmethod
    def read_yaml_file(file_path: Path) -> Optional[Dict[str, Any]]:
        """Read and parse a YAML file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
            return content if content is not None else {}
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in {file_path}: {e}")
            return None
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"Cannot read file {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error reading {file_path}: {e}")
            return None


class ConfigLoader:
    """Loads configuration from existing llm_eval config files."""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.configs_path = self.base_path / "configs"
        self.tasks_path = self.configs_path / "tasks"
        self.batch_path = self.configs_path / "batch"

        # Cache management
        self._cache: Optional[ViewerConfig] = None
        self._cache_timestamp: float = 0
        self._cache_duration: float = 300.0  # 5 minutes

        # Helper instances
        self._file_reader = FileReader()
        self._validator = ConfigValidator()

    def load_viewer_config(self, force_reload: bool = False) -> ViewerConfig:
        """Load complete viewer configuration from existing files."""
        current_time = time.time()

        # Check cache validity
        if not force_reload and self._is_cache_valid(current_time):
            logger.debug("Using cached configuration")
            return self._cache

        logger.debug("Loading configuration from files")

        try:
            # Load task configurations
            task_configs = self._load_task_configs()

            # Load batch configurations to get available models/datasets/templates
            batch_info = self._load_batch_configs()

            config = ViewerConfig(
                available_tasks=list(task_configs.keys()),
                task_configs=task_configs,
                available_models=batch_info.get("models", {}),
                available_datasets=batch_info.get("datasets", {}),
                available_templates=batch_info.get("templates", {}),
                output_dirs=batch_info.get("output_dirs", {}),
            )

            # Update cache
            self._cache = config
            self._cache_timestamp = current_time

            return config

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            if self._cache is not None:
                logger.warning("Returning cached configuration due to error")
                return self._cache
            raise

    def _load_task_configs(self) -> Dict[str, TaskConfig]:
        """Load all task configurations from configs/tasks/."""
        task_configs = {}

        if not self.tasks_path.exists():
            logger.warning(f"Tasks directory not found: {self.tasks_path}")
            return task_configs

        for task_file in self.tasks_path.glob("*.yaml"):
            task_name = task_file.stem

            config = self._file_reader.read_yaml_file(task_file)
            if config is None:
                continue

            if not config:
                logger.warning(f"Empty configuration file: {task_file}")
                continue

            # Validate configuration
            validation_errors = self._validator.validate_task_config(config, task_name)
            if validation_errors:
                logger.error(f"Validation errors in {task_file}: {validation_errors}")
                continue

            try:
                # Handle metrics - convert dict to list format
                raw_metrics = config.get("metrics", [])
                if isinstance(raw_metrics, dict):
                    # Convert dict format to list format
                    metrics = []
                    for metric_name, metric_config in raw_metrics.items():
                        metric_entry = {"name": metric_name}
                        metric_entry.update(metric_config)
                        metrics.append(metric_entry)
                else:
                    metrics = raw_metrics

                task_config = TaskConfig(
                    name=task_name,
                    display_name=config.get("display_name", task_name.upper()),
                    data_dir=config.get("data", {}).get(
                        "dir", f"data/inputs/{task_name}"
                    ),
                    datasets=config.get("data", {}).get("datasets", {}),
                    metrics=metrics,
                    defaults=config.get("defaults", {}),
                )

                task_configs[task_name] = task_config
                logger.debug(f"Loaded task config: {task_name}")

            except Exception as e:
                logger.error(f"Error creating TaskConfig for {task_name}: {e}")
                continue

        return task_configs

    def _load_batch_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load information from batch configurations."""
        batch_info = {"models": {}, "datasets": {}, "templates": {}, "output_dirs": {}}

        if not self.batch_path.exists():
            logger.warning(f"Batch directory not found: {self.batch_path}")
            return batch_info

        # Group by task
        task_models = {}
        task_datasets = {}
        task_templates = {}
        task_output_dirs = {}

        for batch_file in self.batch_path.glob("*.yaml"):
            config = self._file_reader.read_yaml_file(batch_file)
            if config is None:
                continue

            if not config:
                logger.warning(f"Empty batch configuration file: {batch_file}")
                continue

            # Validate configuration
            validation_errors = self._validator.validate_batch_config(
                config, batch_file.name
            )
            if validation_errors:
                logger.error(f"Validation errors in {batch_file}: {validation_errors}")
                continue

            task_field = config.get("task")
            if not task_field:
                logger.warning(f"No task specified in batch config: {batch_file}")
                continue

            # Handle both string and list format for task field
            if isinstance(task_field, list):
                if not task_field:
                    logger.warning(f"Empty task list in batch config: {batch_file}")
                    continue
                task = task_field[0]  # Use first task from list
                if len(task_field) > 1:
                    logger.warning(
                        f"Multiple tasks in batch config {batch_file}, using first: {task}"
                    )
            else:
                task = task_field

            # Collect models, datasets, templates per task
            if task not in task_models:
                task_models[task] = set()
                task_datasets[task] = set()
                task_templates[task] = set()

            if "models" in config and isinstance(config["models"], list):
                task_models[task].update(config["models"])

            if "datasets" in config and isinstance(config["datasets"], list):
                task_datasets[task].update(config["datasets"])

            if "templates" in config and isinstance(config["templates"], list):
                task_templates[task].update(config["templates"])

            # Store output directory (support multiple output_dirs per task)
            if "output_dir" in config:
                if task not in task_output_dirs:
                    task_output_dirs[task] = set()
                task_output_dirs[task].add(config["output_dir"])

            logger.debug(f"Loaded batch config for task {task}: {batch_file}")

        # Convert sets to sorted lists
        batch_info["models"] = {
            task: sorted(list(models)) for task, models in task_models.items()
        }
        batch_info["datasets"] = {
            task: sorted(list(datasets)) for task, datasets in task_datasets.items()
        }
        batch_info["templates"] = {
            task: sorted(list(templates)) for task, templates in task_templates.items()
        }
        # Convert sets to sorted lists for output_dirs as well
        batch_info["output_dirs"] = {
            task: sorted(list(output_dirs))
            for task, output_dirs in task_output_dirs.items()
        }

        return batch_info

    def _is_cache_valid(self, current_time: float) -> bool:
        """Check if cache is still valid."""
        return (
            self._cache is not None
            and current_time - self._cache_timestamp < self._cache_duration
        )

    def invalidate_cache(self) -> None:
        """Manually invalidate the cache."""
        self._cache = None
        self._cache_timestamp = 0
        logger.debug("Configuration cache invalidated")

    def get_task_config(self, task_name: str) -> Optional[TaskConfig]:
        """Get configuration for a specific task."""
        config = self.load_viewer_config()
        return config.task_configs.get(task_name)

    def get_available_tasks_from_results(self) -> List[str]:
        """Get tasks that actually have results available."""
        available_tasks = []

        # Get output directories from batch configurations
        viewer_config = self.load_viewer_config()
        output_dirs = viewer_config.output_dirs

        # Default results directory
        results_paths = [self.base_path / "results"]

        # Add output directories from batch configs
        for task, output_dir_list in output_dirs.items():
            for output_dir in output_dir_list:
                output_path = self.base_path / output_dir
                if output_path not in results_paths:
                    results_paths.append(output_path)

        # Scan each output directory
        for results_path in results_paths:
            if not results_path.exists():
                logger.debug(f"Output directory not found: {results_path}")
                continue

            try:
                for task_dir in results_path.iterdir():
                    if not task_dir.is_dir() or task_dir.name.startswith("."):
                        continue

                    try:
                        # Check if task has configuration
                        task_config_file = self.tasks_path / f"{task_dir.name}.yaml"
                        if not task_config_file.exists():
                            logger.debug(
                                f"No config found for task directory: {task_dir.name}"
                            )
                            continue

                        # Check if has actual results
                        has_results = any(
                            dataset_dir.is_dir() and any(dataset_dir.iterdir())
                            for dataset_dir in task_dir.iterdir()
                            if dataset_dir.is_dir()
                        )

                        if has_results and task_dir.name not in available_tasks:
                            available_tasks.append(task_dir.name)
                            logger.debug(
                                f"Found available task: {task_dir.name} in {results_path}"
                            )

                    except Exception as e:
                        logger.warning(
                            f"Error processing task directory {task_dir}: {e}"
                        )
                        continue

            except Exception as e:
                logger.error(f"Error scanning output directory {results_path}: {e}")

        return sorted(available_tasks)

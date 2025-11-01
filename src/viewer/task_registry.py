"""Task registry with dynamic discovery following llm_eval ComponentFactory pattern."""

import importlib
import inspect
from pathlib import Path
from typing import Type, Tuple, Dict, List, Any
from types import ModuleType

from loguru import logger

from .task_specific.data_loader.base import BaseDataLoader
from .task_specific.metrics_parser.base import BaseMetricsParser
from .task_specific.response_formatter.base import BaseResponseFormatter
from .config_loader import ConfigLoader


class TaskRegistryImpl:
    """Registry for task-specific components with dynamic discovery."""

    # Component configuration: type -> (directory_name, base_class, suffix)
    COMPONENT_CONFIG = {
        "loaders": ("data_loader", BaseDataLoader, "DataLoader"),
        "parsers": ("metrics_parser", BaseMetricsParser, "MetricsParser"),
        "formatters": (
            "response_formatter",
            BaseResponseFormatter,
            "ResponseFormatter",
        ),
    }

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self._config_loader = ConfigLoader(base_path)
        self._viewer_config = self._config_loader.load_viewer_config()

        # Dynamic component registry
        self._components = {comp_type: {} for comp_type in self.COMPONENT_CONFIG}
        self._discover_all_tasks()

    def _discover_all_tasks(self) -> None:
        """Discover all task implementations following ComponentFactory pattern."""
        tasks_module_path = Path(__file__).parent / "task_specific"

        if not tasks_module_path.exists():
            logger.warning("No task_specific directory found")
            return

        # Import task_specific module dynamically
        try:
            from . import task_specific as tasks_module

            self._discover_components_in_module(tasks_module, tasks_module_path)
            logger.info(
                f"Task discovery completed. Found tasks: {list(self.get_discovered_tasks())}"
            )
        except ImportError as e:
            logger.error(f"Failed to import task_specific module: {e}")

    def _discover_components_in_module(
        self, module: ModuleType, module_path: Path
    ) -> None:
        """Discover components in the directory structure."""

        # Scan each component type directory based on COMPONENT_CONFIG
        for comp_type, (
            comp_dir_name,
            base_class,
            suffix,
        ) in self.COMPONENT_CONFIG.items():
            comp_dir_path = module_path / comp_dir_name
            if not comp_dir_path.exists():
                continue

            # Scan task implementation files in each component directory
            for py_file in comp_dir_path.glob("*.py"):
                if py_file.name.startswith("__") or py_file.stem == "base":
                    continue

                task_name = py_file.stem  # medec.py -> medec

                # Build module name: src.viewer.task_specific.data_loader.medec
                relative_path = py_file.relative_to(module_path).with_suffix("")
                module_name = f"{module.__name__}.{'.'.join(relative_path.parts)}"

                try:
                    imported_module = importlib.import_module(module_name)
                    self._discover_type_in_module(
                        comp_type, task_name, imported_module, base_class, suffix
                    )
                    logger.debug(f"Successfully imported {module_name}")

                except Exception as e:
                    logger.debug(f"Failed to import {module_name}: {e}")

    def _discover_type_in_module(
        self,
        component_type: str,
        task_name: str,
        module: ModuleType,
        base_class: Type,
        suffix: str,
    ) -> None:
        """Discover components of a specific type in a module."""

        for name, obj in inspect.getmembers(module):
            if self._is_valid_component(obj, base_class, name, suffix):
                # Use task name as key (medec.py -> medec)
                self._components[component_type][task_name] = obj
                logger.debug(
                    f"Registered {component_type}: {task_name} -> {obj.__name__}"
                )

    def _is_valid_component(
        self, obj, base_class: Type, name: str, expected_suffix: str
    ) -> bool:
        """Check if an object is a valid component class."""
        return (
            inspect.isclass(obj)
            and issubclass(obj, base_class)
            and obj != base_class
            and not name.startswith("_")
            and name.endswith(expected_suffix)
        )

    def get_discovered_tasks(self) -> set:
        """Get set of tasks that have all required components."""
        if not self.COMPONENT_CONFIG:
            return set()

        # Get tasks that have ALL component types
        component_type_keys = list(self.COMPONENT_CONFIG.keys())
        tasks_per_type = [
            set(self._components[comp_type].keys()) for comp_type in component_type_keys
        ]

        # Return intersection - tasks that have ALL component types
        if tasks_per_type:
            return set.intersection(*tasks_per_type)
        return set()

    def get_task_components(self, task_name: str) -> Tuple[object, ...]:
        """Get all components for a task in the order defined by COMPONENT_CONFIG."""

        discovered_tasks = self.get_discovered_tasks()
        if task_name not in discovered_tasks:
            raise ValueError(
                f"Task '{task_name}' not found or incomplete. "
                f"Available: {sorted(discovered_tasks)}"
            )

        components = []
        task_config = self._viewer_config.task_configs.get(task_name)

        for comp_type in self.COMPONENT_CONFIG.keys():
            component_class = self._components[comp_type][task_name]

            # Instantiate based on component type (loaders need base_path)
            if comp_type == "loaders":
                component = component_class(self.base_path, task_config)
            else:
                component = component_class(task_config)

            components.append(component)

        return tuple(components)

    def get_available_tasks(self) -> list[str]:
        """Get list of tasks that are both discovered AND have results."""
        discovered_tasks = self.get_discovered_tasks()
        available_with_results = set(
            self._config_loader.get_available_tasks_from_results()
        )

        # Return tasks that are discovered AND have results
        return sorted(list(discovered_tasks & available_with_results))

    def get_task_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed status for all tasks including implementation and data availability."""
        discovered_tasks = set(self.get_discovered_tasks())
        tasks_with_results = set(self._config_loader.get_available_tasks_from_results())
        all_configured_tasks = set(self._viewer_config.task_configs.keys())

        status = {}

        for task in all_configured_tasks:
            has_implementation = task in discovered_tasks
            has_results = task in tasks_with_results

            # Determine overall status
            if has_implementation and has_results:
                overall_status = "ready"
            elif not has_implementation and has_results:
                overall_status = "data_only"
            elif has_implementation and not has_results:
                overall_status = "implementation_only"
            else:
                overall_status = "configured_only"

            status[task] = {
                "has_implementation": has_implementation,
                "has_results": has_results,
                "overall_status": overall_status,
                "missing_components": self._get_missing_components(task)
                if not has_implementation
                else [],
            }

        return status

    def _get_missing_components(self, task_name: str) -> List[str]:
        """Get list of missing component types for a task."""
        missing = []

        for comp_type in self.COMPONENT_CONFIG.keys():
            if task_name not in self._components[comp_type]:
                missing.append(comp_type)

        return missing

    def get_task_datasets(self, task_name: str) -> list[str]:
        """Get available datasets for a task from config."""
        return self._viewer_config.available_datasets.get(task_name, [])

    def get_task_models(self, task_name: str) -> list[str]:
        """Get available models for a task from config."""
        return self._viewer_config.available_models.get(task_name, [])

    def get_task_templates(self, task_name: str, dataset_name: str) -> list[str]:
        """Get available templates for a task/dataset combination from results directory."""
        templates = set()

        # Get all possible output directories for this task
        task_output_dirs = self._viewer_config.output_dirs.get(task_name, ["results"])

        # Try each output directory
        for output_dir in task_output_dirs:
            results_path = self.base_path / output_dir / task_name / dataset_name

            if not results_path.exists():
                continue

            # Scan each model directory for template subdirectories
            for model_dir in results_path.iterdir():
                if not model_dir.is_dir():
                    continue

                # Each subdirectory in the model dir represents a template
                for template_dir in model_dir.iterdir():
                    if template_dir.is_dir():
                        templates.add(template_dir.name)

        return sorted(list(templates))


# Global registry instances per base_path
_registry_instances: dict[str, TaskRegistryImpl] = {}


def get_task_registry(base_path: Path) -> TaskRegistryImpl:
    """Get TaskRegistry instance for the given base_path."""
    base_path_str = str(base_path.resolve())
    if base_path_str not in _registry_instances:
        _registry_instances[base_path_str] = TaskRegistryImpl(base_path)
    return _registry_instances[base_path_str]

import importlib
import inspect
from pathlib import Path
from types import ModuleType
from typing import Type, Any

from loguru import logger

from . import models, data, templates, parsers, metrics
from .models.base import BaseModel
from .data.base import BaseLoader, BaseSample
from .templates.base import BaseTemplate
from .parsers.base import BaseParser
from .metrics.base import BaseMetric


class ComponentFactory:
    """Factory for discovering and managing framework components."""

    # Component configuration: type -> (module, base_class, suffix)
    COMPONENT_CONFIG = {
        "models": (models, BaseModel, "Model"),
        "samples": (data, BaseSample, "Sample"),
        "loaders": (data, BaseLoader, "Loader"),
        "templates": (templates, BaseTemplate, "Template"),
        "parsers": (parsers, BaseParser, "Parser"),
        "metrics": (metrics, BaseMetric, "Metric"),
    }

    def __init__(self):
        self._components = {comp_type: {} for comp_type in self.COMPONENT_CONFIG}
        self._discover_all()

    def _discover_all(self) -> None:
        """Discover all available components."""
        for component_type, (module, base_class, _) in self.COMPONENT_CONFIG.items():
            self._discover_type(component_type, module, base_class)

        logger.info("Component discovery completed")

    def _discover_type(
        self, component_type: str, module: ModuleType, base_class: Type
    ) -> None:
        """Discover classes in a module that inherit from base_class."""
        module_path = Path(module.__file__).parent

        # Scan all Python files (both direct and in subdirectories)
        for py_file in module_path.rglob("*.py"):
            if py_file.name.startswith("__"):
                continue

            # Build module name from file path
            relative_path = py_file.relative_to(module_path).with_suffix("")
            module_name = f"{module.__name__}.{'.'.join(relative_path.parts)}"

            try:
                imported_module = importlib.import_module(module_name)

                for name, obj in inspect.getmembers(imported_module):
                    if self._is_valid_component(obj, base_class, name):
                        key = self._get_key_from_class_name(name)
                        self._components[component_type][key] = obj
                        logger.debug(
                            f"Registered {base_class.__name__}: {key} -> {obj}"
                        )
            except Exception as e:
                logger.debug(f"Failed to import {module_name}: {e}")

    def _is_valid_component(self, obj: Any, base_class: Type, name: str) -> bool:
        """Check if an object is a valid component class."""
        return (
            inspect.isclass(obj)
            and issubclass(obj, base_class)
            and obj != base_class
            and not name.startswith("_")
        )

    def _normalize_component_name(self, name: str) -> str:
        """Normalize component name for consistent registration and lookup."""
        return name.replace("_", "").replace("-", "").lower()

    def _get_key_from_class_name(self, class_name: str) -> str:
        """Get normalized registry key from class name."""
        # Get all suffixes from component config
        suffixes = [suffix for _, _, suffix in self.COMPONENT_CONFIG.values()]

        for suffix in suffixes:
            if class_name.endswith(suffix):
                base_name = class_name[: -len(suffix)]
                return self._normalize_component_name(
                    base_name
                )  # Use unified normalization
        return self._normalize_component_name(class_name)

    def get_component_class(self, component_type: str, name: str) -> Type:
        """Get component class without instantiating it.

        Args:
            component_type: Type of component ('models', 'templates', etc.)
            name: Name/key of the specific component

        Returns:
            The component class
        """
        components = self._components[component_type]

        # Use the same normalization logic as registration
        normalized_name = self._normalize_component_name(name)

        # Check if normalized name exists
        if normalized_name not in components:
            available = list(components.keys())
            singular = self.COMPONENT_CONFIG[component_type][2]
            raise ValueError(f"{singular} '{name}' not found. Available: {available}")

        return components[normalized_name]

    def get_component_instance(self, component_type: str, name: str, *args, **kwargs):
        """Get an instance of a component by type and name.

        This is a wrapper around get_component_class that instantiates the class.

        Args:
            component_type: Type of component ('models', 'templates', etc.)
            name: Name/key of the specific component
            *args: Positional arguments for the component constructor
            **kwargs: Keyword arguments for the component constructor

        Returns:
            An instance of the requested component
        """
        component_class = self.get_component_class(component_type, name)
        return component_class(*args, **kwargs)


# Global instance for shared use across the framework
component_factory = ComponentFactory()


def main():
    """Display component factory information for debugging."""
    print("\n=== Component Factory ===")
    for component_type, components in component_factory._components.items():
        print(f"\n{component_type.capitalize()}:")
        for key in sorted(components.keys()):
            print(f"  {key}")


if __name__ == "__main__":
    main()

"""Base template class for message generation."""

from abc import ABC, abstractmethod
from typing import Dict, List, ClassVar

from ..data.base import BaseSample


class BaseTemplate(ABC):
    """Base class for all templates that generate conversation messages."""

    # Abstract class variable - subclasses must define TEMPLATES
    TEMPLATES: ClassVar[Dict[str, str]]

    def __init__(self, template_name: str):
        self.template_name = template_name

    @abstractmethod
    def generate_messages(self, sample: BaseSample) -> List[Dict[str, str]]:
        """Generate conversation messages for the sample.

        Returns:
            List of message dictionaries with 'role' and 'content' keys.
            Example: [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ]
        """
        pass

    def format_template(self, template_string: str, **kwargs) -> str:
        """Format template string with keyword arguments."""
        try:
            return template_string.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")

    @classmethod
    def get_available_templates(cls) -> List[str]:
        """Get list of available template names."""
        return list(cls.TEMPLATES.keys())

    def get_template_string(self) -> str:
        """Get the template string for this template."""
        if self.template_name in self.__class__.TEMPLATES:
            return self.__class__.TEMPLATES[self.template_name]
        raise ValueError(
            f"Template '{self.template_name}' not found in {self.__class__.__name__}"
        )

"""Base parser class for output parsing."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import re



class BaseParser(ABC):
    """Base class for all output parsers."""

    def __init__(self, config: Dict[str, Any]):
        self.parser_name = config.get("parser_name", "unknown")
        self.config = config

    @abstractmethod
    def parse(self, output: str) -> Any:
        """Parse model output and return structured result."""
        pass

    def preprocess_output(self, output: str) -> str:
        """Preprocess output before parsing."""
        return output.strip()

    def extract_text_between(
        self, text: str, start_pattern: str, end_pattern: str
    ) -> Optional[str]:
        """Extract text between two patterns."""
        pattern = f"{start_pattern}(.*?){end_pattern}"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    def extract_first_number(self, text: str) -> Optional[int]:
        """Extract first number from text."""
        match = re.search(r"\d+", text)
        return int(match.group()) if match else None

    def extract_yes_no(self, text: str) -> Optional[int]:
        """Extract yes/no as 1/0."""
        text_lower = text.lower()
        if any(
            word in text_lower
            for word in ["yes", "はい", "あり", "エラー", "error", "1"]
        ):
            return 1
        elif any(
            word in text_lower
            for word in ["no", "いいえ", "なし", "エラーなし", "no error", "0"]
        ):
            return 0
        return None

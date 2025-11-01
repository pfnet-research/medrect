"""Parser for MEDEC error type classification results."""

import json
import re
from typing import Dict, Any, Optional
from loguru import logger

from ..base import BaseParser


class ErrorTypeParser(BaseParser):
    """Parser for MEDEC error type classification JSON responses."""

    # Valid error types for the classification task
    VALID_ERROR_TYPES = {
        "history_taking",
        "physical_findings",
        "test_interpretation",
        "diagnosis",
        "medication_selection",
        "medication_dosage",
        "procedure_intervention",
        "monitoring_management",
        "none",
    }

    # Valid confidence levels
    VALID_CONFIDENCE_LEVELS = {"high", "medium", "low"}

    def __init__(self):
        """Initialize parser."""
        super().__init__({"parser_name": "error_type", "language": "multi"})

    def parse(self, output: str) -> Dict[str, Any]:
        """Parse error type classification response.

        Args:
            output: Raw model response text

        Returns:
            Parsed result dictionary with error_type, confidence, and reasoning
        """
        # Initialize result with defaults
        result = {
            "error_type": None,
            "confidence": None,
            "reasoning": None,
            "parse_error": False,
            "raw_response": output,
        }

        try:
            # Clean and extract JSON
            json_str = self._extract_json(output)
            if not json_str:
                raise ValueError("No JSON content found")

            # Parse JSON
            parsed_json = json.loads(json_str)

            # Validate and extract error_type
            error_type = parsed_json.get("error_type", "").strip().lower()
            if not error_type or error_type not in self.VALID_ERROR_TYPES:
                logger.warning(
                    f"Invalid error_type: {error_type}. Must be one of: {self.VALID_ERROR_TYPES}"
                )
                result["parse_error"] = True
                return result

            # Validate and extract confidence
            confidence = parsed_json.get("confidence", "").strip().lower()
            if not confidence or confidence not in self.VALID_CONFIDENCE_LEVELS:
                logger.warning(
                    f"Invalid confidence: {confidence}. Must be one of: {self.VALID_CONFIDENCE_LEVELS}"
                )
                result["parse_error"] = True
                return result

            # Extract explanation (fallback to reasoning for backward compatibility)
            explanation = parsed_json.get(
                "explanation", parsed_json.get("reasoning", "")
            ).strip()
            if not explanation:
                logger.warning("Missing explanation field")
                result["parse_error"] = True
                return result

            # Set validated values
            result.update(
                {
                    "error_type": error_type,
                    "confidence": confidence,
                    "explanation": explanation,
                    "parse_error": False,
                }
            )

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            result["parse_error"] = True
            return result

        except Exception as e:
            logger.error(f"Unexpected parsing error: {e}")
            result["parse_error"] = True
            return result

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON content from response text.

        Args:
            text: Raw response text

        Returns:
            Extracted JSON string or None if not found
        """
        # Clean text
        text = text.strip()

        # Try to find JSON block between ```json and ```
        json_block_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE
        )
        if json_block_match:
            return json_block_match.group(1).strip()

        # Try to find JSON object in text
        json_match = re.search(r"\{.*?\}", text, re.DOTALL)
        if json_match:
            return json_match.group(0).strip()

        # If no JSON patterns found, try the whole text if it looks like JSON
        if text.startswith("{") and text.endswith("}"):
            return text

        return None

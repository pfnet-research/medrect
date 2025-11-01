"""MEDEC screening judge parser."""

import json
import re
from typing import Dict, Any

from loguru import logger

from ..base import BaseParser


class ScreeningJudgeParser(BaseParser):
    """Parser for MEDEC quality screening judgments."""

    # Define criteria for different template languages
    CRITERIA_JA = [
        "ambiguous_error",
        "extra_elements",
        "multiple_errors",
        "numerical_error",
        "synthesis_consistency_error",
    ]

    CRITERIA_EN = [
        "ambiguous_error",
        "multiple_errors",
        "numerical_error",
        "unrealistic_scenario",
        "inconsistent_context",
    ]

    def __init__(self, language: str = "auto"):
        """Initialize parser with language specification.

        Args:
            language: "ja", "en", or "auto" (auto-detect from output format)
        """
        self.language = language
        super().__init__({"parser_name": "screening_judge", "language": language})

    def _detect_language(self, result: Dict[str, Any]) -> str:
        """Auto-detect language based on available keys in the JSON result."""
        if self.language != "auto":
            return self.language

        # Check which criteria set is present
        has_en_criteria = any(
            key in result for key in ["unrealistic_scenario", "inconsistent_context"]
        )
        has_ja_criteria = any(
            key in result for key in ["extra_elements", "synthesis_consistency_error"]
        )

        if has_en_criteria and not has_ja_criteria:
            return "en"
        elif has_ja_criteria and not has_en_criteria:
            return "ja"
        elif has_en_criteria and has_ja_criteria:
            logger.warning("Mixed criteria detected - defaulting to Japanese")
            return "ja"
        else:
            logger.warning(
                "No language-specific criteria detected - defaulting to Japanese"
            )
            return "ja"

    def parse(self, output: str) -> Dict[str, Any]:
        """Parse output to extract screening judgment results.

        Expected JSON format for Japanese:
        {
            "ambiguous_error": 0,
            "extra_elements": 0,
            "multiple_errors": 0,
            "numerical_error": 0,
            "synthesis_consistency_error": 0,
            "explanation": "explanation"
        }

        Expected JSON format for English:
        {
            "ambiguous_error": 0,
            "multiple_errors": 0,
            "numerical_error": 0,
            "unrealistic_scenario": 0,
            "inconsistent_context": 0,
            "explanation": "explanation"
        }

        Returns:
            Dictionary with screening results and calculated totals
        """
        output = self.preprocess_output(output)

        try:
            # Try to extract JSON from the output
            json_match = re.search(r"\{.*?\}", output, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)

                # Detect language if auto
                detected_language = self._detect_language(result)
                criteria = (
                    self.CRITERIA_JA if detected_language == "ja" else self.CRITERIA_EN
                )

                # Extract explanation (required field)
                explanation = result["explanation"]

                # Extract scores based on detected language
                scores = {}
                total_issues = 0

                for criterion in criteria:
                    if criterion not in result:
                        raise KeyError(f"Missing required criterion: {criterion}")
                    score = self._safe_int(result[criterion])
                    scores[criterion] = score
                    total_issues += score

                # For backward compatibility, ensure all fields exist (set to 0 if not present)
                all_possible_criteria = set(self.CRITERIA_JA + self.CRITERIA_EN)
                for criterion in all_possible_criteria:
                    if criterion not in scores:
                        scores[criterion] = 0

                # Determine if sample is valid (no issues)
                is_valid = total_issues == 0

                return {
                    **scores,
                    "total_issues": total_issues,
                    "is_valid": is_valid,
                    "explanation": explanation,
                    "detected_language": detected_language,
                }
            else:
                logger.warning(f"No JSON found in output: {output[:100]}...")
                return self._parse_error_result()

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}. Output: {output[:100]}...")
            return self._parse_error_result()
        except KeyError as e:
            logger.error(
                f"Missing required key in JSON: {e}. Output: {output[:100]}..."
            )
            return self._parse_error_result()
        except Exception as e:
            logger.error(f"Unexpected error in parsing: {e}. Output: {output[:100]}...")
            return self._parse_error_result()

    def _safe_int(self, value: Any) -> int:
        """Safely convert value to integer."""
        try:
            if isinstance(value, (int, float)):
                return int(value)
            elif isinstance(value, str):
                return int(float(value))
            else:
                return 0
        except (ValueError, TypeError):
            return 0

    def _parse_error_result(self) -> Dict[str, Any]:
        """Return result indicating parsing error."""
        # Include all possible criteria for backward compatibility
        result = {
            "total_issues": -1,
            "is_valid": False,
            "explanation": "Parse error",
            "parse_error": True,
            "detected_language": "unknown",
        }

        # Add all criteria with error values
        all_criteria = set(self.CRITERIA_JA + self.CRITERIA_EN)
        for criterion in all_criteria:
            result[criterion] = -1

        return result

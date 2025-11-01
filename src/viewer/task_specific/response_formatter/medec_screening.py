"""MEDEC Screening-specific response formatter for quality assessment."""

from typing import Dict, List, Any

from .base import BaseResponseFormatter


class MEDECScreeningResponseFormatter(BaseResponseFormatter):
    """MEDEC Screening-specific response formatter for quality assessment."""

    def format_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Format MEDEC Screening prediction for display."""
        # For screening task, the "prediction" is actually the quality assessment result
        # First check if this is a detailed metrics entry (from detailed_metrics.screening_judge)
        if "is_valid" in prediction:
            return {
                "is_valid": prediction.get("is_valid", True),
                "total_issues": prediction.get("total_issues", 0),
                # Core criteria (always present)
                "ambiguous_error": prediction.get("ambiguous_error", 0),
                "multiple_errors": prediction.get("multiple_errors", 0),
                "numerical_error": prediction.get("numerical_error", 0),
                # Language-specific criteria (may not be present)
                "extra_elements": prediction.get("extra_elements", 0),
                "synthesis_consistency_error": prediction.get(
                    "synthesis_consistency_error", 0
                ),
                "unrealistic_scenario": prediction.get("unrealistic_scenario", 0),
                "inconsistent_context": prediction.get("inconsistent_context", 0),
                "explanation": prediction.get("explanation", ""),
                "sample_id": prediction.get("sample_id", ""),
            }

        # For non-screening predictions, return empty format
        return {"sample_id": prediction.get("sample_id", "")}

    def format_ground_truth(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format MEDEC Screening ground truth for display."""
        # Use error_flag if has_error is not present
        has_error = (
            sample["has_error"]
            if "has_error" in sample
            else (sample["error_flag"] == 1)
        )

        return {
            "has_error": has_error,
            "error_sentence": sample["error_sentence"]
            if "error_sentence" in sample
            else "",
            "corrected_sentence": sample["corrected_sentence"]
            if "corrected_sentence" in sample
            else "",
            "error_type": sample["error_type"]
            if "error_type" in sample and sample["error_type"]
            else "None",
        }

    def compare_prediction(
        self, prediction: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare MEDEC Screening prediction with ground truth."""
        # For screening task, we're not comparing predictions vs ground truth in the traditional sense
        # Instead, we're showing the quality assessment results
        formatted_prediction = self.format_prediction(prediction)
        formatted_truth = self.format_ground_truth(ground_truth)

        # If this is a screening assessment result
        if "is_valid" in formatted_prediction:
            return {
                "is_quality_assessment": True,
                "is_valid": formatted_prediction["is_valid"],
                "total_issues": formatted_prediction["total_issues"],
                "issue_breakdown": {
                    # Core criteria (always present)
                    "ambiguous_error": formatted_prediction["ambiguous_error"],
                    "multiple_errors": formatted_prediction["multiple_errors"],
                    "numerical_error": formatted_prediction["numerical_error"],
                    # Language-specific criteria (may not be present)
                    "extra_elements": formatted_prediction.get("extra_elements", 0),
                    "synthesis_consistency_error": formatted_prediction.get(
                        "synthesis_consistency_error", 0
                    ),
                    "unrealistic_scenario": formatted_prediction.get(
                        "unrealistic_scenario", 0
                    ),
                    "inconsistent_context": formatted_prediction.get(
                        "inconsistent_context", 0
                    ),
                },
                "explanation": formatted_prediction["explanation"],
                "sample_info": {
                    "has_error": formatted_truth["has_error"],
                    "error_type": formatted_truth["error_type"],
                },
            }

        # For non-screening predictions, return basic comparison
        return {
            "is_quality_assessment": False,
            "sample_info": {
                "has_error": formatted_truth["has_error"],
                "error_type": formatted_truth["error_type"],
            },
        }

    def format_ground_truth_display(
        self, sample_data: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Format ground truth for display in UI."""
        truth = self.format_ground_truth(sample_data)
        display_items = []

        # Has Error
        display_items.append(
            {"label": "Has Error", "value": "Yes" if truth["has_error"] else "No"}
        )

        if truth["has_error"]:
            # Error type
            if truth["error_type"] and truth["error_type"] != "None":
                display_items.append(
                    {"label": "Error Type", "value": truth["error_type"]}
                )

            # Error sentence ID
            if "error_sentence_id" in sample_data and sample_data["error_sentence_id"]:
                display_items.append(
                    {
                        "label": "Error Sentence ID",
                        "value": str(sample_data["error_sentence_id"]),
                    }
                )

            # Error sentence
            if truth["error_sentence"]:
                display_items.append(
                    {"label": "Error Sentence", "value": truth["error_sentence"]}
                )

            # Corrected sentence
            if truth["corrected_sentence"]:
                display_items.append(
                    {
                        "label": "Corrected Sentence",
                        "value": truth["corrected_sentence"],
                    }
                )

        return display_items

    def format_comparison_display(
        self, comparison: Dict[str, Any], composite_score: float = None
    ) -> Dict[str, Any]:
        """Format comparison for display in UI."""
        display = {"metrics": [], "details": []}

        # Check if this is a quality assessment comparison
        if comparison.get("is_quality_assessment", False):
            # Overall validity
            display["metrics"].append(
                {
                    "label": "Problem Validity",
                    "correct": comparison["is_valid"],
                    "caption": f"Valid: {'Yes' if comparison['is_valid'] else 'No'}",
                    "show_score": False,
                }
            )

            # Individual evaluation scores (0/1 for each category)
            # Core criteria (always present)
            issue_breakdown = comparison["issue_breakdown"]
            core_issue_labels = {
                "ambiguous_error": "Ambiguous Error",
                "multiple_errors": "Multiple Errors",
                "numerical_error": "Numerical Error",
            }

            # Language-specific criteria (Japanese version)
            japanese_issue_labels = {
                "extra_elements": "Extra Elements",
                "synthesis_consistency_error": "Synthesis Consistency Error",
            }

            # Language-specific criteria (English version)
            english_issue_labels = {
                "unrealistic_scenario": "Unrealistic Scenario",
                "inconsistent_context": "Inconsistent Context",
            }

            # Combine labels based on what's present in the data
            issue_labels = core_issue_labels.copy()
            # Add Japanese-specific if present
            for key in japanese_issue_labels:
                if key in issue_breakdown:
                    issue_labels[key] = japanese_issue_labels[key]
            # Add English-specific if present
            for key in english_issue_labels:
                if key in issue_breakdown:
                    issue_labels[key] = english_issue_labels[key]

            for issue_key, issue_label in issue_labels.items():
                display["metrics"].append(
                    {
                        "label": issue_label,
                        "correct": issue_breakdown[issue_key]
                        == 0,  # 0 = no issue = good
                        "caption": f"Score: {issue_breakdown[issue_key]}",
                        "show_score": True,
                    }
                )

            # Total issues count
            display["metrics"].append(
                {
                    "label": "Total Issues",
                    "correct": None,  # No boolean judgment for issue count
                    "caption": f"Total: {comparison['total_issues']}",
                    "show_score": True,
                }
            )

            # Explanation as detail
            if comparison["explanation"]:
                display["details"].append(
                    {
                        "label": "Assessment Explanation",
                        "value": comparison["explanation"],
                    }
                )

            return display

        # For non-screening comparisons, show basic info
        if "sample_info" in comparison:
            display["details"].append(
                {
                    "label": "Has Error",
                    "value": "Yes" if comparison["sample_info"]["has_error"] else "No",
                }
            )
            if comparison["sample_info"]["error_type"] != "None":
                display["details"].append(
                    {
                        "label": "Error Type",
                        "value": comparison["sample_info"]["error_type"],
                    }
                )

        return display

    def format_screening_assessment_display(
        self, assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format screening assessment result for detailed display."""
        return {
            "validity": {
                "label": "Problem Validity",
                "value": "Valid" if assessment.get("is_valid", True) else "Invalid",
                "valid": assessment.get("is_valid", True),
            },
            "issues_summary": {
                "label": "Total Issues Found",
                "value": str(assessment.get("total_issues", 0)),
            },
            "issue_breakdown": [
                # Core criteria (always present)
                {
                    "label": "Ambiguous Error",
                    "count": assessment.get("ambiguous_error", 0),
                    "present": assessment.get("ambiguous_error", 0) > 0,
                },
                {
                    "label": "Multiple Errors",
                    "count": assessment.get("multiple_errors", 0),
                    "present": assessment.get("multiple_errors", 0) > 0,
                },
                {
                    "label": "Numerical Error",
                    "count": assessment.get("numerical_error", 0),
                    "present": assessment.get("numerical_error", 0) > 0,
                },
                # Language-specific criteria (Japanese)
                {
                    "label": "Extra Elements",
                    "count": assessment.get("extra_elements", 0),
                    "present": assessment.get("extra_elements", 0) > 0,
                    "language_specific": "japanese",
                },
                {
                    "label": "Synthesis Consistency Error",
                    "count": assessment.get("synthesis_consistency_error", 0),
                    "present": assessment.get("synthesis_consistency_error", 0) > 0,
                    "language_specific": "japanese",
                },
                # Language-specific criteria (English)
                {
                    "label": "Unrealistic Scenario",
                    "count": assessment.get("unrealistic_scenario", 0),
                    "present": assessment.get("unrealistic_scenario", 0) > 0,
                    "language_specific": "english",
                },
                {
                    "label": "Inconsistent Context",
                    "count": assessment.get("inconsistent_context", 0),
                    "present": assessment.get("inconsistent_context", 0) > 0,
                    "language_specific": "english",
                },
            ],
            "explanation": {
                "label": "Assessment Explanation",
                "text": assessment.get("explanation", ""),
            },
        }

"""MEDEC-specific response formatter."""

from typing import Dict, List, Any

from .base import BaseResponseFormatter


class MEDECResponseFormatter(BaseResponseFormatter):
    """MEDEC-specific response formatter."""

    def format_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Format MEDEC prediction for display."""
        pred_data = prediction["predictions"]

        return {
            "has_error": pred_data["errordetection"] == 1,
            "error_sentence_idx": pred_data["sentenceextraction"],
            "corrected_text": pred_data["errorcorrection"],
            "sample_id": prediction["sample_id"],
        }

    def format_ground_truth(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format MEDEC ground truth for display."""
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
        """Compare MEDEC prediction with ground truth."""
        pred = self.format_prediction(prediction)
        truth = self.format_ground_truth(ground_truth)

        # Extract sentences from the sentences field or question text for display
        if "sentences" in ground_truth:
            sentences = ground_truth["sentences"].split("\n")
        else:
            sentences = ground_truth["question"].split("\n")

        predicted_sentence = ""
        if pred["error_sentence_idx"] and 0 < pred["error_sentence_idx"] <= len(
            sentences
        ):
            predicted_sentence = sentences[pred["error_sentence_idx"] - 1]

        # Get actual error sentence index from ground truth
        actual_error_idx = (
            ground_truth["error_sentence_id"]
            if "error_sentence_id" in ground_truth
            else None
        )

        return {
            "error_detection_correct": pred["has_error"] == truth["has_error"],
            "predicted_has_error": pred["has_error"],
            "actual_has_error": truth["has_error"],
            "predicted_sentence_idx": pred["error_sentence_idx"],
            "actual_sentence_idx": actual_error_idx,
            "predicted_sentence": predicted_sentence,
            "actual_sentence": truth["error_sentence"],
            "predicted_correction": pred["corrected_text"],
            "actual_correction": truth["corrected_sentence"],
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

        # Error detection metric
        if "error_detection_correct" in comparison:
            display["metrics"].append(
                {
                    "label": "Error Detection",
                    "correct": comparison["error_detection_correct"],
                    "caption": f"Predicted: {'Error' if comparison['predicted_has_error'] else 'No Error'}",
                }
            )

        # Sentence extraction metric
        if "predicted_sentence_idx" in comparison:
            # Check if both have no error (idx=0 or None) or both point to same sentence index
            pred_idx = comparison["predicted_sentence_idx"]
            actual_idx = comparison["actual_sentence_idx"]
            actual_has_error = comparison["actual_has_error"]

            if not actual_has_error:
                # Ground truth has no error - prediction should be 0 or None
                is_correct = pred_idx == 0 or pred_idx is None
            else:
                # Ground truth has error - check if predicted index matches actual index
                is_correct = pred_idx == actual_idx

            display["metrics"].append(
                {
                    "label": "Sentence Extraction",
                    "correct": is_correct,
                    "caption": f"Sentence Index: {pred_idx if pred_idx else '0 (No error)'}",
                }
            )

        # Error correction metric - always show composite score (even if no correction was predicted)
        display["metrics"].append(
            {
                "label": "Error Correction",
                "correct": None,  # No boolean judgment for error correction
                "caption": f"Composite Score: {composite_score:.3f}"
                if composite_score is not None
                else "Composite Score: 1.000",
                "show_score": True,  # Flag to indicate this should be displayed differently
            }
        )

        # Detailed predictions
        if comparison.get("predicted_sentence"):
            display["details"].append(
                {
                    "label": "Predicted Error Sentence",
                    "value": comparison["predicted_sentence"],
                }
            )

        if comparison.get("predicted_correction"):
            display["details"].append(
                {
                    "label": "Predicted Correction",
                    "value": comparison["predicted_correction"],
                }
            )

        return display

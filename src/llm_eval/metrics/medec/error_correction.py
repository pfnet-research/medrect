"""Error correction metric for MEDEC evaluation using standard evaluation libraries."""

from typing import List, Dict, Any, Optional
import re
import numpy as np
from tqdm import tqdm
import gc
import os

from loguru import logger

from rouge_score import rouge_scorer
from bert_score import score as bertscore
from bleurt import score as bleurtscore

from .language_tokenizer import get_tokenizer_for_language
from .language_detector import detect_language_from_samples

from ..base import BaseMetric
from ...data.medec.samples import MEDECSample
from ...parsers.medec.error_correction import ErrorCorrectionParser


class ErrorCorrectionMetric(BaseMetric):
    """Metric for evaluating error correction performance using standard libraries."""

    def __init__(
        self,
        metric_name: str = "error_correction",
        parser: ErrorCorrectionParser = None,
        use_bertscore: bool = False,
        bertscore_model: str = "microsoft/deberta-xlarge-mnli",
        use_bleurt: bool = False,
        bleurt_checkpoint: str = "BLEURT-20",
        tokenization_method: str = "auto",
        device: str = "cuda",
        batch_size: int = 64,
    ):
        """Initialize error correction metric.

        Args:
            metric_name: Name of the metric
            parser: Parser for extracting corrected text
            use_bertscore: Whether to compute BERTScore (computationally expensive)
            bertscore_model: Model to use for BERTScore computation
            use_bleurt: Whether to compute BLEURT (computationally expensive)
            bleurt_checkpoint: BLEURT checkpoint to use
            tokenization_method: Tokenization method ("auto", "mecab", "char", "none")
            device: Device to use for GPU computations ("cuda", "cpu")
            batch_size: Batch size for BERTScore computation
        """
        if parser is None:
            parser = ErrorCorrectionParser()
        super().__init__(metric_name, parser)
        self.use_bertscore = use_bertscore
        self.bertscore_model = bertscore_model
        self.use_bleurt = use_bleurt
        self.bleurt_checkpoint = bleurt_checkpoint
        self.tokenization_method = tokenization_method
        self.device = device
        self.batch_size = batch_size

        # Tokenizer, ROUGE scorer, and BLEURT scorer will be initialized when needed
        self.tokenizer = None
        self.rouge_scorer = None
        self.bleurt_scorer = None
        self.detected_language = None  # Store detected language for reuse

    def _get_reference_value(self, sample: MEDECSample) -> str:
        """Get reference corrected sentence from MEDEC sample."""
        if not isinstance(sample, MEDECSample):
            raise ValueError(f"Expected MEDECSample, got {type(sample)}")

        if sample.corrected_sentence:
            return sample.corrected_sentence.strip()
        return ""

    def compute_metric(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, Any]:
        """Optimized compute metric that calculates aggregated and per-item scores in one pass."""
        if len(predictions) != len(references):
            raise ValueError(
                f"Predictions and references must have same length: {len(predictions)} vs {len(references)}"
            )

        if not predictions:
            return {"aggregated": self._get_empty_results(), "per_item": []}

        # Detect language and initialize components if not already done
        if self.detected_language is None:
            sample_references = references[: min(10, len(references))]
            self.detected_language = detect_language_from_samples(sample_references)
            logger.info(f"Detected language: {self.detected_language}")

        if self.tokenizer is None or self.rouge_scorer is None:
            self.tokenizer = get_tokenizer_for_language(
                self.detected_language, self.tokenization_method
            )
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"],
                use_stemmer=False,
                tokenizer=self.tokenizer,
            )

        # Prepare data following MEDEC evaluation protocol
        eval_data = self._prepare_evaluation_data(predictions, references)

        aggregated_results = {}

        # Calculate BLEURT score if enabled - get both aggregated and per-item
        bleurt_per_item = None
        if self.use_bleurt:
            bleurt_results, bleurt_per_item = self._compute_bleurt_standard(eval_data)
            aggregated_results.update(bleurt_results)

        # Calculate ROUGE scores - get both aggregated and per-item
        rouge_results, rouge_per_item = self._compute_rouge_standard(eval_data)
        aggregated_results.update(rouge_results)

        # Calculate BERTScore if enabled - get both aggregated and per-item
        bertscore_per_item = None
        if self.use_bertscore:
            bertscore_results, bertscore_per_item = self._compute_bertscore_standard(
                eval_data
            )
            aggregated_results.update(bertscore_results)

        # Calculate aggregate scores (following MEDIQA-CORR 2024 protocol)
        aggregated_results.update(
            self._compute_aggregate_scores(
                eval_data, rouge_per_item, bertscore_per_item, bleurt_per_item
            )
        )

        # Clear GPU cache after all heavy computations - critical for OOM prevention
        self._clear_gpu_cache()

        # Calculate composite scores (following MEDEC protocol)
        aggregated_results.update(
            self._compute_composite_scores(eval_data, aggregated_results)
        )

        # Add metadata and derived metrics
        total_samples = eval_data["counters"]["total_texts"]
        error_samples = len(eval_data["sentence_pairs"])  # samples that need correction
        no_error_samples = total_samples - error_samples  # NA cases
        correct_na_pairs = eval_data["counters"]["system_provided_correct_na"]
        incorrect_na_pairs = eval_data["counters"]["system_provided_incorrect_na"]
        total_na_cases = correct_na_pairs + incorrect_na_pairs

        aggregated_results.update(
            {
                f"{self.metric_name}_total_samples": total_samples,
                f"{self.metric_name}_error_samples": error_samples,
                f"{self.metric_name}_no_error_samples": no_error_samples,
                f"{self.metric_name}_no_error_accuracy": correct_na_pairs
                / total_na_cases
                if total_na_cases > 0
                else 1.0,
            }
        )

        # Build per-item scores from computed per-item results and handle NA cases
        per_item_scores = self._build_per_item_scores_from_computed(
            predictions,
            references,
            eval_data,
            rouge_per_item,
            bertscore_per_item,
            bleurt_per_item,
        )

        return {"aggregated": aggregated_results, "per_item": per_item_scores}

    def _build_per_item_scores_from_computed(
        self,
        predictions: List[str],
        references: List[str],
        eval_data: Dict[str, Any],
        rouge_per_item: List[Dict[str, float]],
        bertscore_per_item: Optional[List[Dict[str, float]]],
        bleurt_per_item: Optional[List[Dict[str, float]]],
    ) -> List[Dict[str, Any]]:
        """Build per-item scores from already computed metric results."""
        per_item_scores = []

        # Create mapping from sentence pairs to their computed scores
        sentence_pairs = eval_data["sentence_pairs"]
        pair_to_scores = {}

        for i, (pred, ref) in enumerate(sentence_pairs):
            pair_to_scores[(pred, ref)] = {
                "rouge": rouge_per_item[i] if rouge_per_item else {},
                "bertscore": bertscore_per_item[i] if bertscore_per_item else {},
                "bleurt": bleurt_per_item[i] if bleurt_per_item else {},
            }

        # Build per-item results for all samples
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            pred_clean = self._clean_text(pred)
            ref_clean = self._clean_text(ref)

            # Determine if this is a valid pair for NLG evaluation
            is_na_case, score_value = self._handle_na_case(pred_clean, ref_clean)

            item_score = {
                "sample_index": i,
                "prediction": pred_clean,
                "reference": ref_clean,
                "is_na_case": is_na_case,
            }

            if is_na_case:
                # Both are NA or one is NA - assign binary score
                item_score.update(
                    {
                        "bleurt_score": score_value if self.use_bleurt else 0.0,
                        "rouge_1_f": score_value,
                        "rouge_2_f": score_value,
                        "rouge_l_f": score_value,
                    }
                )
                if self.use_bertscore:
                    item_score["bertscore_f1"] = score_value
                item_score["composite_score"] = score_value
            else:
                # Use precomputed scores for this pair
                pair_key = (pred_clean, ref_clean)
                if pair_key in pair_to_scores:
                    scores = pair_to_scores[pair_key]

                    # ROUGE scores
                    rouge_scores = scores["rouge"]
                    item_score["rouge_1_f"] = rouge_scores.get("rouge_1_f", 0.0)
                    item_score["rouge_2_f"] = rouge_scores.get("rouge_2_f", 0.0)
                    item_score["rouge_l_f"] = rouge_scores.get("rouge_l_f", 0.0)

                    # BERTScore
                    if self.use_bertscore:
                        bertscore_scores = scores["bertscore"]
                        item_score["bertscore_f1"] = bertscore_scores.get(
                            "bertscore_f1", 0.0
                        )

                    # BLEURT score
                    if self.use_bleurt:
                        bleurt_scores = scores["bleurt"]
                        item_score["bleurt_score"] = bleurt_scores.get(
                            "bleurt_score", 0.0
                        )
                    else:
                        item_score["bleurt_score"] = 0.0

                    # Calculate composite score
                    composite_scores = [
                        item_score["rouge_1_f"],
                        item_score["rouge_2_f"],
                        item_score["rouge_l_f"],
                    ]

                    if self.use_bertscore:
                        composite_scores.append(item_score["bertscore_f1"])

                    if self.use_bleurt:
                        composite_scores.append(item_score["bleurt_score"])

                    item_score["composite_score"] = sum(composite_scores) / len(
                        composite_scores
                    )
                else:
                    # Fallback - should not happen in normal cases
                    item_score.update(
                        {
                            "rouge_1_f": 0.0,
                            "rouge_2_f": 0.0,
                            "rouge_l_f": 0.0,
                            "bleurt_score": 0.0 if self.use_bleurt else 0.0,
                            "composite_score": 0.0,
                        }
                    )
                    if self.use_bertscore:
                        item_score["bertscore_f1"] = 0.0

            per_item_scores.append(item_score)

        return per_item_scores

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove extra whitespace and punctuation
        text = re.sub(r"\s+", " ", text.strip())
        text = re.sub(r"[。、！？，．]", "", text)
        return text.lower()

    def _clean_text(self, text: Optional[str]) -> str:
        """Clean and normalize text input."""
        if not text:
            return ""
        text = text.strip()
        # Remove quotes if present (following MEDEC protocol)
        while text.startswith('"') and len(text) > 1:
            text = text[1:]
        while text.endswith('"') and len(text) > 1:
            text = text[:-1]
        return text

    def _is_na_text(self, text: str) -> bool:
        """Check if text represents 'no error' case."""
        text_upper = text.upper().strip()
        # PARSE_FAILED is not considered a NA text - it's a parsing failure
        if text_upper == "PARSE_FAILED":
            return False
        return text_upper in ["NA", "CORRECT", "_CORRECT_", ""] or re.search(
            r"\bCORRECT\b", text_upper
        )

    def _handle_na_case(self, pred: str, ref: str) -> tuple[bool, float]:
        """Handle NA cases and parsing failures following MEDEC protocol.

        Returns:
            (is_na_case, score): Boolean indicating if this is a NA case or parsing failure, and the score to assign
        """
        # Handle parsing failures first - always score 0
        if pred.upper().strip() == "PARSE_FAILED":
            return True, 0.0

        pred_is_na = self._is_na_text(pred)
        ref_is_na = self._is_na_text(ref)

        if pred_is_na and ref_is_na:
            # Both indicate no error - perfect match
            return True, 1.0
        elif pred_is_na or ref_is_na:
            # One indicates error, other doesn't - mismatch
            return True, 0.0
        else:
            # Both are actual corrections - evaluate with NLG metrics
            return False, 0.0

    def _get_empty_results(self) -> Dict[str, float]:
        """Return empty results structure."""
        results = {
            f"{self.metric_name}_exact_match": 0.0,
            f"{self.metric_name}_rouge_1_f": 0.0,
            f"{self.metric_name}_rouge_2_f": 0.0,
            f"{self.metric_name}_rouge_l_f": 0.0,
            f"{self.metric_name}_composite": 0.0,
            f"{self.metric_name}_total_pairs": 0,
            f"{self.metric_name}_valid_pairs": 0,
            f"{self.metric_name}_correct_na_pairs": 0,
        }
        if self.use_bertscore:
            results[f"{self.metric_name}_bertscore_f1"] = 0.0
        if self.use_bleurt:
            results[f"{self.metric_name}_bleurt"] = 0.0
        return results

    def _prepare_evaluation_data(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, Any]:
        """Prepare data for evaluation following MEDEC protocol."""
        counters = {
            "total_texts": len(predictions),
            "reference_na": 0,
            "total_system_texts": 0,
            "system_provided_na": 0,
            "system_provided_correct_na": 0,
            "system_provided_incorrect_na": 0,
        }

        sentence_pairs = []

        for pred, ref in zip(predictions, references):
            pred_clean = self._clean_text(pred)
            ref_clean = self._clean_text(ref)

            # Count reference NA cases
            if self._is_na_text(ref_clean):
                counters["reference_na"] += 1

            # Count system outputs
            if pred_clean:  # Non-empty prediction
                counters["total_system_texts"] += 1

                if self._is_na_text(pred_clean):
                    counters["system_provided_na"] += 1

                # Check for correct NA matching
                if self._is_na_text(ref_clean) and self._is_na_text(pred_clean):
                    counters["system_provided_correct_na"] += 1
                    continue  # Skip this pair for sentence-level evaluation

                # Check for incorrect NA prediction
                if self._is_na_text(pred_clean) and not self._is_na_text(ref_clean):
                    counters["system_provided_incorrect_na"] += 1
                    continue

                # Skip if reference is NA but prediction isn't (score = 0)
                if self._is_na_text(ref_clean):
                    continue

                # Both are actual sentences - add for NLG evaluation
                sentence_pairs.append((pred_clean, ref_clean))

        return {
            "sentence_pairs": sentence_pairs,
            "counters": counters,
            "all_pairs": list(
                zip(
                    [self._clean_text(p) for p in predictions],
                    [self._clean_text(r) for r in references],
                )
            ),
        }

    def _initialize_bleurt_if_needed(self) -> None:
        """Initialize BLEURT scorer if needed."""
        if not self.use_bleurt or self.bleurt_scorer is not None:
            return

        # Configure TensorFlow GPU memory growth to prevent excessive allocation
        try:
            import tensorflow as tf

            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("TensorFlow GPU memory growth enabled")
        except Exception as e:
            logger.warning(f"Failed to configure TensorFlow GPU memory: {e}")

        self.bleurt_scorer = bleurtscore.BleurtScorer(checkpoint=self.bleurt_checkpoint)
        logger.info(f"Initialized BLEURT with checkpoint: {self.bleurt_checkpoint}")

    def _compute_single_bleurt(self, prediction: str, reference: str) -> float:
        """Compute BLEURT score for a single prediction-reference pair."""
        self._initialize_bleurt_if_needed()
        if not self.use_bleurt or self.bleurt_scorer is None:
            return 0.0

        score = self.bleurt_scorer.score(
            references=[reference], candidates=[prediction], batch_size=1
        )[0]
        # Clip to [0,1] range like MEDIQA-CORR 2024
        return max(0.0, min(1.0, score))

    def _compute_bleurt_standard(
        self, eval_data: Dict[str, Any]
    ) -> tuple[Dict[str, float], Optional[List[Dict[str, float]]]]:
        """Compute BLEURT scores using BLEURT library.

        Returns:
            tuple: (aggregated_results, per_item_scores)
        """
        if not eval_data["sentence_pairs"] or not self.use_bleurt:
            return {
                f"{self.metric_name}_bleurt": 0.0,
                f"{self.metric_name}_bleurt_subset_check": 0.0,
            }, None

        self._initialize_bleurt_if_needed()
        if self.bleurt_scorer is None:
            return {
                f"{self.metric_name}_bleurt": 0.0,
                f"{self.metric_name}_bleurt_subset_check": 0.0,
            }, None

        predictions = [pair[0] for pair in eval_data["sentence_pairs"]]
        references = [pair[1] for pair in eval_data["sentence_pairs"]]

        logger.info(
            f"Computing BLEURT for {len(predictions)} pairs using {self.bleurt_checkpoint}..."
        )

        # Process in batches with progress bar for BLEURT
        batch_size = self.batch_size  # Use configured batch size
        bleurt_scores = []

        with tqdm(total=len(predictions), desc="BLEURT", unit="pairs") as pbar:
            for i in range(0, len(predictions), batch_size):
                batch_refs = references[i : i + batch_size]
                batch_preds = predictions[i : i + batch_size]

                batch_scores = self.bleurt_scorer.score(
                    references=batch_refs,
                    candidates=batch_preds,
                    batch_size=len(batch_refs),
                )
                bleurt_scores.extend(batch_scores)
                pbar.update(len(batch_refs))

        bleurt_scores = np.array(bleurt_scores)

        # Clip scores to [0,1] range like MEDIQA-CORR 2024
        bleurt_scores_clipped = np.array(
            [max(0.0, min(1.0, score)) for score in bleurt_scores]
        )
        avg_bleurt = np.mean(bleurt_scores_clipped)
        logger.info(f"BLEURT computation completed. Average score: {avg_bleurt:.4f}")

        # Per-item scores for aggregate computation
        per_item_scores = [{"bleurt": score} for score in bleurt_scores_clipped]

        results = {f"{self.metric_name}_bleurt": avg_bleurt}

        return results, per_item_scores

    def _compute_rouge_standard(
        self, eval_data: Dict[str, Any]
    ) -> tuple[Dict[str, float], List[Dict[str, float]]]:
        """Compute ROUGE scores using rouge-score library with custom tokenizer.

        Returns:
            tuple: (aggregated_results, per_item_scores)
        """
        if not eval_data["sentence_pairs"]:
            return {
                f"{self.metric_name}_rouge_1_f": 0.0,
                f"{self.metric_name}_rouge_2_f": 0.0,
                f"{self.metric_name}_rouge_l_f": 0.0,
            }, []

        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        per_item_scores = []

        for pred, ref in eval_data["sentence_pairs"]:
            # Use rouge-score library with custom tokenizer
            scores = self.rouge_scorer.score(ref, pred)  # (reference, prediction)
            r1f = scores["rouge1"].fmeasure
            r2f = scores["rouge2"].fmeasure
            rlf = scores["rougeL"].fmeasure

            rouge_1_scores.append(r1f)
            rouge_2_scores.append(r2f)
            rouge_l_scores.append(rlf)
            per_item_scores.append(
                {"rouge_1_f": r1f, "rouge_2_f": r2f, "rouge_l_f": rlf}
            )

        # Subset check scores (mean of sentence pairs only)
        r1f_subset_check = np.mean(rouge_1_scores) if rouge_1_scores else 0.0
        r2f_subset_check = np.mean(rouge_2_scores) if rouge_2_scores else 0.0
        rlf_subset_check = np.mean(rouge_l_scores) if rouge_l_scores else 0.0

        results = {
            f"{self.metric_name}_rouge_1_f": r1f_subset_check,
            f"{self.metric_name}_rouge_2_f": r2f_subset_check,
            f"{self.metric_name}_rouge_l_f": rlf_subset_check,
        }

        return results, per_item_scores

    def _compute_bertscore_standard(
        self, eval_data: Dict[str, Any]
    ) -> tuple[Dict[str, float], Optional[List[Dict[str, float]]]]:
        """Compute BERTScore using bert-score library.

        Returns:
            tuple: (aggregated_results, per_item_scores)
        """
        if not eval_data["sentence_pairs"] or not self.use_bertscore:
            return {f"{self.metric_name}_bertscore_f1": 0.0}, None

        predictions = [pair[0] for pair in eval_data["sentence_pairs"]]
        references = [pair[1] for pair in eval_data["sentence_pairs"]]

        logger.info(
            f"Computing BERTScore for {len(predictions)} pairs using {self.bertscore_model}..."
        )

        # Use already detected language for BERTScore
        if self.detected_language is None:
            # Fallback: detect language if not already done (shouldn't happen in normal flow)
            sample_references = references[: min(10, len(references))]
            self.detected_language = detect_language_from_samples(sample_references)
            logger.info(f"Language detection fallback: {self.detected_language}")

        bertscore_lang = "ja" if self.detected_language == "ja" else "en"
        logger.info(
            f"Using BERTScore lang: {bertscore_lang} (detected: {self.detected_language})"
        )

        # Check if offline mode is enabled via environment variables
        offline_mode = (
            os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            or os.environ.get("HF_DATASETS_OFFLINE") == "1"
        )
        if offline_mode:
            logger.info("Offline mode detected - using local cache for BERTScore")

        _, _, f1_scores = bertscore(
            predictions,
            references,
            model_type=self.bertscore_model,
            lang=bertscore_lang,
            device=self.device,  # Use configured device
            batch_size=self.batch_size,  # Use configured batch size
            verbose=True,  # Enable progress bar
            rescale_with_baseline=True,
        )

        # Clip scores to [0,1] range
        f1_scores_clipped = np.array(
            [max(0.0, min(1.0, score.item())) for score in f1_scores]
        )
        avg_bertscore = np.mean(f1_scores_clipped)
        logger.info(f"BERTScore computation completed. Average F1: {avg_bertscore:.4f}")

        # Per-item scores for aggregate computation
        per_item_scores = [{"bertscore_f1": score} for score in f1_scores_clipped]

        results = {f"{self.metric_name}_bertscore_f1": avg_bertscore}

        return results, per_item_scores

    def _compute_composite_scores(
        self, eval_data: Dict[str, Any], results: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute composite scores following MEDEC protocol.

        Composite scores account for both sentence correction and NA case detection:
        - error_correction_composite_avg: Overall performance (average of all metric composites)
        - error_correction_rouge_1_composite: ROUGE-1 based composite including NA cases
        - error_correction_bleurt_composite: BLEURT based composite including NA cases (if enabled)
        - error_correction_bertscore_composite: BERTScore based composite including NA cases (if enabled)
        """
        total_texts = eval_data["counters"]["total_texts"]
        correct_na_count = eval_data["counters"]["system_provided_correct_na"]
        sentence_count = len(eval_data["sentence_pairs"])

        if total_texts == 0:
            return {f"{self.metric_name}_composite_avg": 0.0}

        # Composite score = (sum of sentence-level scores + correct NA count) / total texts
        # This follows MEDEC's evaluation protocol where:
        # - Correct NA predictions count as 1.0 each
        # - Sentence corrections are scored by the respective metric (ROUGE-1, BLEURT, etc.)
        # - Total texts includes both sentence corrections and NA cases

        composite_scores = []

        # Use ROUGE-1 F-score as primary component (following MEDEC)
        rouge1_sum = results.get(f"{self.metric_name}_rouge_1_f", 0.0) * sentence_count
        rouge1_composite = (rouge1_sum + correct_na_count) / total_texts
        composite_scores.append(rouge1_composite)

        # Add BLEURT composite if available
        if self.use_bleurt:
            bleurt_sum = results.get(f"{self.metric_name}_bleurt", 0.0) * sentence_count
            bleurt_composite = (bleurt_sum + correct_na_count) / total_texts
            composite_scores.append(bleurt_composite)

        # Add BERTScore composite if available
        if self.use_bertscore:
            bertscore_sum = (
                results.get(f"{self.metric_name}_bertscore_f1", 0.0) * sentence_count
            )
            bertscore_composite = (bertscore_sum + correct_na_count) / total_texts
            composite_scores.append(bertscore_composite)

        # Overall composite is average of available composite scores
        overall_composite = np.mean(composite_scores) if composite_scores else 0.0

        composite_results = {
            f"{self.metric_name}_composite_avg": overall_composite,
            f"{self.metric_name}_rouge_1_composite": rouge1_composite,
        }

        if self.use_bleurt:
            composite_results[f"{self.metric_name}_bleurt_composite"] = bleurt_composite

        if self.use_bertscore:
            composite_results[f"{self.metric_name}_bertscore_composite"] = (
                bertscore_composite
            )

        return composite_results

    def _compute_aggregate_scores(
        self,
        eval_data: Dict[str, Any],
        rouge_per_item: Optional[List[Dict[str, float]]],
        bertscore_per_item: Optional[List[Dict[str, float]]],
        bleurt_per_item: Optional[List[Dict[str, float]]],
    ) -> Dict[str, float]:
        """Compute aggregate scores following MEDIQA-CORR 2024 protocol.

        This computes element-wise average of multiple metrics (ROUGE-1, BERTScore, BLEURT).
        Returns:
        - error_correction_average_score: Average of all available metrics for valid sentence pairs only
        """
        total_texts = eval_data["counters"]["total_texts"]
        correct_na_count = eval_data["counters"]["system_provided_correct_na"]
        sentence_count = len(eval_data["sentence_pairs"])

        if total_texts == 0 or sentence_count == 0:
            return {f"{self.metric_name}_average_score": 0.0}

        # Collect available metrics for aggregation
        available_metrics = []
        metric_names = []

        # Always include ROUGE-1 as primary component
        if rouge_per_item:
            available_metrics.append([item["rouge_1_f"] for item in rouge_per_item])
            metric_names.append("rouge_1_f")

        # Add BERTScore if available
        if bertscore_per_item and self.use_bertscore:
            available_metrics.append(
                [item["bertscore_f1"] for item in bertscore_per_item]
            )
            metric_names.append("bertscore_f1")

        # Add BLEURT if available
        if bleurt_per_item and self.use_bleurt:
            available_metrics.append([item["bleurt"] for item in bleurt_per_item])
            metric_names.append("bleurt")

        if not available_metrics:
            return {f"{self.metric_name}_average_score": 0.0}

        # Convert to numpy arrays for element-wise operations
        metric_arrays = [np.array(scores) for scores in available_metrics]
        aggregate_components = len(metric_arrays)

        # Element-wise sum of all metrics
        aggregate_subset_scores = np.zeros(sentence_count)
        for metric_array in metric_arrays:
            aggregate_subset_scores += metric_array

        # Element-wise average
        aggregate_subset_scores = aggregate_subset_scores / aggregate_components

        # Average score (mean of sentence pairs only)
        average_score = np.mean(aggregate_subset_scores)

        with tqdm(
            total=1, desc="Computing aggregate scores", unit="computation"
        ) as pbar:
            logger.info(
                f"Aggregate computation: {aggregate_components} metrics ({', '.join(metric_names)})"
            )
            logger.info(
                f"Sentence pairs: {sentence_count}, Total texts: {total_texts}, Correct NA: {correct_na_count}"
            )
            logger.info(f"Average score: {average_score:.4f}")
            pbar.update(1)

        return {f"{self.metric_name}_average_score": average_score}

    def _clear_gpu_cache(self) -> None:
        """Clear GPU memory cache while keeping models loaded for efficiency."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all GPU operations complete
                logger.debug("PyTorch GPU cache cleared and synchronized")
        except ImportError:
            logger.warning("PyTorch not available, skipping GPU cache cleanup")
        except Exception as e:
            logger.warning(f"PyTorch GPU cache cleanup failed: {e}")

        try:
            import tensorflow as tf

            # Light TensorFlow cleanup - avoid clearing models
            tf.keras.backend.clear_session()
            logger.debug("TensorFlow session cleared")
        except ImportError:
            logger.warning("TensorFlow not available, skipping session cleanup")
        except Exception as e:
            logger.warning(f"TensorFlow session cleanup failed: {e}")

        # Single round of garbage collection
        gc.collect()
        logger.debug("Python garbage collection completed")

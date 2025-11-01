#!/usr/bin/env python3
"""
Enhanced filtering script for JMEDEC dataset with error-type-specific thresholds:
1. Complete Accuracy: Exclude questions based on error-type-specific model count thresholds
2. Accuracy Difference: Exclude questions based on error-type-specific difference thresholds
3. Data Cleaning: Remove inconsistent data (error_flag=1 but error_sentence_id=None/nan)

This script applies different optimal thresholds for each error type instead of global thresholds,
based on the performance analysis showing significant difficulty differences across error types.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
import yaml

# Configuration loaded from YAML file
ERROR_TYPE_THRESHOLDS = None
ERROR_TYPE_MAPPING = None


def load_all_results(
    results_dir: Path, target_templates: List[str] = None
) -> pd.DataFrame:
    """Load all experiment results from the evaluation directory"""
    results = []

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        for template_dir in model_dir.iterdir():
            if not template_dir.is_dir():
                continue

            template_name = template_dir.name

            # Filter by template
            if target_templates and template_name not in target_templates:
                continue

            # Get latest run
            run_dirs = [d for d in template_dir.iterdir() if d.is_dir()]
            if not run_dirs:
                continue

            latest_run = max(run_dirs, key=lambda x: x.name)
            predictions_file = latest_run / "predictions.json"

            if predictions_file.exists():
                try:
                    with open(predictions_file, "r", encoding="utf-8") as f:
                        predictions_data = json.load(f)

                    for idx, sample in enumerate(predictions_data.get("samples", [])):
                        # Determine correct answers
                        error_flag = sample.get("error_flag", 0)
                        error_sentence_id = sample.get("error_sentence_id")

                        # Complete accuracy: exact match
                        if error_flag == 1 and error_sentence_id is not None:
                            correct_complete_answer = str(error_sentence_id)
                        else:
                            correct_complete_answer = "CORRECT"

                        # Error detection accuracy: just need to detect error or not
                        correct_detection_answer = "1" if error_flag == 1 else "0"

                        # Get predictions
                        predictions = sample.get("predictions", {})
                        error_detection_pred = predictions.get("error_detection")

                        # Complete prediction
                        if error_detection_pred == 0:
                            predicted_complete_answer = "CORRECT"
                        elif error_detection_pred == 1:
                            sentence_extraction_pred = predictions.get(
                                "sentence_extraction"
                            )
                            if (
                                sentence_extraction_pred is not None
                                and sentence_extraction_pred != 0
                            ):
                                predicted_complete_answer = str(
                                    sentence_extraction_pred
                                )
                            else:
                                predicted_complete_answer = "1"
                        else:
                            predicted_complete_answer = (
                                str(error_detection_pred)
                                if error_detection_pred is not None
                                else None
                            )

                        # Detection prediction
                        predicted_detection_answer = (
                            "1" if error_detection_pred == 1 else "0"
                        )

                        # Check correctness
                        is_complete_correct = False
                        is_detection_correct = False

                        if correct_complete_answer and predicted_complete_answer:
                            is_complete_correct = (
                                str(predicted_complete_answer).strip()
                                == str(correct_complete_answer).strip()
                            )

                        if correct_detection_answer and predicted_detection_answer:
                            is_detection_correct = (
                                str(predicted_detection_answer).strip()
                                == str(correct_detection_answer).strip()
                            )

                        results.append(
                            {
                                "model": model_name,
                                "template": template_name,
                                "sample_id": sample.get("sample_id", f"{idx:03d}"),
                                "is_complete_correct": is_complete_correct,
                                "is_detection_correct": is_detection_correct,
                            }
                        )

                except Exception as e:
                    print(f"Error loading {predictions_file}: {e}")

    return pd.DataFrame(results)


def calculate_unique_model_accuracies(
    df_results: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Calculate unique model accuracy for each question (both complete and detection)"""
    complete_accuracies = {}
    detection_accuracies = {}

    for sample_id in df_results["sample_id"].unique():
        question_data = df_results[df_results["sample_id"] == sample_id]

        # Get unique models that got this question correct (complete)
        complete_correct_models = question_data[question_data["is_complete_correct"]][
            "model"
        ].nunique()
        total_models = question_data["model"].nunique()

        # Get unique models that got this question correct (detection)
        detection_correct_models = question_data[question_data["is_detection_correct"]][
            "model"
        ].nunique()

        if total_models > 0:
            complete_accuracy = complete_correct_models / total_models
            detection_accuracy = detection_correct_models / total_models
            complete_accuracies[sample_id] = complete_accuracy
            detection_accuracies[sample_id] = detection_accuracy

    return complete_accuracies, detection_accuracies


def map_error_type_to_english(japanese_error_type: str) -> str:
    """Map Japanese error types to English equivalents"""
    if not japanese_error_type or japanese_error_type == "None":
        return "No Error"
    if not ERROR_TYPE_MAPPING:
        return japanese_error_type
    return ERROR_TYPE_MAPPING.get(japanese_error_type, japanese_error_type)


def get_thresholds_for_error_type(
    error_type: str,
    use_error_type_thresholds: bool = True,
    global_thresholds: Dict = None,
) -> Dict:
    """Get the appropriate thresholds for a given error type"""
    if not use_error_type_thresholds and global_thresholds:
        return global_thresholds

    # Map to English if needed
    english_error_type = map_error_type_to_english(error_type)

    # Get error-type-specific thresholds
    if ERROR_TYPE_THRESHOLDS and english_error_type in ERROR_TYPE_THRESHOLDS:
        return ERROR_TYPE_THRESHOLDS[english_error_type]
    elif ERROR_TYPE_THRESHOLDS and "default" in ERROR_TYPE_THRESHOLDS:
        print(
            f"Warning: No specific thresholds found for error type '{english_error_type}', using default"
        )
        return ERROR_TYPE_THRESHOLDS["default"]
    else:
        print(f"Error: No threshold configuration loaded for '{english_error_type}'")
        raise ValueError("Threshold configuration must be loaded from YAML file")


def filter_dataset_with_error_type_thresholds(
    input_file: Path,
    complete_accuracies: Dict[str, float],
    detection_accuracies: Dict[str, float],
    total_models: int = 11,
    use_error_type_thresholds: bool = True,
    global_thresholds: Optional[Dict] = None,
) -> Tuple[List[dict], List[dict], Dict[str, any]]:
    """Filter dataset with error-type-specific 3-stage filtering logic"""

    # Load original dataset
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    adopted_samples = []
    rejected_samples = []

    # Initialize filtering statistics with error type tracking
    stats = {
        "total_samples": len(data),
        "adopted_samples": 0,
        "rejected_samples": 0,
        "use_error_type_thresholds": use_error_type_thresholds,
        "error_type_stats": defaultdict(
            lambda: {
                "total": 0,
                "adopted": 0,
                "rejected": 0,
                "stage1_filtered": 0,
                "stage2_filtered": 0,
                "stage3_filtered": 0,
                "thresholds_used": None,
            }
        ),
        "filter_stages": {
            "stage1_complete_accuracy": {
                "filtered": 0,
                "reason": "Complete accuracy outside error-type-specific range",
            },
            "stage2_accuracy_difference": {
                "filtered": 0,
                "reason": "Accuracy difference exceeds error-type-specific threshold",
            },
            "stage3_data_cleaning": {
                "filtered": 0,
                "reason": "Data inconsistency (error_flag=1 but error_sentence_id=None/nan)",
            },
        },
        "complete_accuracy_distribution": defaultdict(int),
        "detection_accuracy_distribution": defaultdict(int),
        "accuracy_difference_distribution": defaultdict(int),
        "threshold_usage": defaultdict(int),
    }

    # Import cleaning functions
    # Import utility function (defined in step1_convert_synthesis.py)
    import sys
    import os

    sys.path.insert(0, os.path.dirname(__file__))

    # Process each sample
    for sample in data:
        sample_id = sample["sample_id"]
        complete_acc = complete_accuracies.get(sample_id, None)
        detection_acc = detection_accuracies.get(sample_id, None)

        if complete_acc is None or detection_acc is None:
            rejected_samples.append(sample)
            stats["rejected_samples"] += 1
            continue

        # Get error type and map to English
        error_type = sample.get("error_type", "None")
        error_flag = sample.get("error_flag", 0)
        if error_flag == 0:
            error_type = "No Error"

        english_error_type = map_error_type_to_english(error_type)

        # Get appropriate thresholds for this error type
        thresholds = get_thresholds_for_error_type(
            english_error_type, use_error_type_thresholds, global_thresholds
        )

        # Update error type statistics
        stats["error_type_stats"][english_error_type]["total"] += 1
        stats["error_type_stats"][english_error_type]["thresholds_used"] = thresholds
        stats["threshold_usage"][english_error_type] += 1

        # Add minimal accuracy metadata
        sample["metadata"] = sample.get("metadata", {})
        sample["metadata"]["complete_accuracy"] = complete_acc
        sample["metadata"]["detection_accuracy"] = detection_acc
        sample["metadata"]["accuracy_difference"] = detection_acc - complete_acc
        sample["metadata"]["error_type_english"] = english_error_type

        # Update distributions
        complete_bin = get_accuracy_bin(complete_acc)
        detection_bin = get_accuracy_bin(detection_acc)
        diff_bin = get_difference_bin(detection_acc - complete_acc)

        stats["complete_accuracy_distribution"][complete_bin] += 1
        stats["detection_accuracy_distribution"][detection_bin] += 1
        stats["accuracy_difference_distribution"][diff_bin] += 1

        # Stage 1: Complete Accuracy filter (error-type-specific)
        complete_correct_models = int(complete_acc * total_models)

        if (
            complete_correct_models < thresholds["stage1_min_models"]
            or complete_correct_models > thresholds["stage1_max_models"]
        ):
            sample["metadata"]["rejection_stage"] = 1
            sample["metadata"]["rejection_reason"] = (
                f"Complete accuracy: {complete_correct_models}/{total_models} models correct "
                f"(allowed range for {english_error_type}: {thresholds['stage1_min_models']}-{thresholds['stage1_max_models']})"
            )
            rejected_samples.append(sample)
            stats["rejected_samples"] += 1
            stats["filter_stages"]["stage1_complete_accuracy"]["filtered"] += 1
            stats["error_type_stats"][english_error_type]["rejected"] += 1
            stats["error_type_stats"][english_error_type]["stage1_filtered"] += 1
            continue

        # Stage 2: Accuracy Difference filter (error-type-specific)
        accuracy_diff = detection_acc - complete_acc
        if accuracy_diff >= thresholds["stage2_max_diff"]:
            sample["metadata"]["rejection_stage"] = 2
            sample["metadata"]["rejection_reason"] = (
                f"Accuracy difference: {accuracy_diff:.1%} "
                f"(>= {thresholds['stage2_max_diff']:.1%} threshold for {english_error_type})"
            )
            rejected_samples.append(sample)
            stats["rejected_samples"] += 1
            stats["filter_stages"]["stage2_accuracy_difference"]["filtered"] += 1
            stats["error_type_stats"][english_error_type]["rejected"] += 1
            stats["error_type_stats"][english_error_type]["stage2_filtered"] += 1
            continue

        # Stage 3: Data cleaning check
        should_reject_stage3 = False
        rejection_reason = None

        # Check for data inconsistency: error_flag=1 but error_sentence_id is None/nan
        error_sentence_id = sample.get("error_sentence_id")
        if error_flag == 1 and (
            error_sentence_id is None or str(error_sentence_id).lower() == "nan"
        ):
            should_reject_stage3 = True
            rejection_reason = "Data cleaning: inconsistent data (error_flag=1 but error_sentence_id=None/nan)"

        if should_reject_stage3:
            sample["metadata"]["rejection_stage"] = 3
            sample["metadata"]["rejection_reason"] = rejection_reason
            rejected_samples.append(sample)
            stats["rejected_samples"] += 1
            stats["filter_stages"]["stage3_data_cleaning"]["filtered"] += 1
            stats["error_type_stats"][english_error_type]["rejected"] += 1
            stats["error_type_stats"][english_error_type]["stage3_filtered"] += 1
            continue

        # If passed all filters, clean the bold markers and add to adopted
        cleaned_sample = clean_sample_data(sample)
        adopted_samples.append(cleaned_sample)
        stats["adopted_samples"] += 1
        stats["error_type_stats"][english_error_type]["adopted"] += 1

    return adopted_samples, rejected_samples, stats


def get_accuracy_bin(accuracy: float) -> str:
    """Categorize accuracy into bins"""
    if accuracy == 0.0:
        return "0.0 (no models correct)"
    elif accuracy < 0.25:
        return "0.0-0.25"
    elif accuracy < 0.5:
        return "0.25-0.5"
    elif accuracy < 0.75:
        return "0.5-0.75"
    elif accuracy < 1.0:
        return "0.75-1.0"
    else:
        return "1.0 (all models correct)"


def get_difference_bin(diff: float) -> str:
    """Categorize accuracy difference into bins"""
    if diff < 0:
        return "Negative (Complete > Detection)"
    elif diff < 0.1:
        return "0-10%"
    elif diff < 0.2:
        return "10-20%"
    elif diff < 0.3:
        return "20-30%"
    elif diff < 0.4:
        return "30-40%"
    else:
        return "40%+ (may be filtered based on error type)"


def clean_sample_data(sample: dict) -> dict:
    """Clean bold markers from sample data"""
    # Import utility function (defined in step1_convert_synthesis.py)
    import sys
    import os

    sys.path.insert(0, os.path.dirname(__file__))
    from step1_convert_synthesis import remove_bold_markers

    cleaned_sample = sample.copy()

    # Clean text fields
    text_fields = [
        "sentences",
        "corrected_text",
        "error_sentence",
        "corrected_sentence",
    ]
    for field in text_fields:
        if field in cleaned_sample:
            field_value = cleaned_sample[field]
            if isinstance(field_value, list):
                cleaned_sample[field] = [
                    remove_bold_markers(item) if isinstance(item, str) else item
                    for item in field_value
                ]
            elif isinstance(field_value, str):
                cleaned_sample[field] = remove_bold_markers(field_value)

    return cleaned_sample


def print_error_type_summary(stats: Dict) -> None:
    """Print detailed error type filtering summary"""
    print(f"\n{'=' * 80}")
    print("ERROR TYPE FILTERING SUMMARY")
    print(f"{'=' * 80}")
    print(f"Using error-type-specific thresholds: {stats['use_error_type_thresholds']}")

    # Sort error types by total count
    error_types = sorted(
        stats["error_type_stats"].items(), key=lambda x: x[1]["total"], reverse=True
    )

    for error_type, type_stats in error_types:
        adoption_rate = (
            (type_stats["adopted"] / type_stats["total"] * 100)
            if type_stats["total"] > 0
            else 0
        )
        thresholds = type_stats["thresholds_used"]

        print(f"\n【{error_type}】")
        print(f"  Total questions: {type_stats['total']}")
        print(f"  Adopted: {type_stats['adopted']} ({adoption_rate:.1f}%)")
        print(f"  Rejected: {type_stats['rejected']} ({100 - adoption_rate:.1f}%)")

        if thresholds:
            print(
                f"  Thresholds: {thresholds['stage1_min_models']}-{thresholds['stage1_max_models']} models, <{thresholds['stage2_max_diff']:.1%} diff"
            )
            if "rationale" in thresholds:
                print(f"  Rationale: {thresholds['rationale']}")

        if type_stats["rejected"] > 0:
            print("  Rejection breakdown:")
            print(f"    - Stage1 (Complete Acc): {type_stats['stage1_filtered']}")
            print(f"    - Stage2 (Acc Diff): {type_stats['stage2_filtered']}")
            print(f"    - Stage3 (Data Cleaning): {type_stats['stage3_filtered']}")


def print_error_no_error_breakdown(
    original_data: List[dict], adopted_samples: List[dict], rejected_samples: List[dict]
) -> None:
    """Print error/no-error breakdown for original, adopted, and rejected data"""
    print(f"\n{'=' * 80}")
    print("ERROR/NO-ERROR BREAKDOWN")
    print(f"{'=' * 80}")

    def count_error_breakdown(data_list: List[dict], data_name: str) -> None:
        if not data_list:
            print(f"\n【{data_name}】: No data")
            return

        error_count = sum(1 for sample in data_list if sample.get("error_flag", 0) == 1)
        no_error_count = len(data_list) - error_count
        total_count = len(data_list)

        error_pct = (error_count / total_count * 100) if total_count > 0 else 0
        no_error_pct = (no_error_count / total_count * 100) if total_count > 0 else 0

        print(f"\n【{data_name}】")
        print(f"  Total: {total_count}")
        print(f"  Error (error_flag=1): {error_count} ({error_pct:.1f}%)")
        print(f"  No Error (error_flag=0): {no_error_count} ({no_error_pct:.1f}%)")

    count_error_breakdown(original_data, "Original Data")
    count_error_breakdown(adopted_samples, "Adopted Data")
    count_error_breakdown(rejected_samples, "Rejected Data")


def load_threshold_config(config_path: Path = None) -> Tuple[Dict, Dict]:
    """Load threshold configuration and error type mapping from YAML file"""
    if config_path is None:
        config_path = (
            Path(__file__).parent.parent.parent
            / "configs"
            / "filtering"
            / "error_type_thresholds.yaml"
        )

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        thresholds = config.get("error_type_thresholds", {})
        error_mapping = config.get("error_type_mapping", {})
        print(f"[OK] Loaded thresholds from: {config_path}")
        return thresholds, error_mapping
    except FileNotFoundError:
        print(f"[ERROR] Error: Config file {config_path} not found!")
        raise
    except Exception as e:
        print(f"[ERROR] Error loading config file {config_path}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Filter JMEDEC dataset with error-type-specific thresholds"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(
            "data/outputs/jmle/case_119_v3_local_evaluation_3/medec/synthesized_jmle_case_119_v3_qwen3_235b_thinking"
        ),
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path(
            "data/inputs/medec/synthesized_jmle_case_119_v3_cleaned/qwen3-235b-a22b-thinking-2507_medec_data.json"
        ),
        help="Input dataset file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/inputs/medec/synthesized_jmle_case_119_v3_filtered_by_error_type"
        ),
        help="Output directory for filtered datasets",
    )
    parser.add_argument(
        "--templates",
        type=str,
        nargs="*",
        help="Specific templates to include (default: all templates)",
    )

    # Threshold configuration options
    parser.add_argument(
        "--global",
        action="store_true",
        help="Use global thresholds instead of error-type-specific ones",
    )
    parser.add_argument(
        "--config", type=Path, help="Path to YAML config file with custom thresholds"
    )

    # Global threshold fallbacks (used when --global is specified)
    parser.add_argument(
        "--stage1-min-models",
        type=int,
        default=2,
        help="Global: Minimum number of models that must be correct (default: 2)",
    )
    parser.add_argument(
        "--stage1-max-models",
        type=int,
        default=10,
        help="Global: Maximum number of models that can be correct (default: 10)",
    )
    parser.add_argument(
        "--stage2-max-diff",
        type=float,
        default=0.4,
        help="Global: Maximum accuracy difference to keep question (default: 0.4)",
    )
    parser.add_argument(
        "--total-models",
        type=int,
        default=11,
        help="Total number of models in evaluation (default: 11)",
    )

    args = parser.parse_args()

    # Determine whether to use error-type-specific thresholds
    use_error_type_thresholds = not getattr(args, "global", False)

    # Load threshold configuration from YAML file (only if using error-type-specific thresholds)
    global ERROR_TYPE_THRESHOLDS, ERROR_TYPE_MAPPING
    if use_error_type_thresholds:
        ERROR_TYPE_THRESHOLDS, ERROR_TYPE_MAPPING = load_threshold_config(args.config)
    else:
        # For global mode, use minimal mapping
        ERROR_TYPE_THRESHOLDS = None
        ERROR_TYPE_MAPPING = {}

    # Prepare global thresholds for fallback
    global_thresholds = (
        {
            "stage1_min_models": args.stage1_min_models,
            "stage1_max_models": args.stage1_max_models,
            "stage2_max_diff": args.stage2_max_diff,
        }
        if not use_error_type_thresholds
        else None
    )

    print(f"{'=' * 80}")
    print("JMEDEC FILTERING WITH ERROR-TYPE-SPECIFIC THRESHOLDS")
    print(f"{'=' * 80}")
    print(f"Input file: {args.input_file}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Use error-type-specific thresholds: {use_error_type_thresholds}")

    if args.templates:
        print(f"Target templates: {args.templates}")
    else:
        print("Using all available templates")

    # Load and process results
    print(f"\nLoading results from: {args.results_dir}")
    df_results = load_all_results(args.results_dir, args.templates)

    if len(df_results) == 0:
        print("[ERROR] No results found!")
        return

    print(f"[OK] Loaded {len(df_results)} experiment results")
    print(f"   Unique models: {df_results['model'].nunique()}")
    print(f"   Unique templates: {df_results['template'].nunique()}")
    print(f"   Unique questions: {df_results['sample_id'].nunique()}")

    # Calculate unique model accuracies
    print("\nCalculating question-wise unique model accuracies...")
    complete_accuracies, detection_accuracies = calculate_unique_model_accuracies(
        df_results
    )

    # Show accuracy distributions
    complete_acc_values = list(complete_accuracies.values())
    detection_acc_values = list(detection_accuracies.values())

    print("\n[Statistics] Complete Accuracy statistics:")
    print(f"   Mean: {np.mean(complete_acc_values):.3f}")
    print(f"   Std: {np.std(complete_acc_values):.3f}")
    print(f"   Min: {np.min(complete_acc_values):.3f}")
    print(f"   Max: {np.max(complete_acc_values):.3f}")

    print("\n[Statistics] Detection Accuracy statistics:")
    print(f"   Mean: {np.mean(detection_acc_values):.3f}")
    print(f"   Std: {np.std(detection_acc_values):.3f}")
    print(f"   Min: {np.min(detection_acc_values):.3f}")
    print(f"   Max: {np.max(detection_acc_values):.3f}")

    # Calculate accuracy differences
    accuracy_diffs = [
        detection_accuracies[sid] - complete_accuracies[sid]
        for sid in complete_accuracies.keys()
    ]
    print("\n[Statistics] Accuracy Difference (Detection - Complete) statistics:")
    print(f"   Mean: {np.mean(accuracy_diffs):.3f}")
    print(f"   Std: {np.std(accuracy_diffs):.3f}")
    print(f"   Min: {np.min(accuracy_diffs):.3f}")
    print(f"   Max: {np.max(accuracy_diffs):.3f}")

    # Apply filtering
    print(
        f"\nApplying {'error-type-specific' if use_error_type_thresholds else 'global'} filtering..."
    )
    if use_error_type_thresholds:
        print("   Error-type-specific thresholds will be applied based on:")
        print("   - Easy types: Stricter thresholds (higher accuracy required)")
        print("   - Hard types: Lenient thresholds (preserve more questions)")

    # Load original data for breakdown analysis
    with open(args.input_file, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    adopted_samples, rejected_samples, stats = (
        filter_dataset_with_error_type_thresholds(
            args.input_file,
            complete_accuracies,
            detection_accuracies,
            args.total_models,
            use_error_type_thresholds,
            global_thresholds,
        )
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save adopted dataset
    if args.templates:
        template_suffix = "_" + "_".join(args.templates)
        adopted_output = args.output_dir / f"adopted_questions{template_suffix}.json"
        rejected_output = args.output_dir / f"rejected_questions{template_suffix}.json"
    else:
        adopted_output = args.output_dir / "adopted_questions.json"
        rejected_output = args.output_dir / "rejected_questions.json"

    # Prepare minimal metadata
    filter_metadata = {
        "filter_criteria": "Error-type-specific 3-stage filtering",
        "use_error_type_thresholds": use_error_type_thresholds,
        "source_file": str(args.input_file),
        "filtering_stats": stats,
    }

    adopted_data = {
        "samples": adopted_samples,
        "metadata": {**filter_metadata, "dataset_type": "adopted"},
    }

    rejected_data = {
        "samples": rejected_samples,
        "metadata": {**filter_metadata, "dataset_type": "rejected"},
    }

    # Save files
    with open(adopted_output, "w", encoding="utf-8") as f:
        json.dump(adopted_data, f, ensure_ascii=False, indent=2)

    with open(rejected_output, "w", encoding="utf-8") as f:
        json.dump(rejected_data, f, ensure_ascii=False, indent=2)

    # Print results
    print(f"\n{'=' * 80}")
    print("FILTERING RESULTS")
    print(f"{'=' * 80}")
    print(f"Total samples: {stats['total_samples']}")
    print(
        f"Adopted samples: {stats['adopted_samples']} ({stats['adopted_samples'] / stats['total_samples'] * 100:.1f}%)"
    )
    print(
        f"Rejected samples: {stats['rejected_samples']} ({stats['rejected_samples'] / stats['total_samples'] * 100:.1f}%)"
    )

    print(f"\n{'=' * 80}")
    print("FILTERING STAGE BREAKDOWN")
    print(f"{'=' * 80}")
    for stage, info in stats["filter_stages"].items():
        stage_num = stage.split("_")[0][-1]
        filtered_count = info["filtered"]
        print(
            f"Stage {stage_num}: {filtered_count} samples filtered ({filtered_count / stats['total_samples'] * 100:.1f}%)"
        )
        print(f"  Reason: {info['reason']}")

    # Print error type summary
    print_error_type_summary(stats)

    # Print error/no-error breakdown
    print_error_no_error_breakdown(original_data, adopted_samples, rejected_samples)

    print(f"\n{'=' * 80}")
    print("OUTPUT FILES")
    print(f"{'=' * 80}")
    print(f"Adopted: {adopted_output}")
    print(f"Rejected: {rejected_output}")
    print(
        "\n[OK] Filtering complete! Error-type-specific thresholds have been applied successfully."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Batch extract correct samples from JMEDEC evaluation results.

This script processes batch evaluation results and extracts samples where predictions
are correct across all experiments defined in the config YAML.

Usage:
    # Extract correct samples from Japanese training data
    uv run python scripts/training_data/extract_correct_samples.py \
        --config configs/batch/synthesize_reasoning_for_medrect-train/medrect_ja_train.yaml \
        --output-path data/medrect/medrect-ja-train.json

    # Extract correct samples from English training data
    uv run python scripts/training_data/extract_correct_samples.py \
        --config configs/batch/synthesize_reasoning_for_medrect-train/medrect_en_train.yaml \
        --output-path data/medrect/medrect-en-train.json
"""

import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from loguru import logger


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, file_path: Path) -> None:
    """Save data to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_latest_timestamp_dir(base_path: Path) -> Optional[Path]:
    """Find the latest timestamp directory in the given path."""
    if not base_path.exists():
        return None

    timestamp_dirs = [
        d for d in base_path.iterdir() if d.is_dir() and d.name.count("_") == 1
    ]
    if not timestamp_dirs:
        return None

    # Sort by timestamp string (YYYYMMDD_HHMMSS format)
    latest_dir = max(timestamp_dirs, key=lambda x: x.name)
    return latest_dir


def extract_samples_from_predictions(
    predictions_data: Dict[str, Any],
) -> Tuple[List[int], List[int], List[int], Dict[str, Any]]:
    """
    Extract sample indices categorized by correctness and error_flag.

    Args:
        predictions_data: Data from predictions.json

    Returns:
        Tuple of (correct_indices, incorrect_error_flag_0_indices, incorrect_error_flag_1_indices, sample_statistics)
    """
    correct_indices = []
    incorrect_error_flag_0_indices = []  # Incorrect samples with error_flag=0
    incorrect_error_flag_1_indices = []  # Incorrect samples with error_flag=1

    # Get samples and detailed metrics
    samples = predictions_data.get("samples", [])
    detailed_metrics = predictions_data.get("detailed_metrics", {})

    error_detection_metrics = detailed_metrics.get("error_detection", [])
    sentence_extraction_metrics = detailed_metrics.get("sentence_extraction", [])

    # Create lookup dictionaries for faster access
    error_detection_lookup = {
        metric["sample_index"]: metric for metric in error_detection_metrics
    }
    sentence_extraction_lookup = {
        metric["sample_index"]: metric for metric in sentence_extraction_metrics
    }

    # Statistics tracking
    error_samples = 0
    no_error_samples = 0
    correct_error_samples = 0
    correct_no_error_samples = 0
    incorrect_error_samples = 0
    incorrect_no_error_samples = 0

    for i, sample in enumerate(samples):
        sample_index = i
        error_flag = sample.get("error_flag", 0)

        # Count error flag distribution
        if error_flag == 1:
            error_samples += 1
        else:
            no_error_samples += 1

        # Check error detection correctness
        error_detection_metric = error_detection_lookup.get(sample_index)
        is_error_detection_correct = (
            error_detection_metric and error_detection_metric.get("is_correct", False)
        )

        # Check if sample is completely correct
        is_completely_correct = False
        if is_error_detection_correct:
            if error_flag == 1:
                # For error samples, also check sentence extraction
                sentence_extraction_metric = sentence_extraction_lookup.get(
                    sample_index
                )
                if sentence_extraction_metric and sentence_extraction_metric.get(
                    "is_exact_match", False
                ):
                    is_completely_correct = True
            else:
                # For no-error samples, error detection correctness is sufficient
                is_completely_correct = True

        if is_completely_correct:
            correct_indices.append(sample_index)
            if error_flag == 1:
                correct_error_samples += 1
            else:
                correct_no_error_samples += 1
        else:
            # Sample is incorrect
            if error_flag == 1:
                incorrect_error_flag_1_indices.append(sample_index)
                incorrect_error_samples += 1
            else:
                incorrect_error_flag_0_indices.append(sample_index)
                incorrect_no_error_samples += 1

    # Calculate filtered (correct samples only) statistics
    filtered_error_ratio = (
        correct_error_samples / len(correct_indices) if correct_indices else 0.0
    )
    filtered_no_error_ratio = (
        correct_no_error_samples / len(correct_indices) if correct_indices else 0.0
    )

    sample_statistics = {
        "total_samples": len(samples),
        "error_samples": error_samples,
        "no_error_samples": no_error_samples,
        "correct_error_samples": correct_error_samples,
        "correct_no_error_samples": correct_no_error_samples,
        "incorrect_error_samples": incorrect_error_samples,
        "incorrect_no_error_samples": incorrect_no_error_samples,
        "error_ratio": error_samples / len(samples) if samples else 0.0,
        "no_error_ratio": no_error_samples / len(samples) if samples else 0.0,
        "error_accuracy": correct_error_samples / error_samples
        if error_samples > 0
        else 0.0,
        "no_error_accuracy": correct_no_error_samples / no_error_samples
        if no_error_samples > 0
        else 0.0,
        "filtered_error_ratio": filtered_error_ratio,
        "filtered_no_error_ratio": filtered_no_error_ratio,
    }

    return (
        correct_indices,
        incorrect_error_flag_0_indices,
        incorrect_error_flag_1_indices,
        sample_statistics,
    )


def process_experiment(
    results_base: Path, task: str, dataset: str, model: str, template: str
) -> Optional[Dict[str, Any]]:
    """
    Process a single experiment and extract correct samples.

    Args:
        results_base: Base results directory
        task: Task name
        dataset: Dataset name
        model: Model name
        template: Template name

    Returns:
        Dict containing extracted data or None if files not found
    """
    # Find latest timestamp directory
    experiment_path = results_base / task / dataset / model / template
    latest_dir = find_latest_timestamp_dir(experiment_path)

    if not latest_dir:
        logger.warning(
            f"No timestamp directory found for {task}/{dataset}/{model}/{template}"
        )
        return None

    # Check for required files
    predictions_path = latest_dir / "predictions.json"
    raw_responses_path = latest_dir / "raw_responses.json"

    if not predictions_path.exists():
        logger.warning(f"predictions.json not found in {latest_dir}")
        return None

    if not raw_responses_path.exists():
        logger.warning(f"raw_responses.json not found in {latest_dir}")
        return None

    logger.info(
        f"Processing {task}/{dataset}/{model}/{template} from {latest_dir.name}"
    )

    try:
        # Load data
        predictions_data = load_json(predictions_path)
        raw_responses_data = load_json(raw_responses_path)

        # Extract sample indices with statistics
        (
            correct_indices,
            incorrect_error_flag_0_indices,
            incorrect_error_flag_1_indices,
            sample_stats,
        ) = extract_samples_from_predictions(predictions_data)

        if not correct_indices:
            logger.warning(
                f"No correct samples found for {task}/{dataset}/{model}/{template}"
            )
            return None

        # Extract samples and raw responses
        all_samples = predictions_data["samples"]
        raw_interactions = raw_responses_data.get("interactions", [])

        # Correct samples
        correct_samples = [all_samples[i] for i in correct_indices]
        correct_raw_responses = [
            raw_interactions[i] for i in correct_indices if i < len(raw_interactions)
        ]

        # Incorrect samples with error_flag=0
        incorrect_error_flag_0_samples = [
            all_samples[i] for i in incorrect_error_flag_0_indices
        ]
        incorrect_error_flag_0_raw_responses = [
            raw_interactions[i]
            for i in incorrect_error_flag_0_indices
            if i < len(raw_interactions)
        ]

        # Incorrect samples with error_flag=1
        incorrect_error_flag_1_samples = [
            all_samples[i] for i in incorrect_error_flag_1_indices
        ]
        incorrect_error_flag_1_raw_responses = [
            raw_interactions[i]
            for i in incorrect_error_flag_1_indices
            if i < len(raw_interactions)
        ]

        # Log concise experiment statistics
        logger.info(
            f"[Statistics] {dataset}/{model}/{template}: "
            f"{len(correct_indices)}/{sample_stats['total_samples']} "
            f"({len(correct_indices) / sample_stats['total_samples']:.1%})"
        )
        logger.info(
            f"   E/N: {sample_stats['error_samples']}/{sample_stats['no_error_samples']} → "
            f"{sample_stats['correct_error_samples']}/{sample_stats['correct_no_error_samples']} "
            f"({sample_stats['filtered_error_ratio']:.1%}/{sample_stats['filtered_no_error_ratio']:.1%})"
        )

        return {
            "experiment_info": {
                "task": task,
                "dataset": dataset,
                "model": model,
                "template": template,
                "timestamp": latest_dir.name,
                "source_predictions": str(predictions_path),
                "source_raw_responses": str(raw_responses_path),
            },
            "statistics": {
                "total_samples": sample_stats["total_samples"],
                "correct_samples_count": len(correct_indices),
                "accuracy": len(correct_indices) / sample_stats["total_samples"]
                if sample_stats["total_samples"] > 0
                else 0.0,
                "error_samples": sample_stats["error_samples"],
                "no_error_samples": sample_stats["no_error_samples"],
                "correct_error_samples": sample_stats["correct_error_samples"],
                "correct_no_error_samples": sample_stats["correct_no_error_samples"],
                "error_ratio": sample_stats["error_ratio"],
                "no_error_ratio": sample_stats["no_error_ratio"],
                "error_accuracy": sample_stats["error_accuracy"],
                "no_error_accuracy": sample_stats["no_error_accuracy"],
                "filtered_error_ratio": sample_stats["filtered_error_ratio"],
                "filtered_no_error_ratio": sample_stats["filtered_no_error_ratio"],
            },
            "correct_sample_indices": correct_indices,
            "correct_samples": correct_samples,
            "correct_raw_responses": correct_raw_responses,
            "incorrect_error_flag_0_indices": incorrect_error_flag_0_indices,
            "incorrect_error_flag_0_samples": incorrect_error_flag_0_samples,
            "incorrect_error_flag_0_raw_responses": incorrect_error_flag_0_raw_responses,
            "incorrect_error_flag_1_indices": incorrect_error_flag_1_indices,
            "incorrect_error_flag_1_samples": incorrect_error_flag_1_samples,
            "incorrect_error_flag_1_raw_responses": incorrect_error_flag_1_raw_responses,
        }

    except Exception as e:
        logger.error(f"Error processing {task}/{dataset}/{model}/{template}: {e}")
        return None


def create_jmedec_dataset_format(
    all_results: List[Dict[str, Any]], sample_type: str = "correct"
) -> List[Dict[str, Any]]:
    """
    Create jmedec benchmark dataset format from extracted results.

    Args:
        all_results: List of results from all experiments
        sample_type: Type of samples to include ('correct', 'incorrect_error_flag_0', 'incorrect_error_flag_1')

    Returns:
        List of samples in jmedec dataset format
    """
    jmedec_samples = []

    for result in all_results:
        exp_info = result["experiment_info"]

        # Select samples based on sample_type
        if sample_type == "correct":
            samples = result["correct_samples"]
            raw_responses = result["correct_raw_responses"]
        elif sample_type == "incorrect_error_flag_0":
            samples = result["incorrect_error_flag_0_samples"]
            raw_responses = result["incorrect_error_flag_0_raw_responses"]
        elif sample_type == "incorrect_error_flag_1":
            samples = result["incorrect_error_flag_1_samples"]
            raw_responses = result["incorrect_error_flag_1_raw_responses"]
        else:
            raise ValueError(f"Unknown sample_type: {sample_type}")

        for i, sample in enumerate(samples):
            # Get corresponding raw response
            raw_response_data = raw_responses[i] if i < len(raw_responses) else {}
            raw_response = raw_response_data.get("raw_response", {})
            response_metadata = raw_response_data.get("response_metadata", {})

            # Create jmedec format sample
            jmedec_sample = {
                "sample_id": sample["sample_id"],
                "sentences": sample["sentences"],
                "error_flag": sample["error_flag"],
                "error_type": sample.get("error_type"),
                "error_sentence_id": sample.get("error_sentence_id"),
                "error_sentence": sample.get("error_sentence"),
                "corrected_sentence": sample.get("corrected_sentence"),
                "corrected_text": sample.get("corrected_text"),
                "metadata": sample.get("metadata", {}),
                "raw_response": {
                    "content": raw_response.get("content", ""),
                    "reasoning": raw_response.get("reasoning", ""),
                    "source_model": response_metadata.get("model", exp_info["model"]),
                },
            }

            # Clean up None values
            jmedec_sample = {k: v for k, v in jmedec_sample.items() if v is not None}

            jmedec_samples.append(jmedec_sample)

    return jmedec_samples


def main():
    parser = argparse.ArgumentParser(
        description="Batch extract correct samples from JMEDEC evaluation results"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to batch config YAML file (e.g., configs/batch/jmedec_112_117_api.yaml)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="correct_samples_batch_output.json",
        help="Output path for extracted correct samples",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["jmedec_dataset", "combined", "separate"],
        default="jmedec_dataset",
        help='Output format: "jmedec_dataset" for benchmark dataset format (default), "combined" for single file, "separate" for per-experiment files',
    )
    parser.add_argument(
        "--include-detailed-output",
        action="store_true",
        help="Also output detailed format with full raw responses and metadata",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = load_yaml(config_path)

    task = config["task"]
    models = config["models"]
    datasets = config["datasets"]
    templates = config["templates"]
    output_dir = config.get("output_dir", "results")

    results_base = Path(output_dir)

    logger.info(
        f"Processing {len(models)} models × {len(datasets)} datasets × {len(templates)} templates = {len(models) * len(datasets) * len(templates)} experiments"
    )

    # Process all experiments
    all_results = []
    successful_experiments = 0
    total_experiments = len(models) * len(datasets) * len(templates)

    # Cumulative statistics tracking
    cumulative_total_samples = 0
    cumulative_correct_samples = 0
    cumulative_error_samples = 0
    cumulative_no_error_samples = 0
    cumulative_correct_error_samples = 0
    cumulative_correct_no_error_samples = 0

    logger.info(f"🚀 Starting batch processing: {total_experiments} experiments")
    logger.info("=" * 60)

    for model in models:
        for dataset in datasets:
            for template in templates:
                result = process_experiment(
                    results_base, task, dataset, model, template
                )
                if result:
                    all_results.append(result)
                    successful_experiments += 1

                    # Update cumulative statistics
                    stats = result["statistics"]
                    cumulative_total_samples += stats["total_samples"]
                    cumulative_correct_samples += stats["correct_samples_count"]
                    cumulative_error_samples += stats["error_samples"]
                    cumulative_no_error_samples += stats["no_error_samples"]
                    cumulative_correct_error_samples += stats["correct_error_samples"]
                    cumulative_correct_no_error_samples += stats[
                        "correct_no_error_samples"
                    ]

    if not all_results:
        logger.error("No successful experiments found")
        return 1

    logger.info("=" * 60)
    final_accuracy = (
        cumulative_correct_samples / cumulative_total_samples
        if cumulative_total_samples > 0
        else 0.0
    )
    final_filtered_error_ratio = (
        cumulative_correct_error_samples / cumulative_correct_samples
        if cumulative_correct_samples > 0
        else 0.0
    )
    final_filtered_no_error_ratio = (
        cumulative_correct_no_error_samples / cumulative_correct_samples
        if cumulative_correct_samples > 0
        else 0.0
    )

    logger.info(
        f"[Target] FINAL BATCH: {cumulative_correct_samples}/{cumulative_total_samples} ({final_accuracy:.1%})"
    )
    logger.info(
        f"   E/N: {cumulative_error_samples}/{cumulative_no_error_samples} → "
        f"{cumulative_correct_error_samples}/{cumulative_correct_no_error_samples} "
        f"({final_filtered_error_ratio:.1%}/{final_filtered_no_error_ratio:.1%})"
    )
    logger.info("=" * 60)

    # Prepare output based on format
    if args.output_format == "jmedec_dataset":
        # Create jmedec benchmark dataset format for correct samples (backward compatibility)
        jmedec_samples_correct = create_jmedec_dataset_format(
            all_results, sample_type="correct"
        )

        # Create separate formats for incorrect samples by error_flag
        jmedec_samples_incorrect_error_0 = create_jmedec_dataset_format(
            all_results, sample_type="incorrect_error_flag_0"
        )
        jmedec_samples_incorrect_error_1 = create_jmedec_dataset_format(
            all_results, sample_type="incorrect_error_flag_1"
        )

        # Prepare dataset metadata
        dataset_metadata = {
            "extraction_info": {
                "extraction_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "config_file": str(config_path),
                "task": task,
                "total_experiments": total_experiments,
                "successful_experiments": successful_experiments,
            },
            "source_experiments": [
                {
                    "dataset": r["experiment_info"]["dataset"],
                    "model": r["experiment_info"]["model"],
                    "template": r["experiment_info"]["template"],
                    "timestamp": r["experiment_info"]["timestamp"],
                    "correct_samples": r["statistics"]["correct_samples_count"],
                    "total_samples": r["statistics"]["total_samples"],
                    "incorrect_error_flag_0_samples": len(
                        r["incorrect_error_flag_0_samples"]
                    ),
                    "incorrect_error_flag_1_samples": len(
                        r["incorrect_error_flag_1_samples"]
                    ),
                }
                for r in all_results
            ],
        }

        # Determine output directory based on config
        # Use first dataset name for directory structure
        primary_dataset = datasets[0] if datasets else "default"
        output_base_path = Path("data/inputs") / task / primary_dataset

        # Save correct samples
        correct_output_path = output_base_path / "correct_samples.json"
        save_json(jmedec_samples_correct, correct_output_path)

        # Save incorrect samples with error_flag=0
        incorrect_error_0_output_path = (
            output_base_path / "incorrect_error_flag_0_samples.json"
        )
        save_json(jmedec_samples_incorrect_error_0, incorrect_error_0_output_path)

        # Save incorrect samples with error_flag=1
        incorrect_error_1_output_path = (
            output_base_path / "incorrect_error_flag_1_samples.json"
        )
        save_json(jmedec_samples_incorrect_error_1, incorrect_error_1_output_path)

        # Save metadata separately
        metadata_output_path = output_base_path / "samples_metadata.json"
        save_json(dataset_metadata, metadata_output_path)

        logger.info(f"[OK] Correct samples saved to {correct_output_path}")
        logger.info(f"[Statistics] Total correct samples: {len(jmedec_samples_correct)}")
        logger.info(
            f"[OK] Incorrect error_flag=0 samples saved to {incorrect_error_0_output_path}"
        )
        logger.info(
            f"[Statistics] Total incorrect error_flag=0 samples: {len(jmedec_samples_incorrect_error_0)}"
        )
        logger.info(
            f"[OK] Incorrect error_flag=1 samples saved to {incorrect_error_1_output_path}"
        )
        logger.info(
            f"[Statistics] Total incorrect error_flag=1 samples: {len(jmedec_samples_incorrect_error_1)}"
        )
        logger.info(f"📋 Metadata saved to {metadata_output_path}")

        # Verify that all samples are accounted for
        total_extracted = (
            len(jmedec_samples_correct)
            + len(jmedec_samples_incorrect_error_0)
            + len(jmedec_samples_incorrect_error_1)
        )
        logger.info(f"[Debug] Total samples extracted: {total_extracted}")
        logger.info(f"   Correct: {len(jmedec_samples_correct)}")
        logger.info(
            f"   Incorrect (error_flag=0): {len(jmedec_samples_incorrect_error_0)}"
        )
        logger.info(
            f"   Incorrect (error_flag=1): {len(jmedec_samples_incorrect_error_1)}"
        )

        # Optionally also output detailed format
        if args.include_detailed_output:
            detailed_output_path = Path(args.output_path)
            all_raw_responses = []
            experiment_metadata = []

            for result in all_results:
                exp_info = result["experiment_info"]
                for i, raw_response in enumerate(result["correct_raw_responses"]):
                    enhanced_response = raw_response.copy()
                    enhanced_response["_experiment_metadata"] = {
                        "task": exp_info["task"],
                        "dataset": exp_info["dataset"],
                        "model": exp_info["model"],
                        "template": exp_info["template"],
                        "timestamp": exp_info["timestamp"],
                        "original_sample_index": result["correct_sample_indices"][i],
                    }
                    all_raw_responses.append(enhanced_response)

                experiment_metadata.append(
                    {
                        "experiment": f"{exp_info['dataset']}/{exp_info['model']}/{exp_info['template']}",
                        "timestamp": exp_info["timestamp"],
                        "correct_samples": result["statistics"][
                            "correct_samples_count"
                        ],
                        "total_samples": result["statistics"]["total_samples"],
                        "accuracy": result["statistics"]["accuracy"],
                    }
                )

            total_samples_all = sum(
                r["statistics"]["total_samples"] for r in all_results
            )
            total_correct_all = sum(
                r["statistics"]["correct_samples_count"] for r in all_results
            )

            detailed_output_data = {
                "metadata": {
                    "extraction_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "config_file": str(config_path),
                    "task": task,
                    "total_experiments": total_experiments,
                    "successful_experiments": successful_experiments,
                },
                "aggregate_statistics": {
                    "total_samples_across_all_experiments": total_samples_all,
                    "total_correct_samples": total_correct_all,
                    "overall_accuracy": total_correct_all / total_samples_all
                    if total_samples_all > 0
                    else 0.0,
                },
                "experiment_summary": experiment_metadata,
                "raw_responses_only_correct": all_raw_responses,
            }

            save_json(detailed_output_data, detailed_output_path)
            logger.info(f"📄 Detailed output also saved to {detailed_output_path}")

    elif args.output_format == "combined":
        # Combine all results into a single file

        # Aggregate statistics
        total_samples_all = sum(r["statistics"]["total_samples"] for r in all_results)
        total_correct_all = sum(
            r["statistics"]["correct_samples_count"] for r in all_results
        )

        # Combine all raw responses
        all_raw_responses = []
        experiment_metadata = []

        for result in all_results:
            exp_info = result["experiment_info"]
            for i, raw_response in enumerate(result["correct_raw_responses"]):
                # Add experiment metadata to each response
                enhanced_response = raw_response.copy()
                enhanced_response["_experiment_metadata"] = {
                    "task": exp_info["task"],
                    "dataset": exp_info["dataset"],
                    "model": exp_info["model"],
                    "template": exp_info["template"],
                    "timestamp": exp_info["timestamp"],
                    "original_sample_index": result["correct_sample_indices"][i],
                }
                all_raw_responses.append(enhanced_response)

            experiment_metadata.append(
                {
                    "experiment": f"{exp_info['dataset']}/{exp_info['model']}/{exp_info['template']}",
                    "timestamp": exp_info["timestamp"],
                    "correct_samples": result["statistics"]["correct_samples_count"],
                    "total_samples": result["statistics"]["total_samples"],
                    "accuracy": result["statistics"]["accuracy"],
                }
            )

        output_data = {
            "metadata": {
                "extraction_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "config_file": str(config_path),
                "task": task,
                "total_experiments": total_experiments,
                "successful_experiments": successful_experiments,
            },
            "aggregate_statistics": {
                "total_samples_across_all_experiments": total_samples_all,
                "total_correct_samples": total_correct_all,
                "overall_accuracy": total_correct_all / total_samples_all
                if total_samples_all > 0
                else 0.0,
            },
            "experiment_summary": experiment_metadata,
            "raw_responses_only_correct": all_raw_responses,
        }

        output_path = Path(args.output_path)
        save_json(output_data, output_path)

        logger.info(f"[OK] Combined results saved to {output_path}")
        logger.info(
            f"[Statistics] Total correct samples: {total_correct_all}/{total_samples_all} = {total_correct_all / total_samples_all:.1%}"
        )
        logger.info(
            "💡 Tip: Use --output_format jmedec_dataset to create benchmark dataset format"
        )

    else:
        # Save separate files for each experiment
        output_base = Path(args.output_path).parent / Path(args.output_path).stem

        for result in all_results:
            exp_info = result["experiment_info"]
            filename = f"{exp_info['dataset']}_{exp_info['model']}_{exp_info['template']}_{exp_info['timestamp']}.json"
            output_path = output_base / filename

            save_json(result, output_path)
            logger.info(
                f"Saved {exp_info['dataset']}/{exp_info['model']}/{exp_info['template']} to {output_path}"
            )

        # Also save summary
        summary_data = {
            "metadata": {
                "extraction_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "config_file": str(config_path),
                "task": task,
                "total_experiments": total_experiments,
                "successful_experiments": successful_experiments,
            },
            "experiments": [r["experiment_info"] for r in all_results],
            "statistics": [r["statistics"] for r in all_results],
        }

        summary_path = output_base / "summary.json"
        save_json(summary_data, summary_path)

        logger.info(f"[OK] Separate files saved to {output_base}/")
        logger.info(f"📋 Summary saved to {summary_path}")
        logger.info(
            "💡 Tip: Use --output_format jmedec_dataset to create benchmark dataset format"
        )

    return 0


if __name__ == "__main__":
    exit(main())

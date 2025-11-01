#!/usr/bin/env python3
"""
Script to merge two adopted_questions.json files while avoiding sample_id duplicates.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Union


def load_json_file(file_path: Path) -> Union[Dict, List]:
    """Load JSON file and return the data."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data: Union[Dict, List], file_path: Path) -> None:
    """Save data to JSON file with proper formatting."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def deduplicate_single_file(
    input_path: Path, output_path: Path, strategy: str = "alternating"
) -> None:
    """
    Remove duplicates from a single adopted_questions.json file where multiple models
    generated the same base question (same jmle_id + choice combination).

    Args:
        input_path: Path to input JSON file
        output_path: Path where deduplicated file will be saved
        strategy: Deduplication strategy - "alternating", "first", "best_accuracy"
    """
    # Load file
    data = load_json_file(input_path)
    # Handle both array and dict formats
    if isinstance(data, list):
        samples = data
    else:
        samples = data.get("samples", [])

    print(f"Input file: {len(samples)} samples")

    # Group by base ID (jmle_id + choice)
    grouped_samples = {}
    for sample in samples:
        sample_id = sample.get("sample_id", "")
        # Extract base ID (everything before the last underscore that contains model suffix)
        parts = sample_id.split("_")
        if len(parts) >= 3:
            base_id = "_".join(parts[:-1])  # Remove last part which is model suffix
        else:
            base_id = sample_id

        if base_id not in grouped_samples:
            grouped_samples[base_id] = []
        grouped_samples[base_id].append(sample)

    # Select one sample from each group based on strategy
    selected_samples = []
    duplicates_removed = 0
    alternating_counter = 0

    for base_id, group in grouped_samples.items():
        if len(group) == 1:
            # No duplicates
            selected_samples.append(group[0])
        else:
            # Handle duplicates
            duplicates_removed += len(group) - 1

            if strategy == "alternating":
                # Alternate between models
                selected_idx = alternating_counter % len(group)
                selected_samples.append(group[selected_idx])
                alternating_counter += 1
                print(
                    f"Duplicate {base_id}: selected model {selected_idx + 1}/{len(group)}"
                )

            elif strategy == "first":
                # Always select first occurrence
                selected_samples.append(group[0])
                print(f"Duplicate {base_id}: selected first model")

            elif strategy == "best_accuracy":
                # Select based on accuracy metrics
                best_sample = group[0]
                best_accuracy = best_sample.get("metadata", {}).get(
                    "complete_accuracy", 0
                )

                for sample in group[1:]:
                    accuracy = sample.get("metadata", {}).get("complete_accuracy", 0)
                    if accuracy > best_accuracy:
                        best_sample = sample
                        best_accuracy = accuracy

                selected_samples.append(best_sample)
                print(
                    f"Duplicate {base_id}: selected best accuracy ({best_accuracy:.3f})"
                )

    # Create output data in the same format as input
    if isinstance(data, list):
        output_data = selected_samples
    else:
        output_data = {"samples": selected_samples}
        # Add other top-level keys
        for key in data:
            if key != "samples":
                output_data[key] = data[key]

    # Save result
    save_json_file(output_data, output_path)

    # Statistics
    error_samples = [s for s in selected_samples if s.get("error_flag") == 1]
    no_error_samples = [s for s in selected_samples if s.get("error_flag") == 0]

    print("\nDeduplication completed:")
    print(f"Original samples: {len(samples)}")
    print(f"Deduplicated samples: {len(selected_samples)}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Strategy used: {strategy}")
    print("\nError distribution:")
    print(
        f"Error samples: {len(error_samples)} ({len(error_samples) / len(selected_samples) * 100:.1f}%)"
    )
    print(
        f"No error samples: {len(no_error_samples)} ({len(no_error_samples) / len(selected_samples) * 100:.1f}%)"
    )
    print(f"Output saved to: {output_path}")


def merge_two_files(
    file1_path: Path,
    file2_path: Path,
    output_path: Path,
    file1_suffix: str = "",
    file2_suffix: str = "",
) -> None:
    """
    Merge two adopted_questions.json files without duplicate sample_ids.
    When duplicates exist, alternates between file1 and file2.

    Args:
        file1_path: Path to first JSON file
        file2_path: Path to second JSON file
        output_path: Path where merged file will be saved
        file1_suffix: Suffix to add to sample_ids from file1
        file2_suffix: Suffix to add to sample_ids from file2
    """
    # Load both files
    data1 = load_json_file(file1_path)
    data2 = load_json_file(file2_path)

    # Extract samples from both files
    # Handle both array and dict formats
    samples1 = data1 if isinstance(data1, list) else data1.get("samples", [])
    samples2 = data2 if isinstance(data2, list) else data2.get("samples", [])

    print(f"File 1: {len(samples1)} samples")
    print(f"File 2: {len(samples2)} samples")

    # Create dictionaries for quick lookup by original sample_id
    samples1_dict = {
        sample.get("sample_id"): sample
        for sample in samples1
        if sample.get("sample_id")
    }
    samples2_dict = {
        sample.get("sample_id"): sample
        for sample in samples2
        if sample.get("sample_id")
    }

    # Get all unique sample_ids
    all_sample_ids = set(samples1_dict.keys()) | set(samples2_dict.keys())

    # Track which file to use for duplicates (alternating)
    use_file1_for_duplicate = True
    merged_samples: List[Dict] = []
    duplicates_found = 0

    for sample_id in sorted(all_sample_ids):  # Sort for consistent ordering
        in_file1 = sample_id in samples1_dict
        in_file2 = sample_id in samples2_dict

        if in_file1 and in_file2:
            # Duplicate: alternate between files and add suffix to selected one
            if use_file1_for_duplicate:
                sample_copy = samples1_dict[sample_id].copy()
                sample_copy["sample_id"] = sample_id + file1_suffix
                merged_samples.append(sample_copy)
                print(f"Duplicate {sample_id}: using file 1 with suffix")
            else:
                sample_copy = samples2_dict[sample_id].copy()
                sample_copy["sample_id"] = sample_id + file2_suffix
                merged_samples.append(sample_copy)
                print(f"Duplicate {sample_id}: using file 2 with suffix")
            use_file1_for_duplicate = not use_file1_for_duplicate
            duplicates_found += 1
        elif in_file1:
            # Only in file 1: add suffix
            sample_copy = samples1_dict[sample_id].copy()
            sample_copy["sample_id"] = sample_id + file1_suffix
            merged_samples.append(sample_copy)
        else:
            # Only in file 2: add suffix
            sample_copy = samples2_dict[sample_id].copy()
            sample_copy["sample_id"] = sample_id + file2_suffix
            merged_samples.append(sample_copy)

    # Create merged data structure in the same format as input
    if isinstance(data1, list):
        merged_data = merged_samples
    else:
        merged_data = {"samples": merged_samples}
        # Add any other top-level keys from the original files
        for key in data1:
            if key != "samples":
                merged_data[key] = data1[key]

    # Save merged file
    save_json_file(merged_data, output_path)

    # Calculate error statistics
    error_samples = [
        sample for sample in merged_samples if sample.get("error_flag") == 1
    ]
    no_error_samples = [
        sample for sample in merged_samples if sample.get("error_flag") == 0
    ]

    print("\nMerge completed:")
    print(f"Total samples: {len(merged_samples)}")
    print(f"Duplicates found: {duplicates_found}")
    print(f"From file1: {len(samples1_dict)} samples")
    print(f"From file2: {len(samples2_dict)} samples")
    print("\nError distribution:")
    print(
        f"Error samples: {len(error_samples)} ({len(error_samples) / len(merged_samples) * 100:.1f}%)"
    )
    print(
        f"No error samples: {len(no_error_samples)} ({len(no_error_samples) / len(merged_samples) * 100:.1f}%)"
    )
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge two adopted_questions.json files or deduplicate a single file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  
  # Merge two files:
  python scripts/dataset_construction/step3_model_deduplication.py file1.json file2.json -o merged.json
  
  # Deduplicate single file:
  python scripts/dataset_construction/step3_model_deduplication.py --deduplicate input.json -o output.json
  
  # Deduplicate with specific strategy:
  python scripts/dataset_construction/step3_model_deduplication.py --deduplicate input.json -o output.json --strategy best_accuracy
        """,
    )

    # Mode selection
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Deduplicate a single file instead of merging two files",
    )

    # File arguments - conditional based on mode
    parser.add_argument(
        "file1",
        type=Path,
        help="Path to first JSON file (or input file for deduplication)",
    )
    parser.add_argument(
        "file2",
        type=Path,
        nargs="?",
        help="Path to second JSON file (required for merge mode)",
    )

    # Output
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Output path for result file"
    )

    # Merge mode options
    parser.add_argument(
        "--file1-suffix",
        type=str,
        default="",
        help="Suffix to add to sample_ids from file1 (merge mode)",
    )
    parser.add_argument(
        "--file2-suffix",
        type=str,
        default="",
        help="Suffix to add to sample_ids from file2 (merge mode)",
    )

    # Deduplicate mode options
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["alternating", "first", "best_accuracy"],
        default="alternating",
        help="Deduplication strategy (deduplicate mode)",
    )

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.deduplicate:
        # Deduplication mode
        if args.file2:
            print("Error: --deduplicate mode takes only one input file")
            return 1

        if not args.file1.exists():
            print(f"Error: Input file does not exist: {args.file1}")
            return 1

        # Create output directory if it doesn't exist
        args.output.parent.mkdir(parents=True, exist_ok=True)

        # Perform deduplication
        deduplicate_single_file(args.file1, args.output, args.strategy)

    else:
        # Merge mode
        if not args.file2:
            print("Error: Merge mode requires two input files")
            return 1

        # Validate input files exist
        if not args.file1.exists():
            print(f"Error: File 1 does not exist: {args.file1}")
            return 1

        if not args.file2.exists():
            print(f"Error: File 2 does not exist: {args.file2}")
            return 1

        # Create output directory if it doesn't exist
        args.output.parent.mkdir(parents=True, exist_ok=True)

        # Perform merge
        merge_two_files(
            args.file1, args.file2, args.output, args.file1_suffix, args.file2_suffix
        )

    return 0


if __name__ == "__main__":
    exit(main())


"""
Example usage:

# Merge two separate files:
uv run python scripts/dataset_construction/step3_model_deduplication.py \
    "data/inputs/medec/synthesized_jmle_case_119_v3_q3_filtered_strict_hard/adopted_questions.json" \
    "data/inputs/medec/synthesized_jmle_case_119_v3_r1_filtered_strict_hard/adopted_questions.json" \
    -o "data/inputs/medec/jmedec_119_adopted_questions.json" \
    --file1-suffix "_qwen3-235b-a22b-thinking-2507" \
    --file2-suffix "_deepseek-r1-0528"

# Deduplicate a single file with existing model suffixes (alternating strategy):
uv run python scripts/dataset_construction/step3_model_deduplication.py \
    --deduplicate "data/inputs/medec/synthesized_jmle_case_118_filtered_strict_hard/adopted_questions.json" \
    -o "data/inputs/medec/jmedec_118_adopted_questions.json"

# Deduplicate using best accuracy strategy:
uv run python scripts/dataset_construction/step3_model_deduplication.py \
    --deduplicate "data/inputs/medec/synthesized_jmle_case_118_filtered_strict_hard/adopted_questions.json" \
    -o "data/inputs/medec/jmedec_118_adopted_questions.json" \
    --strategy best_accuracy
"""

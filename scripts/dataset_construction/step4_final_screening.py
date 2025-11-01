#!/usr/bin/env python3
"""
Separate JMEDEC questions by screening results.

This script reads LLM as a judge screening results and separates the original
dataset into valid and invalid questions based on the screening judgment.
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, Any, Optional
import glob
import os


# Language-specific error criteria mappings
ERROR_CRITERIA = {
    "ja": [
        "ambiguous_error",
        "multiple_errors",
        "numerical_error",
        "extra_elements",
        "synthesis_consistency_error",
    ],
    "en": [
        "ambiguous_error",
        "multiple_errors",
        "numerical_error",
        "unrealistic_scenario",
        "inconsistent_context",
    ],
}


def find_latest_results(results_dir: Path) -> Optional[Path]:
    """Find the latest results file in a directory."""
    pattern = str(results_dir / "*" / "predictions.json")
    files = glob.glob(pattern)

    if not files:
        return None

    # Sort by timestamp in filename (assuming YYYYMMDD_HHMMSS format)
    latest_file = max(files, key=lambda x: os.path.basename(os.path.dirname(x)))
    return Path(latest_file)


def load_screening_results(
    base_path: Path, dataset_name: str, model_name: str, language: str
) -> Dict[str, Dict[str, Any]]:
    """Load screening results from both prompt types."""
    screening_results = {}

    # Derive template suffix from language
    template_suffix = language

    screening_base = (
        base_path / "results" / "medec_screening" / dataset_name / model_name
    )

    # Check for detailed_explanation results
    detailed_dir = screening_base / f"detailed_explanation_{template_suffix}"
    detailed_file = find_latest_results(detailed_dir)

    # Check for with_examples results
    examples_dir = screening_base / f"with_examples_{template_suffix}"
    examples_file = find_latest_results(examples_dir)

    templates_found = []

    if detailed_file and detailed_file.exists():
        print(
            f"Loading detailed_explanation_{template_suffix} results from: {detailed_file}"
        )
        with open(detailed_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        detailed_metrics = data.get("detailed_metrics", {}).get("screening_judge", [])
        for sample in detailed_metrics:
            sample_id = sample.get("sample_id")
            if sample_id:
                if sample_id not in screening_results:
                    screening_results[sample_id] = {}
                screening_results[sample_id][
                    f"detailed_explanation_{template_suffix}"
                ] = sample

        templates_found.append(f"detailed_explanation_{template_suffix}")

    if examples_file and examples_file.exists():
        print(f"Loading with_examples_{template_suffix} results from: {examples_file}")
        with open(examples_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        detailed_metrics = data.get("detailed_metrics", {}).get("screening_judge", [])
        for sample in detailed_metrics:
            sample_id = sample.get("sample_id")
            if sample_id:
                if sample_id not in screening_results:
                    screening_results[sample_id] = {}
                screening_results[sample_id][f"with_examples_{template_suffix}"] = (
                    sample
                )

        templates_found.append(f"with_examples_{template_suffix}")

    if not templates_found:
        raise FileNotFoundError("No screening results found")

    print(f"Found results for templates: {templates_found}")
    return screening_results


def consolidate_screening_judgment(
    sample_results: Dict[str, Any], language: str = "en"
) -> Dict[str, Any]:
    """Consolidate screening results from multiple prompts using conservative approach."""

    # Get language-specific error criteria
    error_criteria = ERROR_CRITERIA.get(language, ERROR_CRITERIA["en"])

    # Initialize consolidated results
    consolidated = {
        "is_valid": True,  # Start as valid, set to False if any template finds issues
        "total_issues": 0,
        "explanations": {},
        "prompt_results": sample_results,
    }

    # Initialize all error criteria for this language
    for criterion in error_criteria:
        consolidated[criterion] = 0

    # Check each template result
    for template, result in sample_results.items():
        # If any template marks as invalid, mark consolidated as invalid
        if not result.get("is_valid", True):
            consolidated["is_valid"] = False

        # Take the maximum count for each error type (most conservative)
        consolidated["total_issues"] = max(
            consolidated["total_issues"], result.get("total_issues", 0)
        )

        # Update error criteria dynamically based on language
        for criterion in error_criteria:
            consolidated[criterion] = max(
                consolidated[criterion], result.get(criterion, 0)
            )

        # Store explanations
        consolidated["explanations"][template] = result.get("explanation", "")

    return consolidated


def main():
    parser = argparse.ArgumentParser(
        description="Separate JMEDEC questions by screening results"
    )
    parser.add_argument(
        "--base-path", type=Path, default=Path("."), help="Base path of the project"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/inputs/medec/MEDEC-MS-JSON/test.json"),
        help="Input dataset file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/inputs/medec"),
        help="Output directory for separated files",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="medec_test",
        help="Dataset name for screening results (e.g., medec_test, medrect_ja, medrect_en)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemini-2_5-pro",
        help="Model name used for screening",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["ja", "en"],
        default="en",
        help="Language for screening (determines both template suffix and error criteria)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving output files (only show statistics)",
    )

    args = parser.parse_args()

    base_path = args.base_path
    input_file = base_path / args.input_file
    output_dir = base_path / args.output_dir

    print(f"Processing: {input_file}")
    print(f"Output directory: {output_dir}")

    # Load original dataset
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return 1

    with open(input_file, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    print(f"Loaded {len(original_data)} samples from original dataset")

    # Load screening results
    try:
        screening_results = load_screening_results(
            base_path, args.dataset_name, args.model_name, args.language
        )
        print(f"Loaded screening results for {len(screening_results)} samples")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Process each sample
    valid_samples = []
    invalid_samples = []
    samples_with_screening = []

    stats = {
        "total": len(original_data),
        "screened": 0,
        "valid": 0,
        "invalid": 0,
        "no_screening": 0,
        "error_types": Counter(),
        "total_issues_invalid": 0,  # Total issues count for invalid samples
    }

    for sample in original_data:
        sample_id = sample.get("sample_id")

        # Create a copy with screening information
        enhanced_sample = sample.copy()

        if sample_id in screening_results:
            # Consolidate screening results
            consolidated = consolidate_screening_judgment(
                screening_results[sample_id], args.language
            )
            enhanced_sample["screening_results"] = consolidated

            # Categorize based on validity
            if consolidated["is_valid"]:
                valid_samples.append(enhanced_sample)
                stats["valid"] += 1
            else:
                invalid_samples.append(enhanced_sample)
                stats["invalid"] += 1

                # Add total issues count for average calculation
                stats["total_issues_invalid"] += consolidated.get("total_issues", 0)

                # Count error types dynamically based on language
                error_criteria = ERROR_CRITERIA.get(args.language, ERROR_CRITERIA["en"])
                for criterion in error_criteria:
                    if consolidated.get(criterion, 0) > 0:
                        stats["error_types"][criterion] += 1

            stats["screened"] += 1
        else:
            # No screening data available - default to valid for now
            enhanced_sample["screening_results"] = {
                "is_valid": None,
                "note": "No screening data available",
            }
            valid_samples.append(enhanced_sample)
            stats["no_screening"] += 1

        samples_with_screening.append(enhanced_sample)

    # Save separated files if not disabled
    if not args.no_save:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save separated files
        dataset_basename = args.input_file.stem
        valid_file = output_dir / f"{dataset_basename}_screened_accepted.json"
        invalid_file = output_dir / f"{dataset_basename}_screened_rejected.json"
        with_screening_file = output_dir / f"{dataset_basename}_screened_all.json"

        with open(valid_file, "w", encoding="utf-8") as f:
            json.dump(valid_samples, f, ensure_ascii=False, indent=2)

        with open(invalid_file, "w", encoding="utf-8") as f:
            json.dump(invalid_samples, f, ensure_ascii=False, indent=2)

        with open(with_screening_file, "w", encoding="utf-8") as f:
            json.dump(samples_with_screening, f, ensure_ascii=False, indent=2)

    # Print statistics
    print("\n" + "=" * 50)
    print("SEPARATION RESULTS")
    print("=" * 50)
    print(f"Total samples: {stats['total']}")
    print(f"Samples with screening: {stats['screened']}")
    print(f"Samples without screening: {stats['no_screening']}")
    print(f"Valid samples: {stats['valid']}")
    print(f"Invalid samples: {stats['invalid']}")
    print(f"Validity rate: {stats['valid'] / stats['total']:.1%}")

    # Calculate and display average issues per invalid sample
    if stats["invalid"] > 0:
        avg_issues = stats["total_issues_invalid"] / stats["invalid"]
        print(f"Average issues per invalid sample: {avg_issues:.2f}")

    if stats["error_types"]:
        print("\nError type distribution:")
        for error_type, count in stats["error_types"].most_common():
            print(f"  {error_type}: {count}")

    if not args.no_save:
        print("\nOutput files created:")
        print(f"  Accepted samples: {valid_file}")
        print(f"  Rejected samples: {invalid_file}")
        print(f"  All samples with screening: {with_screening_file}")
    else:
        print("\nFile saving skipped (--no-save flag enabled)")

    # Analyze agreement between templates
    if len(screening_results) > 0:
        template_agreement_stats = analyze_template_agreement(screening_results)
        print("\nTemplate Agreement Analysis:")
        print(
            f"  Samples with both templates: {template_agreement_stats['both_templates']}"
        )
        print(f"  Agreement rate: {template_agreement_stats['agreement_rate']:.1%}")
        if template_agreement_stats["disagreements"]:
            print(f"  Disagreements: {template_agreement_stats['disagreements']}")

    return 0


def analyze_template_agreement(
    screening_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze agreement between different template results."""
    both_templates = 0
    agreements = 0
    disagreements = 0

    for sample_id, results in screening_results.items():
        if len(results) == 2:  # Has both template results
            both_templates += 1

            templates = list(results.keys())
            result1 = results[templates[0]]
            result2 = results[templates[1]]

            valid1 = result1.get("is_valid", True)
            valid2 = result2.get("is_valid", True)

            if valid1 == valid2:
                agreements += 1
            else:
                disagreements += 1

    agreement_rate = (agreements / both_templates) if both_templates > 0 else 0

    return {
        "both_templates": both_templates,
        "agreements": agreements,
        "disagreements": disagreements,
        "agreement_rate": agreement_rate,
    }


if __name__ == "__main__":
    exit(main())

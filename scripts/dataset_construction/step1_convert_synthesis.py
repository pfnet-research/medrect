#!/usr/bin/env python3
"""
Convert JMLE synthesis data to MEDEC format.

This script provides the following features:
1. Convert JMLE synthesis data to MEDEC format
2. Remove bold markers (**text**) (enabled by default)
3. Filter questions containing '下線部' (underlined parts) (enabled by default)
4. Filter questions with numeric choices (1-5) (enabled by default)
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


def remove_bold_markers(text: str) -> str:
    """
    Remove **text** patterns from text, keeping only the content inside.

    Args:
        text: Input text that may contain **bold** markers

    Returns:
        Text with bold markers removed
    """
    if not text:
        return text

    # Replace **text** with text (remove the ** markers)
    pattern = r"\*\*(.*?)\*\*"
    return re.sub(pattern, r"\1", text)


def contains_underlined_part(text: str) -> bool:
    """
    Check if text contains '下線部' (underlined part) pattern.

    Args:
        text: Input text to check

    Returns:
        True if text contains underlined part pattern
    """
    if not text:
        return False

    # Pattern to match '下線部' or '下線 部' or '下 線 部' etc.
    pattern = r"下\s*線\s*部"
    return bool(re.search(pattern, text))


def has_numeric_choices(data: Dict[str, Any]) -> bool:
    """
    Check if the original question has numeric choice values (1-5).

    Args:
        data: Original JMLE data containing choice information

    Returns:
        True if choices are numeric values 1-5
    """
    if not data:
        return False

    # Get choices from the original data structure
    choices = data.get("choices", {})
    if not choices or not isinstance(choices, dict):
        return False

    # Check if choices are exactly {"a": "1", "b": "2", "c": "3", "d": "4", "e": "5"}
    expected_choices = {"a": "1", "b": "2", "c": "3", "d": "4", "e": "5"}

    # Compare choices with expected numeric pattern
    return choices == expected_choices


def calculate_theoretical_statistics(
    input_dir: Path, model_list: List[str], template_list: List[str], converter
) -> Dict[str, Any]:
    """
    Preliminary analysis to calculate theoretical number of synthesized samples.
    """
    print("\nTheoretical Value Calculation & Data Analysis")
    print("=" * 50)

    theoretical_stats = {
        "total_jmle_questions": 0,
        "theoretical_samples_per_model": 0,
        "filtered_questions_underlined": 0,
        "expected_choices_per_question": 5,  # JMLE questions typically have 5 choices
        "models_analyzed": 0,
        "sample_distribution": {},
    }

    # Calculate theoretical values using data from the first model
    for model_name in model_list:
        model_files = []
        for template in template_list:
            template_pattern = f"{model_name}/{template}/*/raw_responses.json"
            pattern_files = list(input_dir.glob(template_pattern))
            model_files.extend(pattern_files)

        if not model_files:
            continue

        latest_file = max(model_files, key=lambda p: p.stat().st_mtime)

        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            total_questions = len(raw_data["interactions"])
            filtered_questions = 0

            # Calculate number of excluded samples when filtering is enabled
            if converter.filter_underlined:
                for interaction in raw_data["interactions"]:
                    original_data = interaction.get("sample_data", {})
                    original_question = original_data.get("question", "")

                    # Check for underlined parts
                    if contains_underlined_part(original_question):
                        filtered_questions += 1
                        continue

                    # Check for numeric choices (1-5)
                    if has_numeric_choices(original_data):
                        filtered_questions += 1

            valid_questions = total_questions - filtered_questions
            theoretical_samples = (
                valid_questions * theoretical_stats["expected_choices_per_question"]
            )

            theoretical_stats.update(
                {
                    "total_jmle_questions": total_questions,
                    "filtered_questions_underlined": filtered_questions,
                    "valid_questions_after_filtering": valid_questions,
                    "theoretical_samples_per_model": theoretical_samples,
                    "models_analyzed": 1,
                    "analyzed_model": model_name,
                    "analyzed_file": str(latest_file),
                }
            )

            print(f"  [OK] Total JMLE questions: {total_questions:,}")
            print(
                f"  [OK] Choices per question: {theoretical_stats['expected_choices_per_question']}"
            )

            if converter.filter_underlined:
                print(
                    f"Filtered (underlined part + numeric choices): {filtered_questions:,} questions"
                )
                print(f"  [OK] Valid questions: {valid_questions:,}")
            else:
                print(f"  [OK] Valid questions: {total_questions:,} (no filtering)")

            print(
                f"  [Target] Theoretical synthetic samples: {theoretical_samples:,} per model"
            )
            print(
                f"     (= {valid_questions:,} questions × {theoretical_stats['expected_choices_per_question']} choices)"
            )

            if len(model_list) > 1:
                total_theoretical = theoretical_samples * len(model_list)
                print(
                    f"  [Statistics] Total for all {len(model_list)} models: {total_theoretical:,} samples"
                )

            break  # Calculation complete with first model

        except Exception as e:
            print(f"  [ERROR] {model_name} analysis error: {e}")
            continue

    if theoretical_stats["models_analyzed"] == 0:
        print("  [WARNING] No data found for theoretical value calculation")

    print("=" * 50)
    return theoretical_stats


class JMLEToMEDECConverter:
    """Extract and convert JMLE synthesis data to MEDEC format."""

    def __init__(self, debug_mode=False, filter_underlined=True, remove_bold=True):
        self.debug_mode = debug_mode
        self.filter_underlined = filter_underlined
        self.remove_bold = remove_bold
        self.conversion_stats = {
            "total_samples": 0,
            "successful_conversions": 0,
            "failed_conversions": 0,
            "filtered_underlined": 0,  # Number of samples excluded due to '下線部' (underlined parts)
            "total_records": 0,
            "correct_records": 0,
            "error_records": 0,
        }
        self.failed_samples = []
        self.partial_samples = []  # Record partially successful cases (1-4 records extracted)
        self.filtered_samples = []  # Samples excluded due to '下線部' (underlined parts)

    def parse_jmle_response(self, response_content: str) -> List[Dict[str, Any]]:
        """Parse JMLE synthesis response and extract MEDEC format data."""
        records = []

        if not response_content or len(response_content.strip()) == 0:
            return records

        # Extract choice-specific records supporting multiple patterns
        patterns = [
            # Standard format: ### Correct record/Error record synthesized from choice a
            r"###\s*選択肢\s*([a-e])\s*(?:（.+?）)?\s*から合成した\s*(Correct record|Error record)([^#]*?)(?=###\s*選択肢[a-e]|$)",
            # deepseek-r1 format 1: #### Correct record synthesized from choice d (correct answer)
            r"####\s*選択肢\s*([a-e])\s*(?:\([^)]*\))?\s*から合成した\s*(Correct record|Error record)([^#]*?)(?=####\s*選択肢[a-e]|###|$)",
            # deepseek-r1 format 2: ### Correct record synthesized from choice b (correct answer)
            r"###\s*選択肢\s*([a-e])\s*(?:\([^)]*\))?\s*から合成した\s*(Correct record|Error record)([^#]*?)(?=###\s*選択肢[a-e]|$)",
            # deepseek-r1 format 3: Pattern with spaces - ### Correct record synthesized from choice e
            r"###\s*選択肢\s+([a-e])\s+から合成した\s*(Correct record|Error record)([^#]*?)(?=###\s*選択肢\s+[a-e]|$)",
            # deepseek-r1 format 4: Alternative pattern - #### Correct record synthesized from choice c (correct answer)
            r"####\s*選択肢\s*([a-e])\s*から合成した\s*(Correct record|Error record)([^#]*?)(?=####\s*選択肢[a-e]|###|$)",
            # deepseek-r1 format 5: Pattern with description - ### Correct record synthesized from choice c (correct answer choice)
            r"###\s*選択肢\s*([a-e])\s*から合成した\s*(Correct record|Error record)(?:\s*[（(][^)）]*[)）])?\s*([^#]*?)(?=###\s*選択肢[a-e]|$)",
            # deepseek-r1 format 6: More flexible pattern - infer record type
            r"###[#]?\s*選択肢\s*([a-e])\s*(?:\([^)]*\))?\s*(?:から合成した|から合成)?\s*([^#\n]*?)([^#]*?)(?=###[#]?\s*選択肢[a-e]|$)",
        ]

        # Try all patterns and adopt the one with the most matches
        best_matches = []
        best_matched_choices = set()

        for pattern in patterns:
            current_matches = []
            current_matched = set()
            matches = list(
                re.finditer(pattern, response_content, re.DOTALL | re.IGNORECASE)
            )

            # Add while removing duplicates
            for match in matches:
                choice_letter = match.group(1)
                if choice_letter not in current_matched:
                    current_matches.append(match)
                    current_matched.add(choice_letter)

            # Adopt the pattern with more matches
            if len(current_matches) > len(best_matches):
                best_matches = current_matches
                best_matched_choices = current_matched

        choice_matches = best_matches
        matched_choices = best_matched_choices

        # Fallback: search for individual choices if no pattern finds enough matches
        if len(choice_matches) < 4:
            fallback_matches = []
            for choice in ["a", "b", "c", "d", "e"]:
                # Skip choices that have already been matched
                if choice in matched_choices:
                    continue

                # Broader search patterns
                fallback_patterns = [
                    rf"###[#]*\s*選択肢\s*{choice}\s*[^#]*?(\d+\.\s*[^\n]+)",
                    rf"選択肢\s*{choice}\s*[^#]*?から[^#]*?(\d+\.\s*[^\n]+)",
                    rf"選択肢\s*{choice}[^#]*?記録[^#]*?(\d+\.\s*[^\n]+)",
                ]

                for fb_pattern in fallback_patterns:
                    fb_matches = re.search(
                        fb_pattern, response_content, re.DOTALL | re.IGNORECASE
                    )
                    if fb_matches:
                        # Create simple match object
                        start_pos = fb_matches.start()
                        # Extract entire section for this choice
                        section_match = re.search(
                            rf"(選択肢\s*{choice}[^#]*?)(?=選択肢\s*[a-e]|$)",
                            response_content[start_pos:],
                            re.DOTALL | re.IGNORECASE,
                        )
                        if section_match:
                            # Add pseudo match object
                            class FallbackMatch:
                                def __init__(self, choice, content):
                                    self.choice = choice
                                    self.content = content

                                def group(self, n):
                                    if n == 1:
                                        return self.choice
                                    elif n == 2:
                                        return "Error record"  # Default
                                    elif n == 3:
                                        return self.content

                            fallback_matches.append(
                                FallbackMatch(choice, section_match.group(1))
                            )
                            matched_choices.add(
                                choice
                            )  # Also record choices matched by fallback
                        break

            if fallback_matches:
                choice_matches.extend(fallback_matches)

        for match in choice_matches:
            choice_letter = match.group(1)
            record_type_raw = match.group(2)
            choice_content = match.group(3).strip()

            # Normalize record_type (handle various notations from r1-0528)
            if (
                "Correct record" in record_type_raw
                or "正答" in record_type_raw
                or "正しい" in record_type_raw
            ):
                record_type = "Correct record"
            elif (
                "Error record" in record_type_raw
                or "誤答" in record_type_raw
                or "エラー" in record_type_raw
            ):
                record_type = "Error record"
            else:
                # For pattern 3, infer from header content
                header_line = record_type_raw.lower()
                if "正答" in header_line or "正しい" in header_line:
                    record_type = "Correct record"
                elif "誤答" in header_line or "エラー" in header_line:
                    record_type = "Error record"
                else:
                    # Default to error record
                    record_type = "Error record"

            # Extract sentences from numbered list (stop before error meta-information)
            # Identify error meta-information section
            content_parts = choice_content.split("エラータイプ:")
            medical_text = content_parts[
                0
            ].strip()  # Only the part before error meta-information

            sentence_pattern = r"(\d+)\.\s*([^\n]+(?:\n(?!\d+\.)[^\n]*)*)"
            sentences = re.findall(sentence_pattern, medical_text, re.MULTILINE)

            # Alternative method for sentence extraction
            if not sentences:
                lines = [
                    line.strip()
                    for line in medical_text.split("\n")
                    if line.strip() and not line.strip().startswith("エラー")
                ]
                sentences = [(str(i + 1), line) for i, line in enumerate(lines[:10])]

            if not sentences:
                continue

            # Extract error information
            error_info = None
            if record_type == "Error record":
                error_type_match = re.search(
                    r"エラータイプ:\s*([^\n]+)", choice_content
                )
                error_num_match = re.search(r"エラー文番号:\s*(\d+)", choice_content)
                error_sentence_match = re.search(
                    r"エラー文:\s*([^\n]+)", choice_content
                )
                corrected_sentence_match = re.search(
                    r"修正文:\s*([^\n]+)", choice_content
                )

                if (
                    error_type_match
                    and error_sentence_match
                    and corrected_sentence_match
                ):
                    error_info = {
                        "error_type": error_type_match.group(1).strip(),
                        "error_sentence_number": error_num_match.group(1).strip()
                        if error_num_match
                        else None,
                        "error_sentence": error_sentence_match.group(1).strip(),
                        "corrected_sentence": corrected_sentence_match.group(1).strip(),
                    }

            records.append(
                {
                    "choice": choice_letter,
                    "type": record_type,
                    "sentences": sentences,
                    "error_info": error_info,
                    "raw_content": choice_content,
                }
            )

        return records

    def convert_to_medec_format(
        self,
        sample_id: str,
        original_data: Dict[str, Any],
        records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert JMLE synthesis data to MEDEC format."""
        medec_samples = []

        for record in records:
            if not record["sentences"]:
                continue

            # Prepare sentence data (aligned with MEDEC format)
            full_text_parts = []
            sentences_lines = []

            for i, (_, sentence) in enumerate(record["sentences"]):
                clean_sentence = sentence.strip()
                # Remove bold markers if enabled
                if self.remove_bold:
                    clean_sentence = remove_bold_markers(clean_sentence)
                full_text_parts.append(clean_sentence)
                # Store sentences in 1-indexed markdown format
                sentences_lines.append(f"{i + 1}. {clean_sentence}")

            full_text = " ".join(full_text_parts)
            sentences_formatted = "\n".join(sentences_lines)

            # Process error information
            error_flag = 1 if record["type"] == "Error record" else 0
            error_type = None
            error_sentence_id = None
            error_sentence = None
            corrected_sentence = None
            corrected_text = None

            if record["error_info"]:
                error_type = record["error_info"]["error_type"]
                error_sentence = record["error_info"]["error_sentence"]
                corrected_sentence = record["error_info"]["corrected_sentence"]

                # Remove bold markers from error info if enabled
                if self.remove_bold:
                    if error_sentence:
                        error_sentence = remove_bold_markers(error_sentence)
                    if corrected_sentence:
                        corrected_sentence = remove_bold_markers(corrected_sentence)

                # Set error_sentence_id as 1-indexed
                if record["error_info"]["error_sentence_number"]:
                    try:
                        error_sentence_id = int(
                            record["error_info"]["error_sentence_number"]
                        )
                    except ValueError:
                        pass

                # Identify error sentence (when there's no number)
                if error_sentence_id is None and error_sentence:
                    for i, (_, sentence) in enumerate(record["sentences"]):
                        if error_sentence in sentence:
                            error_sentence_id = i + 1  # 1-indexed
                            break

                # Generate corrected text
                if error_sentence and corrected_sentence:
                    corrected_text = full_text.replace(
                        error_sentence, corrected_sentence
                    )
                    # Remove bold markers from corrected text if enabled
                    if self.remove_bold:
                        corrected_text = remove_bold_markers(corrected_text)
                else:
                    corrected_text = full_text

            # Get JMLE question ID
            jmle_question_id = (
                original_data.get("metadata", {})
                .get("original_data", {})
                .get("id", sample_id)
            )

            # Exclude unnecessary MEDEC keys from original_jmle_data
            clean_jmle_data = {}
            for key, value in original_data.items():
                # Exclude MEDEC keys (these are managed at top level)
                if key not in [
                    "metadata",
                    "clinical_narrative",
                    "sentences",
                    "error_flag",
                    "error_type",
                    "error_sentence_id",
                    "error_sentence",
                    "corrected_sentence",
                ]:
                    clean_jmle_data[key] = value

            # Data consistency check: skip if it's an error record but error_sentence_id cannot be determined
            if error_flag == 1 and error_sentence_id is None:
                if self.debug_mode:
                    print(
                        f"  [WARNING] {jmle_question_id}_{record['choice']}: Error record but cannot identify error_sentence_id - skipped"
                    )
                self.conversion_stats["failed_conversions"] += 1
                continue

            # Create MEDEC format sample (eliminate redundancy)
            medec_sample = {
                "sample_id": f"{jmle_question_id}_{record['choice']}",
                "sentences": sentences_formatted,
                "error_flag": error_flag,
                "error_type": error_type,
                "error_sentence_id": error_sentence_id,
                "error_sentence": error_sentence,
                "corrected_sentence": corrected_sentence,
                "corrected_text": corrected_text,
                "metadata": {
                    "jmle_question_id": jmle_question_id,
                    "choice": record["choice"],
                    "record_type": "correct"
                    if record["type"] == "Correct record"
                    else "error",
                    "original_jmle_data": clean_jmle_data,
                    "conversion_timestamp": datetime.now().isoformat(),
                },
            }

            medec_samples.append(medec_sample)

        return medec_samples

    def convert_model_results(
        self, model_name: str, results_path: Path
    ) -> List[Dict[str, Any]]:
        """Convert results from a single model to MEDEC format."""
        print(f"\n[Statistics] {model_name} conversion started...")

        with open(results_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        all_medec_samples = []

        for interaction in raw_data["interactions"]:
            self.conversion_stats["total_samples"] += 1

            sample_id = interaction["sample_id"]

            # Check if sample should be filtered out due to underlined parts or numeric choices
            if self.filter_underlined:
                original_data = interaction.get("sample_data", {})
                original_question = original_data.get("question", "")

                # Check for underlined parts
                if contains_underlined_part(original_question):
                    self.conversion_stats["filtered_underlined"] += 1
                    if self.debug_mode:
                        self.filtered_samples.append(
                            {
                                "sample_id": sample_id,
                                "model": model_name,
                                "reason": "Contains underlined part (下線部)",
                                "original_question": original_question[:200] + "..."
                                if len(original_question) > 200
                                else original_question,
                            }
                        )
                        print(f"{sample_id}: Filtered (contains underlined part)")
                    continue

                # Check for numeric choices (1-5)
                if has_numeric_choices(original_data):
                    self.conversion_stats["filtered_underlined"] += 1
                    if self.debug_mode:
                        choices = original_data.get("choices", {})
                        self.filtered_samples.append(
                            {
                                "sample_id": sample_id,
                                "model": model_name,
                                "reason": "Contains numeric choices (1-5)",
                                "choices": str(choices),
                            }
                        )
                        print(f"{sample_id}: Filtered (contains numeric choices 1-5)")
                    continue

            # Get response content (structure varies by model)
            if "content" in interaction["raw_response"]:
                response_content = interaction["raw_response"]["content"]
            elif (
                "choices" in interaction["raw_response"]
                and len(interaction["raw_response"]["choices"]) > 0
            ):
                response_content = interaction["raw_response"]["choices"][0]["message"][
                    "content"
                ]
            else:
                self.conversion_stats["failed_conversions"] += 1
                continue

            if not response_content:
                self.conversion_stats["failed_conversions"] += 1
                continue

            try:
                # Get original data
                original_data = interaction.get("sample_data", {})

                # Parse response
                records = self.parse_jmle_response(response_content)

                if not records:
                    self.conversion_stats["failed_conversions"] += 1
                    if self.debug_mode:
                        self.failed_samples.append(
                            {
                                "sample_id": sample_id,
                                "model": model_name,
                                "reason": "No records extracted",
                                "raw_response": response_content[:500] + "..."
                                if len(response_content) > 500
                                else response_content,
                            }
                        )
                        print(f"  [ERROR] {sample_id}: Response parsing failed")
                        print(f"     Raw response preview: {response_content[:200]}...")
                    continue

                # Convert to MEDEC format
                medec_samples = self.convert_to_medec_format(
                    sample_id, original_data, records
                )

                # Update statistics
                self.conversion_stats["successful_conversions"] += 1
                self.conversion_stats["total_records"] += len(records)

                # Record partially successful cases (less than theoretical 5 records)
                if len(records) < 5:
                    self.partial_samples.append(
                        {
                            "sample_id": sample_id,
                            "model": model_name,
                            "extracted_count": len(records),
                            "missing_count": 5 - len(records),
                            "extracted_choices": [r["choice"] for r in records],
                            "raw_response": response_content[:800] + "..."
                            if len(response_content) > 800
                            else response_content,
                        }
                    )
                    if self.debug_mode:
                        print(
                            f"  [WARNING] {sample_id}: Partial extraction ({len(records)}/5 records)"
                        )

                for sample in medec_samples:
                    if sample["error_flag"] == 1:
                        self.conversion_stats["error_records"] += 1
                    else:
                        self.conversion_stats["correct_records"] += 1

                all_medec_samples.extend(medec_samples)

            except Exception as e:
                self.conversion_stats["failed_conversions"] += 1
                if self.debug_mode:
                    self.failed_samples.append(
                        {
                            "sample_id": sample_id,
                            "model": model_name,
                            "reason": f"Exception: {str(e)}",
                            "raw_response": response_content[:500] + "..."
                            if len(response_content) > 500
                            else response_content,
                        }
                    )
                print(f"  [WARNING] Sample {sample_id} conversion error: {e}")

        print(f"  Conversion successful: {len(all_medec_samples)} records")

        return all_medec_samples

    def save_results(
        self, all_samples: Dict[str, List[Dict[str, Any]]], output_dir: Path
    ):
        """Save results in JSON format."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save by model
        for model_name, samples in all_samples.items():
            model_file = output_dir / f"{model_name}_medec_data.json"

            with open(model_file, "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)

            file_size = model_file.stat().st_size / (1024 * 1024)
            print(f"  {model_name}: {len(samples)} records ({file_size:.2f} MB)")

        # Save integrated data (add model name to sample_id to ensure uniqueness)
        all_combined = []
        for model_name, samples in all_samples.items():
            for sample in samples:
                # Add model name to sample_id for integrated file
                original_sample_id = sample["sample_id"]
                sample["sample_id"] = f"{original_sample_id}_{model_name}"
                sample["source_model"] = model_name
                all_combined.append(sample)

        combined_file = output_dir / "all_models_medec_data.json"
        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(all_combined, f, ensure_ascii=False, indent=2)

        print(f"  Integrated data: {len(all_combined)} records")

        # Save metadata
        metadata = {
            "conversion_timestamp": datetime.now().isoformat(),
            "total_samples": self.conversion_stats["total_samples"],
            "successful_conversions": self.conversion_stats["successful_conversions"],
            "failed_conversions": self.conversion_stats["failed_conversions"],
            "total_medec_records": len(all_combined),
            "conversion_stats": self.conversion_stats,
        }

        metadata_file = output_dir / "conversion_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Convert JMLE synthesis data to MEDEC format and perform quality evaluation"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Base directory path for JMLE synthesis results",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for MEDEC format data",
    )
    parser.add_argument(
        "--models",
        type=str,
        required=False,
        help="Target models for conversion (comma-separated). Auto-detected from input directory if omitted",
    )
    parser.add_argument(
        "--templates",
        type=str,
        required=False,
        help='Target templates for conversion (comma-separated). Uses "simple_choice_synthesis" if omitted',
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: Display details of failed responses",
    )
    parser.add_argument(
        "--no-filter-underlined",
        action="store_true",
        help="Disable filtering of original questions containing underlined parts and numeric choices (1-5) (enabled by default)",
    )
    parser.add_argument(
        "--no-remove-bold",
        action="store_true",
        help="Disable removal of bold markers (**text**) (enabled by default)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    # Determine template list (from arguments or default)
    if args.templates:
        template_list = [t.strip() for t in args.templates.split(",")]
        print(f"Specified templates: {', '.join(template_list)}")
    else:
        template_list = ["simple_choice_synthesis"]
        print(f"Default templates: {', '.join(template_list)}")

    # Determine model list (from arguments or auto-detection)
    if args.models:
        model_list = [m.strip() for m in args.models.split(",")]
        print(f"Specified models: {', '.join(model_list)}")
    else:
        # Auto-detect available models by walking directories
        model_list = []
        for model_dir in input_dir.iterdir():
            if model_dir.is_dir():
                # Check if raw_responses.json exists for any of the specified templates
                has_template = False
                for template in template_list:
                    template_files = list(
                        model_dir.glob(f"{template}/*/raw_responses.json")
                    )
                    if template_files:
                        has_template = True
                        break
                if has_template:
                    model_list.append(model_dir.name)

        if not model_list:
            print(f"Error: No usable model results found in {input_dir}")
            print(
                "Expected file structure: {model_name}/{template}/*/raw_responses.json"
            )
            print(f"Target templates to search: {', '.join(template_list)}")
            return 1

        model_list.sort()  # Sort alphabetically
        print(f"Auto-detected models: {', '.join(model_list)}")

    converter = JMLEToMEDECConverter(
        debug_mode=args.debug,
        filter_underlined=not args.no_filter_underlined,
        remove_bold=not args.no_remove_bold,
    )

    print("=" * 80)
    print("JMLE→MEDEC Conversion & Quality Evaluation")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target models: {', '.join(model_list)}")
    print(f"Target templates: {', '.join(template_list)}")

    # Processing options
    options = []
    if converter.filter_underlined:
        options.append("Underlined part + numeric choice filtering enabled")
    if converter.remove_bold:
        options.append("Bold marker removal enabled")
    if options:
        print(f"Processing options: {', '.join(options)}")

    # Pre-analysis for theoretical value calculation
    theoretical_stats = calculate_theoretical_statistics(
        input_dir, model_list, template_list, converter
    )

    all_results = {}

    for model_name in model_list:
        # Search for result files among specified templates
        model_files = []
        for template in template_list:
            template_pattern = f"{model_name}/{template}/*/raw_responses.json"
            pattern_files = list(input_dir.glob(template_pattern))
            model_files.extend(pattern_files)

        if not model_files:
            print(
                f"[WARNING]  Result file for {model_name} not found (templates: {', '.join(template_list)})"
            )
            continue

        # Use the latest one if multiple template files exist
        latest_file = max(model_files, key=lambda p: p.stat().st_mtime)

        # Show which template was used
        template_from_path = latest_file.parent.parent.name
        print(f"{model_name}: Using template '{template_from_path}'")

        try:
            medec_samples = converter.convert_model_results(model_name, latest_file)
            all_results[model_name] = medec_samples
        except Exception as e:
            print(f"[ERROR] {model_name} conversion error: {e}")

    if not all_results:
        print("No data could be converted.")
        return 1

    # Save results
    print("\n[Save] Saving results:")
    converter.save_results(all_results, output_dir)

    # Final statistics - comparison of theoretical and actual results
    print("\n[Statistics] Conversion Results & Theoretical Comparison:")
    print("=" * 50)

    # Actual conversion results
    total_extracted = (
        converter.conversion_stats["correct_records"]
        + converter.conversion_stats["error_records"]
    )
    processed_samples = converter.conversion_stats["total_samples"]

    # Comparison with theoretical values
    if theoretical_stats["models_analyzed"] > 0:
        theoretical_per_model = theoretical_stats["theoretical_samples_per_model"]
        actual_per_model = total_extracted // len(all_results) if all_results else 0

        print("[Analysis] JMLE Question Count Analysis:")
        print(
            f"  • Total questions: {theoretical_stats['total_jmle_questions']:,}questions"
        )
        if converter.filter_underlined:
            print(
                f"  • Filtered (underlined part + numeric choices): {theoretical_stats['filtered_questions_underlined']:,}questions"
            )
            print(
                f"  • Valid questions: {theoretical_stats['valid_questions_after_filtering']:,}questions"
            )
        else:
            print(
                f"  • Valid questions: {theoretical_stats['total_jmle_questions']:,}questions (no filtering)"
            )

        print("[Target] Theoretical vs actual:")
        print(f"  • Theoretical samples per model: {theoretical_per_model:,}samples")
        print(
            f"    (= {theoretical_stats['valid_questions_after_filtering']:,}questions × 5 choices)"
        )
        print(f"  • Actual samples per model: {actual_per_model:,}samples")

        if theoretical_per_model > 0:
            achievement_rate = (actual_per_model / theoretical_per_model) * 100
            print(f"  • Achievement rate: {achievement_rate:.1f}%")
            print(
                f"  • Missing samples: {theoretical_per_model - actual_per_model:,}samples per model"
            )

        if len(all_results) > 1:
            total_theoretical = theoretical_per_model * len(all_results)
            print(
                f"  • Total for all models (theoretical): {total_theoretical:,}samples"
            )
            print(f"  • Total for all models (actual): {total_extracted:,}samples")

    print("\nDetailed Conversion Statistics:")
    print(f"  • Processed samples: {processed_samples:,}")
    print(
        f"  • Successful conversions: {converter.conversion_stats['successful_conversions']:,}"
    )
    print(
        f"  • Failed conversions: {converter.conversion_stats['failed_conversions']:,}"
    )
    print(
        f"  • Partial success: {len(converter.partial_samples):,}cases (1-4 records extracted)"
    )
    print(f"  • Total MEDEC records: {total_extracted:,}")
    print(f"  • Correct records: {converter.conversion_stats['correct_records']:,}")
    print(f"  • Error record: {converter.conversion_stats['error_records']:,}")
    if converter.filter_underlined:
        print(
            f"  • Filtered (underlined part + numeric choices): {converter.conversion_stats['filtered_underlined']:,}"
        )

    # Output detailed information for failed samples in debug mode
    if args.debug and converter.failed_samples:
        print("\n[Debug] Debug Info: Complete Failure Sample Details")
        print("=" * 80)
        for failed in converter.failed_samples:
            print(f"Sample ID: {failed['sample_id']}")
            print(f"Model: {failed['model']}")
            print(f"Reason: {failed['reason']}")
            print("Raw Response (first 300 chars):")
            print(f"  {failed['raw_response'][:300]}...")
            print("-" * 60)

    # Output detailed information for partially successful samples
    if args.debug and converter.partial_samples:
        print("\n[Debug] Debug Info: Partially Successful Sample Details")
        print("=" * 80)
        for partial in converter.partial_samples:
            print(f"Sample ID: {partial['sample_id']}")
            print(f"Model: {partial['model']}")
            print(
                f"Extracted: {partial['extracted_count']}/5 cases (missing: {partial['missing_count']} cases)"
            )
            print(f"Extracted choices: {', '.join(partial['extracted_choices'])}")
            missing_choices = set(["a", "b", "c", "d", "e"]) - set(
                partial["extracted_choices"]
            )
            print(f"Missing choices: {', '.join(sorted(missing_choices))}")
            print("Raw Response (first 400 chars):")
            print(f"  {partial['raw_response'][:400]}...")
            print("-" * 60)

    # Output detailed information for samples filtered by '下線部' (underlined parts)
    if args.debug and converter.filtered_samples:
        print("\n[Debug] Debug Info: Samples Filtered by Underlined Parts")
        print("=" * 80)
        for filtered in converter.filtered_samples:
            print(f"Sample ID: {filtered['sample_id']}")
            print(f"Model: {filtered['model']}")
            print(f"Reason: {filtered['reason']}")
            if "original_question" in filtered:
                print("Original Question (first 200 chars):")
                print(f"  {filtered['original_question']}")
            elif "choices" in filtered:
                print("Choices:")
                print(f"  {filtered['choices']}")
            print("-" * 60)

    print(f"\n[OK] Conversion complete: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())

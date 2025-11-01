# Training Data Construction

This directory contains scripts for constructing training data with reasoning annotations using DeepSeek-R1-0528.

## Overview

The training data construction pipeline consists of 3 steps:

1. **Generate Reasoning**: Synthesize reasoning annotations using DeepSeek-R1-0528
2. **Clean Meta-References**: Remove meta-references from reasoning text
3. **Extract Correct Samples**: Extract only samples with correct predictions

## Step 1: Generate Reasoning with DeepSeek-R1-0528

Synthesize reasoning annotations for training data:

```bash
# Synthesize reasoning for Japanese training data
uv run --env-file .env batch \
  --config configs/batch/synthesize_reasoning_for_medrect-train/medrect_ja_train.yaml

# Synthesize reasoning for English training data
uv run --env-file .env batch \
  --config configs/batch/synthesize_reasoning_for_medrect-train/medrect_en_train.yaml
```

**Outputs**:
- `results/medec/medrect_ja_train_raw/deepseek-r1-0528/0_shot_reasoning_ja/TIMESTAMP/raw_responses.json`
- `results/medec/medec_train_and_val/deepseek-r1-0528/0_shot_reasoning_cheat_en/TIMESTAMP/raw_responses.json`

## Step 2: Clean Meta-References

Remove meta-references from reasoning text:

```bash
# Clean Japanese reasoning
uv run python scripts/training_data/clean_reasoning_ja.py \
  --input results/medec/medrect_ja_train_raw/deepseek-r1-0528/0_shot_reasoning_ja/TIMESTAMP/raw_responses.json \
  --output data/medrect/cleaned_reasoning_ja.json

# Clean English reasoning
uv run python scripts/training_data/clean_reasoning_en.py \
  --input results/medec/medec_train_and_val/deepseek-r1-0528/0_shot_reasoning_cheat_en/TIMESTAMP/raw_responses.json \
  --output data/medrect/cleaned_reasoning_en.json
```

**Outputs**:
- `data/medrect/cleaned_reasoning_ja.json`
- `data/medrect/cleaned_reasoning_en.json`

## Step 3: Extract Correct Samples

Extract samples where DeepSeek-R1-0528 produced correct responses:

```bash
# Extract correct samples for Japanese training data
uv run python scripts/training_data/extract_correct_samples.py \
  --config configs/batch/synthesize_reasoning_for_medrect-train/medrect_ja_train.yaml \
  --output-path data/medrect/medrect-ja-train.json

# Extract correct samples for English training data
uv run python scripts/training_data/extract_correct_samples.py \
  --config configs/batch/synthesize_reasoning_for_medrect-train/medrect_en_train.yaml \
  --output-path data/medrect/medrect-en-train.json
```

**Outputs**:
- `data/medrect/medrect-ja-train.json` - Japanese training data
- `data/medrect/medrect-en-train.json` - English training data

## Scripts

- `clean_reasoning_ja.py` - Clean meta-references from Japanese reasoning text
- `clean_reasoning_en.py` - Clean meta-references from English reasoning text
- `extract_correct_samples.py` - Extract samples with correct predictions from batch evaluation results

# Dataset Construction Pipeline

This directory contains scripts for constructing the MedRECT-ja benchmark dataset from JMLE (Japanese Medical Licensing Examination) questions.

## Overview

The complete dataset construction pipeline consists of 4 steps:

1. **Data Synthesis**: Transform JMLE questions into clinical texts using LLMs
2. **Quality Filtering**: Filter synthesized samples using validation models
3. **Model Deduplication**: Remove duplicates by alternately selecting samples
4. **Final Quality Screening**: Apply LLM-as-a-Judge screening

## Step 1: Data Synthesis

Transform JMLE questions into clinical texts using LLMs:

```bash
# 1. Run synthesis with DeepSeek-R1-0528 and Qwen3-235B-A22B-Thinking-2507
uv run --env-file .env batch \
  --config configs/batch/synthesize_medrect-ja_from_jmle/synthesis_from_jmle.yaml

# 2. Convert synthesis results to MEDEC format
uv run python scripts/dataset_construction/step1_convert_synthesis.py \
  --input results/jmle/case_all \
  --output data/medrect/step1_output
```

**Output**: `data/medrect/step1_output/all_models_medec_data.json`

## Step 2: Quality Filtering

Filter synthesized samples using 11 validation models:

```bash
uv run python scripts/dataset_construction/step2_quality_filtering.py \
  --global \
  --stage1-min-models 1 --stage1-max-models 7 --stage2-max-diff 0.3 \
  --results-dir results/medec/medrect_ja_step1 \
  --input-file data/medrect/medrect-ja-step1-synthesized.json \
  --output-dir data/medrect/step2_output
```

**Output**: `data/medrect/step2_output/medrect-ja-step2-filtered.json`

## Step 3: Model Deduplication

Remove duplicates by alternately selecting samples from each synthesis model:

```bash
uv run python scripts/dataset_construction/step3_model_deduplication.py \
  --deduplicate \
  data/medrect/medrect-ja-step2-filtered.json \
  -o data/medrect/medrect-ja-step3-deduplicated.json \
  --strategy alternating
```

**Output**: `data/medrect/medrect-ja-step3-deduplicated.json`

## Step 4: Final Quality Screening

Apply LLM-as-a-Judge screening with Gemini 2.5 Pro:

```bash
# Run screening evaluation
uv run --env-file .env batch \
  --config configs/batch/screening_medrect/medrect_ja_screening.yaml

# Separate by screening results
uv run python scripts/dataset_construction/step4_final_screening.py \
  --dataset-name medrect_ja_step3_screened \
  --model-name gemini-2_5-pro \
  --language ja \
  --input-file data/medrect/medrect-ja-step3-deduplicated.json \
  --output-dir data/medrect
```

**Outputs**:
- `data/medrect/medrect-ja-step3-deduplicated_screened_accepted.json` - Accepted samples (final benchmark)
- `data/medrect/medrect-ja-step3-deduplicated_screened_rejected.json` - Rejected samples
- `data/medrect/medrect-ja-step3-deduplicated_screened_all.json` - All samples with screening metadata

## Scripts

- `step1_convert_synthesis.py` - Convert synthesis results to MEDEC format
- `step2_quality_filtering.py` - Filter samples using validation models
- `step3_model_deduplication.py` - Remove duplicate samples
- `step4_final_screening.py` - Separate samples by LLM-as-a-Judge screening results

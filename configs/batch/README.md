# Batch Configuration

This directory contains configuration files for orchestrating evaluation experiments. Batch configs combine tasks, models, datasets, and templates to run evaluations.

## Basic Structure

A batch configuration file specifies:

```yaml
task: medec                 # Task name (from configs/tasks/)

models:                     # List of models to evaluate (from configs/models/)
  - gpt-4_1
  - claude-sonnet-4-thinking
  - qwen3-32b-local

datasets:                   # List of datasets to use (defined in task config)
  - medrect_ja
  - medrect_en

templates:                  # List of prompt templates (auto-discovered)
  - 0_shot_ja
  - 0_shot_en

output_dir: results         # Where to save results (default: results/)
```

## Available Configs

- **`medrect_ja_all.yaml`**: Evaluate all models on MedRECT-ja (Japanese benchmark)
- **`medrect_en_all.yaml`**: Evaluate all models on MedRECT-en (English benchmark)

Subdirectories contain specialized configs for dataset construction and training data preparation.

## Usage

Run a batch evaluation with:

```bash
uv run --env-file .env batch --config configs/batch/medrect_ja_all.yaml
```

## Creating a New Batch Config

1. Create a new YAML file (e.g., `my_experiment.yaml`)
2. Specify task, models, datasets, and templates
3. Run with: `uv run --env-file .env batch --config configs/batch/my_experiment.yaml`

**Example** - Evaluate GPT-4.1 on both languages:

```yaml
task: medec

models:
  - gpt-4_1

datasets:
  - medrect_ja
  - medrect_en

templates:
  - 0_shot_ja
  - 0_shot_en
```

This will run 4 evaluations (1 model × 2 datasets × 2 templates).

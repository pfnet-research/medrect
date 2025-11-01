# Task Configuration

This directory contains configuration files that define evaluation tasks, including datasets, metrics, and component classes.

## Main Configuration: `medec.yaml`

The primary task configuration for medical error detection and correction evaluation.

### Key Sections

#### 1. Components
Specifies which sample and template classes to use:

```yaml
components:
  sample_class: "medec"      # Uses MEDECSample class
  template_class: "medec"    # Uses MEDECTemplate class
```

#### 2. Datasets
Defines available datasets and their file locations:

```yaml
data:
  dir: "data"
  datasets:
    medrect_ja:
      file: "medrect/medrect-ja-step4-accepted.json"
      loader: "medecjson"
    medrect_en:
      file: "medrect/medrect-en-accepted.json"
      loader: "medecjson"
    medec_test:
      file: "medec/medec-test.json"
      loader: "medecjson"
```

#### 3. Metrics
Configures evaluation metrics:

```yaml
metrics:
  error_detection:
    class: "errordetection"
    parser: "errordetection"

  sentence_extraction:
    class: "sentenceextraction"
    parser: "sentenceextraction"

  error_correction_full:
    class: "errorcorrection"
    parser: "errorcorrection"
    tokenization_method: "auto"     # Auto-detects Japanese/English
    use_bertscore: true
    bertscore_model: "microsoft/deberta-xlarge-mnli"
    use_bleurt: true
    bleurt_checkpoint: "/path/to/bleurt/BLEURT-20"  # Set your path here
```

##### BLEURT Setup

The `error_correction_full` metric uses BLEURT for advanced semantic similarity evaluation. To enable:

1. Download BLEURT-20 checkpoint:
   ```bash
   git clone https://github.com/google-research/bleurt.git
   cd bleurt
   wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
   unzip BLEURT-20.zip
   ```

2. Update `bleurt_checkpoint` path in `medec.yaml`:
   ```yaml
   bleurt_checkpoint: "/path/to/bleurt/BLEURT-20"
   ```

Note: BLEURT requires GPU and substantial memory. If unavailable, set `use_bleurt: false`.

#### 4. Defaults
Default settings for evaluations:

```yaml
defaults:
  template: "0_shot_en"
  dataset: "medrect_en"
  metrics:
    - "error_detection"
    - "sentence_extraction"
    - "error_correction_full"
```

## Other Task Configs

- **`jmle.yaml`**: Synthesize MedRECT-ja from JMLE
- **`medec_screening.yaml`**: LLM-as-a-Judge quality screening
- **`medec_error_type.yaml`**: Error type classification

## Adding a New Dataset

1. Place your dataset file in `data/` directory (JSON format)

2. Add entry to `medec.yaml`:

```yaml
data:
  datasets:
    my_dataset:
      file: "path/to/my_dataset.json"
      loader: "medecjson"  # Use existing loader for MEDEC format
```

3. Use in batch config:

```yaml
datasets:
  - my_dataset
```

## Dataset File Format

Datasets should follow the MEDEC JSON format:

```json
{
  "data": [
    {
      "id": "sample_001",
      "sentences": "1. Sentence one.\n2. Sentence two.",
      "error_flag": 1,
      "error_sentence": "Sentence with error",
      "corrected_sentence": "Corrected sentence"
    }
  ]
}
```

## Available Templates

Templates are auto-discovered from `src/llm_eval/templates/medec.py`:

- **English prompts**: `0_shot_en`, `2_shot_en`, `5_shot_en`
- **Japanese prompts**: `0_shot_ja`, `2_shot_ja`, `5_shot_ja`
- **Reasoning prompts**: `0_shot_reasoning_en`, `0_shot_reasoning_ja`

Templates with `_ja` suffix use Japanese, `_en` use English.

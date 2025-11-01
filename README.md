# MedRECT: A Medical Reasoning Benchmark for Error Correction in Clinical Texts

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation and datasets for **MedRECT**, the first cross-lingual benchmark (Japanese/English) for medical error detection and correction in clinical texts.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Datasets](#datasets)
6. [Citation](#citation)
7. [License](#license)

## Overview

Large language models (LLMs) show increasing promise in medical applications, but their ability to **detect and correct errors in clinical texts**—a prerequisite for safe deployment—remains under-evaluated, particularly beyond English. MedRECT addresses this gap by providing:

- **Cross-lingual evaluation** (Japanese/English) for medical error correction
- **Three progressive subtasks**: Error Detection, Error Sentence Extraction, and Error Correction
- **Scalable automated pipeline** for creating high-quality benchmarks from medical licensing examinations
- **Diverse error types** including diagnosis, monitoring/management, physical findings, procedures, and medication
- **Comprehensive evaluation** of 9 contemporary LLMs spanning proprietary, open-weight, and reasoning families

## Installation

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/pfnet-research/medrect.git
cd medrect

# Install dependencies (basic: API evaluation + viewer)
uv sync

# For local LLM inference
uv sync --group vllm
```

### Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Required for evaluation:
# - OPENROUTER_API_KEY (for OpenAI models)
# - AZURE_OPENAI_API_KEY (for Azure OpenAI)
```

### BLEURT Setup (Optional)

The `error_correction_full` metric requires the [BLEURT-20 checkpoint](https://github.com/google-research/bleurt). Download and configure the path in `configs/tasks/medec.yaml`. See the [Task Configuration Guide](configs/tasks/README.md#bleurt-setup) for detailed setup instructions.

## Quick Start

### Evaluating Models on MedRECT

```bash
# Evaluate a model on MedRECT-ja (Japanese)
uv run --env-file .env batch \
  --task medec \
  --models gpt-4_1 \
  --datasets medrect_ja \
  --templates 0_shot_ja

# Evaluate on MedRECT-en (English)
uv run --env-file .env batch \
  --task medec \
  --models gpt-4_1 \
  --datasets medrect_en \
  --templates 0_shot_en
```

### Using Configuration Files

```bash
# Run evaluation with a config file (Japanese)
uv run --env-file .env batch \
  --config configs/batch/medrect_ja_all.yaml

# Run evaluation with a config file (English)
uv run --env-file .env batch \
  --config configs/batch/medrect_en_all.yaml
```

### Viewing Results

After running evaluations, you can view the results using the interactive viewer:

```bash
# Launch the interactive results viewer
uv run viewer
```

The viewer displays evaluation results (`*raw_responses.json`, `*predictions.json`) from the `results/` directory. Run the evaluation commands above to generate results, which will be automatically available in the viewer.

## Configuration

The evaluation system uses a three-level configuration hierarchy in the `configs/` directory:

- **`configs/tasks/`**: Define evaluation tasks, datasets, and metrics ([details](configs/tasks/README.md))
  - Specify which datasets to use (MedRECT-ja, MedRECT-en, etc.)
  - Configure evaluation metrics (error detection, sentence extraction, error correction)

- **`configs/models/`**: Configure LLM models and inference parameters ([details](configs/models/README.md))
  - API-based models (OpenAI, Azure, OpenRouter, etc.)
  - Local models (vLLM)

- **`configs/batch/`**: Orchestrate evaluation experiments ([details](configs/batch/README.md))
  - Combine tasks, models, datasets, and templates
  - Run multiple evaluations in parallel

See the README.md in each subdirectory for detailed configuration instructions.

## Datasets

### Benchmark Data

| Dataset | Samples | Error % | Correct % | Source |
|---------|---------|---------|-----------|--------|
| **MedRECT-ja** | 663 | 55.4% | 44.6% | JMLE 2024-2025 |
| **MedRECT-en** | 458 | 53.1% | 46.9% | MEDEC MS Subset Test |

To reproduce the MedRECT-ja benchmark dataset construction from JMLE questions, see the [Dataset Construction Pipeline documentation](scripts/dataset_construction/README.md).

### Training Data

| Dataset | Samples | Error % | Correct % | Source |
|---------|---------|---------|-----------|--------|
| **MedRECT-ja-train** | 5,538 | 65.2% | 34.8% | JMLE 2018-2023 |
| **MedRECT-en-train** | 2,439 | 51.0% | 49.0% | MEDEC MS Subset Training & Validation |

To construct training data with reasoning, see the [Training Data Construction documentation](scripts/training_data/README.md).

## Citation

If you use MedRECT in your research, please cite our paper:

```bibtex
@article{iwase2025medrect,
  title={MedRECT: A Medical Reasoning Benchmark for Error Correction in Clinical Texts},
  author={Iwase, Naoto and Okuyama, Hiroki and Iwasawa, Junichiro},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

### Source Code

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Datasets

Datasets in `data/` are licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

Our MedRECT datasets are derived from the following sources:

- **JMLE (Japanese Medical Licensing Examinations)**
  - Source: Ministry of Health, Labour and Welfare (厚生労働省)
  - License: [PDL-1.0](https://www.digital.go.jp/resources/open_data/public_data_license_v1.0) (compatible with CC-BY-4.0)
  - Used for: MedRECT-ja

- **MEDEC MS Subset**
  - Source: Ben Abacha, A., Yim, W., Fu, Y., Sun, Z., Yetisgen, M., Xia, F., & Lin, T. (2024). MEDEC: A Benchmark for Medical Error Detection and Correction in Clinical Notes. arXiv preprint arXiv:2412.19260.
  - Repository: [https://github.com/abachaa/MEDEC](https://github.com/abachaa/MEDEC)
  - License: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
  - Used for: MedRECT-en

## Acknowledgments

We are grateful to:
- The creators of MEDEC for providing the foundational methodology and dataset
- The Ministry of Health, Labour and Welfare (厚生労働省) for making the Japanese Medical Licensing Examination data publicly available

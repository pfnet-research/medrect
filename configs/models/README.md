# Model Configuration

This directory contains configuration files for LLM models used in evaluation. Models are organized into three main config files based on their deployment method.

## Configuration Files

### `openai.yaml`
API-based models accessed via OpenRouter or direct APIs.

**Example models:**
- OpenAI: `gpt-5-medium`, `o3-medium`, `gpt-4_1`
- Anthropic: `claude-sonnet-4-thinking`
- Google: `gemini-2_5-pro`
- DeepSeek: `deepseek-r1-0528`, `deepseek-v3-0324`
- Qwen: `qwen3-235b-a22b-thinking-2507`

**Required environment variables:**
- `OPENROUTER_API_KEY` for OpenRouter models
- `PLAMO_API_KEY` for PLaMo models

### `azure_openai.yaml`
Microsoft Azure OpenAI models.

**Example models:**
- `azure_gpt5_medium`, `azure_gpt5_high`
- `azure_o3_medium`, `azure_o3_high`

**Required environment variables:**
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`

### `vllm.yaml`
Local models running with vLLM (requires GPU).

**Example models:**
- Qwen3 series: `qwen3-8b-local`, `qwen3-32b-local`
- QwQ: `qwq-32b-local`
- Fine-tuned: `preferred-medrect-32b`

**Features:**
- Supports reasoning mode (thinking) toggle via `enable_thinking` parameter
- Configurable sampling parameters (temperature, top_p, top_k)
- GPU memory management settings

### `vllm_lora.yaml`
LoRA fine-tuned models using vLLM.

## Basic Model Configuration Structure

```yaml
model_name:
  client:                      # Client settings
    api_key_env: "ENV_VAR"    # Environment variable for API key
    base_url: "https://..."   # API endpoint
  request:                     # Request parameters
    model: "model_id"         # Model identifier
    extra_body:               # Model-specific parameters
      reasoning:
        effort: "medium"      # Reasoning effort (for reasoning models)
```

## Adding a New Model

### For API Models (openai.yaml)

1. Add a new entry following this pattern:

```yaml
my-model-name:
  client:
    api_key_env: "OPENROUTER_API_KEY"
    base_url: "https://openrouter.ai/api/v1"
  request:
    model: "provider/model-id"
```

2. Set the required API key in `.env`

3. Use in batch config: `models: [my-model-name]`

### For Local Models (vllm.yaml)

1. Add model configuration:

```yaml
my-local-model:
  client:
    model: "HuggingFace/model-path"
    gpu_memory_utilization: 0.90
  sampling:
    temperature: 0.6
    max_tokens: 32768
```

2. Download the model from HuggingFace

3. Ensure sufficient GPU memory

## Environment Setup

Create a `.env` file with your API keys:

```bash
OPENROUTER_API_KEY=your_key_here
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

See `.env.example` for a template.

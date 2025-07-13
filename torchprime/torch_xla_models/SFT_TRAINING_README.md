# SFT (Supervised Fine-Tuning) Training Guide

## Overview

The training infrastructure supports both pre-training and SFT modes:
- **Pre-training**: All tokens can be masked
- **SFT**: Only response tokens are masked, instruction tokens remain unmasked (automatic masking)

## Quick Start

```bash
# SFT Training
./recipes/train_qwen3_1.7b_sft.sh

# Pre-training (original)
./recipes/train_qwen3_1.7b.sh
```

## SFT Data Formats

### 1. Alpaca Format (Recommended)
```json
{
  "instruction": "What is the capital of France?",
  "output": "The capital of France is Paris.",
  "system": "You are a helpful assistant."
}
```

### 2. ShareGPT Format
```json
{
  "conversations": [
    {"from": "human", "value": "What is the capital of France?"},
    {"from": "assistant", "value": "The capital of France is Paris."}
  ]
}
```

### 3. Custom Format
```json
{
  "instruction": "What is the capital of France?",
  "response": "The capital of France is Paris.",
  "system": "You are a helpful assistant."
}
```

## Configuration

### Basic SFT Command
```bash
python torchprime/torch_xla_models/train.py \
    data=sft \
    model=flex-qwen-1b \
    training_mode=sft \
    data.dataset_name=your_dataset_name \
    data.sft.format=alpaca \
    model.attention_kernel=default
```

### OpenCoder Dataset Example
```bash
python torchprime/torch_xla_models/train.py \
    data=sft \
    model=flex-qwen-1b \
    training_mode=sft \
    data.dataset_name=OpenCoder-LLM/opc-sft-stage1 \
    data.dataset_config_name=filtered_infinity_instruct \
    data.sft.format=alpaca \
    model.attention_kernel=default
```

### Mixed OpenCoder Datasets Example
```bash
python torchprime/torch_xla_models/train.py \
    data=sft_mixed \
    model=flex-qwen-1b \
    training_mode=sft \
    data.sft.format=alpaca \
    model.attention_kernel=default
```

### SFT Data Config
```yaml
data:
  sft:
    format: alpaca  # or "sharegpt", "custom"
    include_system_prompt: true
    instruction_response_separator: "\n\n### Response:\n"
    custom_format:
      instruction_field: "instruction"
      response_field: "response"
      system_field: "system"
```

### Mixed SFT Data Config
```yaml
data:
  # Multiple HuggingFace datasets with weights
  hf_datasets:
    - name: OpenCoder-LLM/opc-sft-stage1
      config: filtered_infinity_instruct
      weight: 0.4
    - name: OpenCoder-LLM/opc-sft-stage1
      config: largescale_diverse_instruct
      weight: 0.3
    - name: OpenCoder-LLM/opc-sft-stage1
      config: realuser_instruct
      weight: 0.3
  
  sft:
    format: alpaca
    include_system_prompt: true
    instruction_response_separator: "\n\n### Response:\n"
```

## Training Process

1. **Data Loading**: Instruction-response pairs loaded
2. **Processing**: `instruction + separator + response` sequence created
3. **Masking**: Only response tokens masked during diffusion
4. **Training**: Model learns to generate responses while preserving instruction context

## Key Differences

| Aspect | Pre-training | SFT |
|--------|-------------|-----|
| Masking | All tokens | Only response tokens |
| Context | Full sequence | Instruction preserved |
| Data format | Raw text | Instruction-response pairs |

## Troubleshooting

### Common Issues
- **"src_mask must be provided"**: Ensure `training_mode=sft` is set
- **"Unsupported format"**: Use "alpaca", "sharegpt", or "custom"
- **Dataset errors**: Check dataset name and accessibility

### Debug Data Format
```python
from datasets import load_dataset
dataset = load_dataset("your_dataset_name")
print(dataset["train"][0])
```

## Popular Datasets
- **Alpaca**: `tatsu-lab/alpaca`
- **ShareGPT**: `anon8231489123/ShareGPT_Vicuna_unfiltered`
- **Dolly**: `databricks/databricks-dolly-15k`
- **CodeAlpaca**: `sahil2801/CodeAlpaca-20k`
- **OpenCoder**: `OpenCoder-LLM/opc-sft-stage1` (configs: `filtered_infinity_instruct`, `largescale_diverse_instruct`, `realuser_instruct`) 
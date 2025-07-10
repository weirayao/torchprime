# SFT (Supervised Fine-Tuning) Training Guide

This guide explains how to use the extended training infrastructure for SFT (Supervised Fine-Tuning) with the Qwen3ForCausalLM model.

## Overview

The training infrastructure now supports both pre-training and SFT modes:

- **Pre-training mode**: All tokens can be masked (original behavior)
- **SFT mode**: Only response tokens are masked, instruction/context tokens remain unmasked

## Key Components

### 1. Model Changes
- Extended `Qwen3ForCausalLM.forward()` method with SFT support
- Added `create_sft_src_mask()` helper method
- Backward compatible with existing pre-training code

### 2. Data Processing
- `SFTDataCollator`: Handles instruction-response pairs
- `create_sft_dataset()`: Processes raw datasets for SFT
- Support for multiple formats: Alpaca, ShareGPT, custom

### 3. Training Infrastructure
- Modified `train.py` to support SFT mode
- Automatic src_mask generation from instruction lengths
- Configurable training mode via `training_mode` parameter

## Configuration

### Basic SFT Configuration

Add to your config or command line:

```bash
training_mode=sft
data=sft
data.dataset_name=your_dataset_name
data.sft.format=alpaca
```

### SFT Data Configuration

Create a config file or specify via command line:

```yaml
data:
  sft:
    format: alpaca  # or "sharegpt", "custom"
    include_system_prompt: true
    instruction_response_separator: "\n\n### Response:\n"
    custom_format:
      instruction_field: "instruction"
      response_field: "response"
      system_field: "system"  # optional
```

## Supported Data Formats

### 1. Alpaca Format
```json
{
  "instruction": "What is the capital of France?",
  "output": "The capital of France is Paris.",
  "system": "You are a helpful assistant."  // optional
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
  "system": "You are a helpful assistant."  // optional
}
```

## Usage Examples

### 1. Basic SFT Training

```bash
python torchprime/torch_xla_models/train.py \
    data=sft \
    model=flex-qwen-1b \
    training_mode=sft \
    data.dataset_name=your_dataset_name \
    data.sft.format=alpaca \
    global_batch_size=8 \
    max_steps=1000
```

### 2. SFT with Custom Format

```bash
python torchprime/torch_xla_models/train.py \
    data=sft \
    model=flex-qwen-1b \
    training_mode=sft \
    data.dataset_name=your_custom_dataset \
    data.sft.format=custom \
    data.sft.custom_format.instruction_field=prompt \
    data.sft.custom_format.response_field=answer \
    global_batch_size=8 \
    max_steps=1000
```

### 3. SFT from Pre-trained Checkpoint

```bash
python torchprime/torch_xla_models/train.py \
    data=sft \
    model=flex-qwen-1b \
    training_mode=sft \
    data.dataset_name=your_dataset_name \
    checkpoint_dir=gs://your-bucket/pretrained-checkpoint \
    resume_from_checkpoint=latest \
    global_batch_size=8 \
    max_steps=1000
```

## Training Scripts

### Pre-training (Original)
```bash
./recipes/train_qwen3_1.7b.sh
```

### SFT Training
```bash
./recipes/train_qwen3_1.7b_sft.sh
```

**Note**: Update the dataset name in the SFT script before running:
```bash
data.dataset_name=your_actual_dataset_name
```

## Data Preparation

### 1. HuggingFace Dataset
If your dataset is on HuggingFace Hub:
```bash
data.dataset_name=your_dataset_name
data.dataset_config_name=your_config_name  # if needed
```

### 2. GCS Dataset
If your dataset is in Google Cloud Storage:
```bash
data.gcs_dataset_names=["gs://your-bucket/dataset1", "gs://your-bucket/dataset2"]
data.weights=[0.7, 0.3]  # mixing weights
```

### 3. Local Dataset
For local datasets, you'll need to upload to GCS or use HuggingFace format.

## Training Process

### What Happens During SFT Training

1. **Data Loading**: Raw instruction-response pairs are loaded
2. **Processing**: Each example is converted to a sequence with instruction + separator + response
3. **Masking**: Only response tokens are masked during diffusion process
4. **Training**: Model learns to generate responses while preserving instruction context

### Key Differences from Pre-training

| Aspect | Pre-training | SFT |
|--------|-------------|-----|
| Masking | All tokens | Only response tokens |
| Context | Full sequence | Instruction preserved |
| Objective | General language modeling | Instruction following |
| Data format | Raw text | Instruction-response pairs |

## Monitoring Training

### Logs
The training logs will show:
- Training mode (pretrain/sft)
- Loss values
- Learning rate
- Step information

### WandB Integration
Training metrics are automatically logged to WandB:
- `train/loss`: Training loss
- `train/ppl`: Perplexity
- `train/step_time`: Step execution time
- `train/lr`: Learning rate

## Troubleshooting

### Common Issues

1. **"src_mask must be provided for SFT training mode"**
   - Ensure `training_mode=sft` is set
   - Check that data format is correctly specified

2. **"Unsupported format"**
   - Verify `data.sft.format` is one of: "alpaca", "sharegpt", "custom"
   - For custom format, ensure field names are correctly specified

3. **Dataset loading errors**
   - Check dataset name and format
   - Verify dataset is accessible (HuggingFace Hub or GCS)

### Debug Tips

1. **Check data format**:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("your_dataset_name")
   print(dataset["train"][0])  # Check first example
   ```

2. **Verify instruction lengths**:
   - The training logs should show instruction lengths being processed
   - Check that instruction_lengths are reasonable (not 0 or too large)

3. **Monitor loss**:
   - SFT loss should be lower than pre-training loss
   - Loss should decrease over time

## Migration from Pre-training

### Existing Pre-training Code
No changes needed! The default behavior remains the same:
```bash
python torchprime/torch_xla_models/train.py \
    data=wikitext \
    model=flex-qwen-1b \
    # ... other parameters
```

### Adding SFT Support
Simply add SFT-specific parameters:
```bash
python torchprime/torch_xla_models/train.py \
    data=sft \
    model=flex-qwen-1b \
    training_mode=sft \
    data.dataset_name=your_sft_dataset \
    # ... other parameters
```

## Best Practices

1. **Start with pre-trained model**: Load from a pre-trained checkpoint for better results
2. **Use appropriate learning rate**: SFT typically uses lower learning rates than pre-training
3. **Monitor overfitting**: SFT can overfit quickly, use validation data if available
4. **Data quality**: Ensure high-quality instruction-response pairs
5. **Format consistency**: Use consistent formatting across your dataset

## Example Datasets

### Popular SFT Datasets
- **Alpaca**: Stanford's instruction-following dataset
- **ShareGPT**: Conversation datasets from ShareGPT
- **Dolly**: Databricks' instruction dataset
- **CodeAlpaca**: Code instruction dataset

### Custom Dataset Creation
Create your own dataset in Alpaca format:
```json
[
  {
    "instruction": "Your instruction here",
    "output": "Expected response here",
    "system": "Optional system prompt"
  }
]
``` 
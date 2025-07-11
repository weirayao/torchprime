# SFT (Supervised Fine-Tuning) Support for Qwen3ForCausalLM

This document describes the extended functionality of `Qwen3ForCausalLM` to support both pre-training and SFT (Supervised Fine-Tuning) modes.

## Overview

The model now supports two training modes:
1. **Pre-training mode**: All tokens can be masked (original behavior)
2. **SFT mode**: Only response tokens are masked, instruction/context tokens remain unmasked

## Key Changes

### 1. Extended Forward Method

The `forward` method now accepts additional parameters:

```python
def forward(
    self,
    input_ids: torch.LongTensor,
    labels: torch.LongTensor | None = None,
    attention_mask: torch.FloatTensor | None = None,
    src_mask: torch.BoolTensor | None = None,  # NEW: Source mask for SFT
    training_mode: str = "pretrain",           # NEW: Training mode selection
) -> tuple[torch.FloatTensor, torch.FloatTensor | None]:
```

### 2. Automatic Source Mask Generation

The SFT training pipeline automatically generates source masks from instruction lengths during data preprocessing. No manual mask creation is required.

## Usage Examples

### Pre-training Mode (Default)

```python
# Pre-training: all tokens can be masked
logits, loss = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    training_mode="pretrain"  # or omit this parameter (default)
)
```

### SFT Mode (Automatic Masking)

```python
# SFT: src_mask is automatically generated from instruction lengths
logits, loss = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    src_mask=batch["src_mask"],  # Automatically provided by data collator
    training_mode="sft"
)
```

### SFT Mode with Custom Mask

```python
# Create custom src_mask manually
src_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
src_mask[0, :4] = True   # First 4 tokens are instruction
src_mask[1, :6] = True   # First 6 tokens are instruction

logits, loss = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    src_mask=src_mask,
    training_mode="sft"
)
```

## Source Mask Semantics

- **`src_mask[i, j] = True`**: Token at position `j` in sequence `i` is instruction/context (should NOT be masked)
- **`src_mask[i, j] = False`**: Token at position `j` in sequence `i` is response (should be masked for training)

## Implementation Details

### Pre-training Mode
- `src_mask` is automatically set to all `False` (all tokens maskable)
- `maskable_mask = ~src_mask` becomes all `True`
- All tokens can be masked during diffusion process

### SFT Mode
- `src_mask` must be provided by the user
- `maskable_mask = ~src_mask` (only response tokens are maskable)
- Instruction/context tokens remain unmasked during diffusion process

### Diffusion Process
The diffusion-based masking process (`transition` function) only masks tokens where `maskable_mask` is `True`. This ensures that:
- In pre-training: all tokens can be masked
- In SFT: only response tokens are masked, preserving instruction context

## Error Handling

- If `training_mode="sft"` but `src_mask` is not provided, a `ValueError` is raised
- The model maintains backward compatibility with existing pre-training code

## Example Script

See `sft_example.py` for complete working examples of both modes.

## Migration Guide

### For Existing Pre-training Code

No changes needed! The default behavior remains the same:

```python
# This still works exactly as before
logits, loss = model(input_ids=input_ids, attention_mask=attention_mask)
```

### For New SFT Code

The SFT training pipeline automatically handles src_mask generation. Simply use the training script:

```bash
./recipes/train_qwen3_1.7b_sft.sh
```

Or manually specify SFT parameters:

```python
logits, loss = model(
    input_ids=input_ids, 
    attention_mask=attention_mask, 
    src_mask=batch["src_mask"],  # Provided by data collator
    training_mode="sft"
)
``` 
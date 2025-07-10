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

### 2. New Helper Method

Added `create_sft_src_mask` method to easily create source masks for SFT:

```python
def create_sft_src_mask(
    self,
    input_ids: torch.LongTensor,
    instruction_lengths: torch.LongTensor,
) -> torch.BoolTensor:
```

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

### SFT Mode with Helper Method

```python
# SFT: only response tokens are masked
instruction_lengths = torch.tensor([4, 6])  # Length of instruction for each sequence
src_mask = model.create_sft_src_mask(input_ids, instruction_lengths)

logits, loss = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    src_mask=src_mask,
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

Add the required parameters:

```python
# Before (would raise error)
logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, training_mode="sft")

# After (correct)
src_mask = model.create_sft_src_mask(input_ids, instruction_lengths)
logits, loss = model(
    input_ids=input_ids, 
    attention_mask=attention_mask, 
    src_mask=src_mask,
    training_mode="sft"
)
``` 
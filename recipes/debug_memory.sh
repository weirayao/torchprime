#!/bin/bash

echo "=== TPU Memory Diagnostic Script ==="
echo "Date: $(date)"
echo "TPU VM: sfr-shiyu-wang-v5p-32"
echo "Zone: us-central1-a"
echo ""

echo "=== Testing Basic Model Loading ==="
echo "This will test if the model can be loaded without training:"

XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python -c "
import torch
import torch_xla
import torch_xla.runtime as xr
from omegaconf import OmegaConf
import hydra
from torchprime.torch_xla_models.model_utils import initialize_model_class

# Basic configuration
config = OmegaConf.create({
    'model': {
        'model_class': 'flex.Qwen3ForCausalLM',
        'tokenizer_name': 'Qwen/Qwen3-1.7B',
        'hidden_size': 2048,
        'num_hidden_layers': 28,
        'num_attention_heads': 16,
        'num_key_value_heads': 8,
        'intermediate_size': 6144,
        'vocab_size': 151936,
        'max_position_embeddings': 40960,
        'torch_dtype': 'bfloat16',
        'attention_kernel': None,
        'block_masking_probability': 0.3,
        'mask_block_sizes': [2, 4, 8],
        'truncate_probability': 0.25,
        'prefix_probability': 0.25,
        'sharding': {},
        'remat': {
            'activation_checkpoint_layers': ['Qwen3DecoderLayer'],
            'optimization_barrier_layers': ['Qwen3DecoderLayer'],
            'scan_layers': None
        }
    }
})

print('Configuration loaded successfully')
print('Attempting to initialize model...')

try:
    with torch_xla.device():
        model = initialize_model_class(config.model, load_from_hf=False)
    print('Model initialized successfully')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
except Exception as e:
    print(f'Model initialization failed: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "=== Testing with Scan Layers ==="
echo "This will test if scan layers cause the issue:"

XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python -c "
import torch
import torch_xla
import torch_xla.runtime as xr
from omegaconf import OmegaConf
import hydra
from torchprime.torch_xla_models.model_utils import initialize_model_class

# Configuration with scan layers
config = OmegaConf.create({
    'model': {
        'model_class': 'flex.Qwen3ForCausalLM',
        'tokenizer_name': 'Qwen/Qwen3-1.7B',
        'hidden_size': 2048,
        'num_hidden_layers': 28,
        'num_attention_heads': 16,
        'num_key_value_heads': 8,
        'intermediate_size': 6144,
        'vocab_size': 151936,
        'max_position_embeddings': 40960,
        'torch_dtype': 'bfloat16',
        'attention_kernel': None,
        'block_masking_probability': 0.3,
        'mask_block_sizes': [2, 4, 8],
        'truncate_probability': 0.25,
        'prefix_probability': 0.25,
        'sharding': {},
        'remat': {
            'activation_checkpoint_layers': ['Qwen3DecoderLayer'],
            'optimization_barrier_layers': ['Qwen3DecoderLayer'],
            'scan_layers': 'model.layers'
        }
    }
})

print('Configuration with scan layers loaded successfully')
print('Attempting to initialize model with scan...')

try:
    with torch_xla.device():
        model = initialize_model_class(config.model, load_from_hf=False)
    print('Model with scan initialized successfully')
except Exception as e:
    print(f'Model with scan initialization failed: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "=== Memory Usage Summary ==="
echo "If both tests pass, the issue is likely in the training loop or batch size."
echo "If scan test fails, scan layers are causing the issue."
echo "If basic test fails, there's a fundamental model loading issue."

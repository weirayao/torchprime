#!/usr/bin/env python3
"""Test script to verify Qwen2ForCausalLM import works correctly."""

import sys
import os

# Add the torchprime directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'torchprime'))

def test_qwen2_import():
    """Test importing Qwen2ForCausalLM from the flex module."""
    try:
        from torchprime.torch_xla_models.flex import Qwen2ForCausalLM
        print("✅ Successfully imported Qwen2ForCausalLM from flex module")
        
        # Test creating a simple config
        from omegaconf import OmegaConf
        config_dict = {
            'hidden_size': 1536,
            'num_attention_heads': 12,
            'num_hidden_layers': 28,
            'vocab_size': 151936,
            'intermediate_size': 8960,
            'hidden_act': 'silu',
            'rms_norm_eps': 1e-6,
            'rope_theta': 1000000,
            'head_dim': 128,
            'num_key_value_heads': 8,
            'max_position_embeddings': 32768,
            'sliding_window': 32768,
            'use_sliding_window': False,
            'attention_bias': False,
            'attention_dropout': 0.0,
            'torch_dtype': 'bfloat16',
            'use_cache': True,
            'tie_word_embeddings': True,
            'initializer_range': 0.02,
            'bos_token_id': 151643,
            'eos_token_id': 151643,
            'pad_token_id': 151643,
            'mask_token_id': 151669,
            'tokenizer_name': 'Qwen/Qwen2.5-Coder-1.5B',
            'max_window_layers': 28,
            'rope_scaling': None,
            'attention_kernel': 'flash_attention'
        }
        
        config = OmegaConf.create(config_dict)
        
        # Test model instantiation
        model = Qwen2ForCausalLM(config)
        print("✅ Successfully created Qwen2ForCausalLM instance")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False

if __name__ == "__main__":
    success = test_qwen2_import()
    sys.exit(0 if success else 1)

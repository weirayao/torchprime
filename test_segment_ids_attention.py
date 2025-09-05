#!/usr/bin/env python3
"""Unit test to compare flash attention with and without segment_ids on TPU v5p-8"""

import os
import torch
import torch_xla

from omegaconf import OmegaConf
from torchprime.torch_xla_models.flex.modeling_qwen2 import (
    Qwen2ForCausalLM,
)
from torchprime.torch_xla_models.flex.attention import AttentionModule

import numpy as np
import time


def create_test_config():
    """Create a minimal config for testing"""
    config = {
        "vocab_size": 151936,
        "hidden_size": 1536,
        "num_hidden_layers": 2,  # Small for testing
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "head_dim": 128,
        "intermediate_size": 8960,
        "hidden_act": "silu",
        "max_position_embeddings": 131072,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-06,
        "use_sliding_window": False,
        "rope_theta": 1000000.0,
        "attention_dropout": 0.0,
        "mask_token_id": 151659,
        "pad_token_id": 151643,
        "attention_kernel": "flash_attention",
    }
    return OmegaConf.create(config)


def test_attention_module(device):
    """Test AttentionModule directly with and without segment_ids"""
    print(f"\n=== Testing AttentionModule on device: {device} ===")

    # Create config
    config = create_test_config()

    # Initialize attention module
    attn_module = AttentionModule(config).to(device)

    # Create test inputs
    batch_size = 2
    seq_len = 256
    num_heads = config.num_attention_heads
    head_dim = config.head_dim

    # Create query, key, value states
    hidden_states = torch.randn(batch_size, num_heads, seq_len // 2, head_dim, device=device)
    query_states = torch.cat([hidden_states, hidden_states.clone()], dim=2)
    key_states = query_states.clone()
    value_states = query_states.clone()

    # Create segment_ids
    segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    # Create two segments per sequence
    segment_ids[:, seq_len // 2 :] = 1

    # Test without segment_ids
    print("\nTesting attention WITHOUT segment_ids...")
    start_time = time.time()
    output_no_seg = attn_module(
        query_states, key_states, value_states, attention_mask=None, segment_ids=None
    )
    time_no_seg = time.time() - start_time
    print(f"Time without segment_ids: {time_no_seg:.4f}s")
    print(f"Output shape: {output_no_seg.shape}")

    # Test with segment_ids
    print("\nTesting attention WITH segment_ids...")
    start_time = time.time()
    output_with_seg = attn_module(
        query_states,
        key_states,
        value_states,
        attention_mask=None,
        segment_ids=segment_ids,
    )
    time_with_seg = time.time() - start_time
    print(f"Time with segment_ids: {time_with_seg:.4f}s")
    print(f"Output shape: {output_with_seg.shape}")
    print(f"Diff: {torch.norm(output_with_seg - output_no_seg).item():.4f}")
    print(f"Diff first half: {torch.norm(output_with_seg[:, :, :seq_len // 2] - output_no_seg[:, :, :seq_len // 2]).item():.4f}")
    # print(f"Output with segment_ids: {output_with_seg.detach().cpu()}")
    # print(f"Output without segment_ids: {output_no_seg.detach().cpu()}")


def main():
    device = torch_xla.device()
    test_attention_module(device)


if __name__ == "__main__":
    main()

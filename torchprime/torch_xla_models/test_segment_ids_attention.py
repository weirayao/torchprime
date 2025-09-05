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
    hidden_states = torch.randn(
        batch_size, num_heads, seq_len // 2, head_dim, device=device
    )
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
    print(
        f"Diff first half: {torch.norm(output_with_seg[:, :, :seq_len // 2, :] - output_no_seg[:, :, :seq_len // 2, :]).item():.4f}"
    )
    # print(f"Output with segment_ids: {output_with_seg.detach().cpu()}")
    # print(f"Output without segment_ids: {output_no_seg.detach().cpu()}")


def test_model_forward(device):
    """Test Qwen2ForCausalLM with and without segment_ids"""
    print(f"\n=== Testing Qwen2ForCausalLM on device: {device} ===")

    # Load model config
    config_path = "torchprime/torch_xla_models/configs/model/flex-qwen2-1b.yaml"
    config = OmegaConf.load(config_path)
    print(f"Loaded config from {config_path}")

    # Initialize model
    from torchprime.torch_xla_models.model_utils import initialize_model_class

    model = initialize_model_class(config, load_from_hf=True)
    model = model.to(device)
    model.train()
    print("Model loaded and moved to device")

    # Create dummy inputs
    batch_size = 2
    seq_len = 256

    # Create input_ids like [[1,2,3,4,5,6],[1,2,3,1,2,3]]
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # Create segment_ids where first half is segment 0, second half is segment 1
    segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    segment_ids[:, seq_len // 2 :] = 1

    print(f"\nInput IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids}")
    print(f"Segment IDs: {segment_ids}")

    # Test without segment_ids
    print("\n=== Testing model WITHOUT segment_ids ===")
    logits_no_seg, _ = model(input_ids=input_ids)
    logits_no_seg = logits_no_seg.detach().cpu()
    print(f"Logits shape: {logits_no_seg.shape}")

    # Test with segment_ids
    print("\n=== Testing model WITH segment_ids ===")
    logits_with_seg, _ = model(input_ids=input_ids, segment_ids=segment_ids)
    logits_with_seg = logits_with_seg.detach().cpu()
    print(f"Logits shape: {logits_with_seg.shape}")

    # Compare norms of logits for each row
    print("\n=== Comparing logits norms ===")
    for i in range(batch_size):

        # Also compute overall difference in logits
        diff = torch.norm(logits_with_seg[i] - logits_no_seg[i])
        print(f"  Logits difference (L2 norm): {diff.item():.6f}")

    # Also compare overall logits
    total_diff = torch.norm(logits_with_seg - logits_no_seg)
    print(f"\nTotal logits difference (L2 norm): {total_diff.item():.6f}")


def main():
    device = torch_xla.device()
    test_attention_module(device)
    test_model_forward(device)


if __name__ == "__main__":
    main()

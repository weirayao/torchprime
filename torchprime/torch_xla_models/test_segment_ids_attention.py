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
        "num_key_value_heads": 2,
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
    num_query_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = config.head_dim

    # Create query, key, value states
    query_states = torch.randn(
        batch_size, num_query_heads, seq_len // 2, head_dim, device=device
    )
    key_states = torch.randn(
        batch_size, num_key_value_heads, seq_len // 2, head_dim, device=device
    )
    query_states = torch.cat([query_states, query_states.clone()], dim=2)
    key_states = torch.cat([key_states, key_states.clone()], dim=2)
    value_states = key_states.clone()

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

    print("Now test with default attention")
    config.attention_kernel = "default"
    attn_module_default = AttentionModule(config).to(device)
    start_time = time.time()
    output_no_seg_default = attn_module_default(
        query_states,
        key_states,
        value_states,
        attention_mask=None,
        segment_ids=None,
    )
    time_no_seg_default = time.time() - start_time
    print(f"Default attention time without segment_ids: {time_no_seg_default:.4f}s")
    print(f"Output shape: {output_no_seg_default.shape}")
    start_time = time.time()
    output_with_seg_default = attn_module_default(
        query_states,
        key_states,
        value_states,
        attention_mask=None,
        segment_ids=segment_ids,
    )
    time_with_seg_default = time.time() - start_time
    print(f"Default attention time with segment_ids: {time_with_seg_default:.4f}s")
    print(f"Output shape: {output_with_seg_default.shape}")

    print(f"Diff between with segment_ids and without segment_ids (flash): {torch.norm(output_with_seg - output_no_seg).item():.4f}")
    print(f"Diff between with segment_ids and without segment_ids (eager): {torch.norm(output_with_seg_default - output_no_seg_default).item():.4f}")
    print(f"Diff between eager attention and flash attention with segment_ids: {torch.norm(output_with_seg_default - output_with_seg).item():.4f}")
    print(f"Diff between eager attention and flash attention without segment_ids: {torch.norm(output_no_seg_default - output_no_seg).item():.4f}")


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
    model.eval()
    print("Model loaded and moved to device")

    # Create dummy inputs
    batch_size = 1
    seq_len = 256

    # Create input_ids like [[1,2,3,4,5,6],[1,2,3,1,2,3]]
    input_ids_a = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    input_ids_b = input_ids_a[:, :seq_len // 2]

    # Create segment_ids where first half is segment 0, second half is segment 1
    segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    segment_ids[:, seq_len // 2 :] = 1

    print(f"\nInput IDs shape: {input_ids_a.shape}")
    print(f"Input IDs: {input_ids_a}")
    print(f"Segment IDs: {segment_ids}")

    # Test without segment_ids
    print("\n=== Testing model WITHOUT segment_ids ===")
    logits_no_seg, _ = model(input_ids=input_ids_b)
    logits_no_seg = logits_no_seg.detach().cpu()
    print(f"Logits shape: {logits_no_seg.shape}")

    # Test with segment_ids
    print("\n=== Testing model WITH segment_ids ===")
    logits_with_seg, _ = model(input_ids=input_ids_a, segment_ids=segment_ids)
    logits_with_seg = logits_with_seg.detach().cpu()
    logits_with_seg = logits_with_seg[:, :seq_len // 2, :]
    print(f"Logits shape: {logits_with_seg.shape}")


    print("Now test with default attention")
    config.attention_kernel = "default"
    model_default = initialize_model_class(config, load_from_hf=True)
    model_default = model_default.to(device)
    model_default.eval()
    print("Model loaded and moved to device")
    print("\n=== Testing model WITH segment_ids ===")
    logits_with_seg_default, _ = model_default(input_ids=input_ids_a, segment_ids=segment_ids)
    logits_with_seg_default = logits_with_seg_default.detach().cpu()
    logits_with_seg_default = logits_with_seg_default[:, :seq_len // 2, :]
    print(f"Logits shape: {logits_with_seg_default.shape}")

    print("\n=== Testing model WITHOUT segment_ids ===")
    logits_no_seg_default, _ = model_default(input_ids=input_ids_b)
    logits_no_seg_default = logits_no_seg_default.detach().cpu()
    print(f"Logits shape: {logits_no_seg_default.shape}")

    print("\n=== Comparing logits norms ===")
    print(f"Diff between eager attention and attention module: {torch.norm(logits_with_seg_default - logits_with_seg).item():.4f}")
    print(f"Diff between eager attention and attention module without segment_ids: {torch.norm(logits_no_seg_default - logits_no_seg).item():.4f}")

    print(f"Diff between segment_ids and no segment_ids (eager): {torch.norm(logits_with_seg_default - logits_no_seg_default).item():.4f}")
    print(f"Diff between segment_ids and no segment_ids (flash): {torch.norm(logits_with_seg - logits_no_seg).item():.4f}")


def main():
    device = torch_xla.device()
    test_attention_module(device)
    test_model_forward(device)


if __name__ == "__main__":
    main()

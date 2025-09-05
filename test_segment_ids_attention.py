#!/usr/bin/env python3
"""Unit test to compare flash attention with and without segment_ids on TPU v5p-8"""

import os
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.spmd as xs
from torch_xla.experimental.custom_kernel import FlashAttention

from omegaconf import OmegaConf
from torchprime.torch_xla_models.flex.modeling_qwen2 import (
    Qwen2ForCausalLM,
    EOS_TOKEN_ID,
)
from torchprime.torch_xla_models.flex.attention import AttentionModule

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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


def create_input_with_segments(batch_size=4, seq_len=512, vocab_size=151936):
    """Create input_ids with multiple segments separated by EOS tokens"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Insert EOS tokens at random positions to create segments
    for i in range(batch_size):
        # Create 2-4 segments per sequence
        num_segments = torch.randint(2, 5, (1,)).item()
        segment_positions = torch.randperm(seq_len - 1)[: num_segments - 1].sort()[0]

        for pos in segment_positions:
            input_ids[i, pos] = EOS_TOKEN_ID

    return input_ids


def compute_attention_scores(query_states, key_states, head_dim):
    """Compute raw attention scores (before softmax)"""
    # query_states, key_states: [batch_size, num_heads, seq_len, head_dim]
    scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / (head_dim**0.5)
    return scores


def visualize_attention_heatmap(
    attn_scores, segment_ids=None, title="Attention Heatmap", save_path=None
):
    """Visualize attention scores as a heatmap"""
    # Take the mean across heads and batch for visualization
    # attn_scores: [batch_size, num_heads, seq_len, seq_len]
    avg_scores = attn_scores.mean(dim=[0, 1]).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.imshow(avg_scores, cmap="hot", interpolation="nearest", aspect="auto")
    plt.colorbar(im, ax=ax, label="Attention Score")

    # Add segment boundaries if provided
    if segment_ids is not None:
        # segment_ids: [batch_size, seq_len]
        seg_ids = segment_ids[0].cpu().numpy()  # Use first batch item

        # Find segment boundaries
        boundaries = [0]
        for i in range(1, len(seg_ids)):
            if seg_ids[i] != seg_ids[i - 1]:
                boundaries.append(i)
        boundaries.append(len(seg_ids))

        # Draw rectangles for each segment block
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            # Draw rectangle
            rect = patches.Rectangle(
                (start - 0.5, start - 0.5),
                end - start,
                end - start,
                linewidth=2,
                edgecolor="cyan",
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)

            # Add segment label
            ax.text(
                start + (end - start) / 2,
                -2,
                f"Seg {seg_ids[start]}",
                ha="center",
                va="top",
                color="cyan",
                fontweight="bold",
            )
            ax.text(
                -2,
                start + (end - start) / 2,
                f"Seg {seg_ids[start]}",
                ha="right",
                va="center",
                color="cyan",
                fontweight="bold",
                rotation=90,
            )

    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(title)

    # Set ticks
    if avg_scores.shape[0] <= 64:
        ax.set_xticks(range(0, avg_scores.shape[0], 8))
        ax.set_yticks(range(0, avg_scores.shape[0], 8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved attention heatmap to {save_path}")
    else:
        plt.show()

    plt.close()


def test_attention_module_with_visualization(device):
    """Test AttentionModule directly with and without segment_ids, including visualization"""
    print(f"\n=== Testing AttentionModule with Visualization on device: {device} ===")

    # Create config - use default attention for visualization (not flash attention)
    config = create_test_config()

    # Initialize attention modules
    attn_module = AttentionModule(config).to(device)

    # Create test inputs - smaller for visualization
    batch_size = 1
    seq_len = 64  # Smaller sequence for clearer visualization
    num_heads = config.num_attention_heads
    head_dim = config.head_dim

    # Create query, key, value states
    query_states = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    # Create segment_ids with 3 segments
    segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    segment_ids[:, seq_len // 3 : 2 * seq_len // 3] = 1
    segment_ids[:, 2 * seq_len // 3 :] = 2

    # Compute attention scores for visualization
    print("\nComputing attention scores for visualization...")
    attn_scores_no_seg = compute_attention_scores(query_states, key_states, head_dim)

    # For segment masking visualization, we need to manually apply the mask
    # Create segment mask
    q_seg = segment_ids.unsqueeze(2)  # [batch, seq_len, 1]
    kv_seg = segment_ids.unsqueeze(1)  # [batch, 1, seq_len]
    segment_mask = (q_seg == kv_seg).float()  # [batch, seq_len, seq_len]
    segment_mask = segment_mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]

    # Apply segment mask to attention scores
    attn_scores_with_seg = attn_scores_no_seg.clone()
    attn_scores_with_seg = attn_scores_with_seg * segment_mask + (1 - segment_mask) * (
        -1e9
    )

    # Apply softmax for visualization
    attn_weights_no_seg = torch.softmax(attn_scores_no_seg, dim=-1)
    attn_weights_with_seg = torch.softmax(attn_scores_with_seg, dim=-1)

    # Visualize attention patterns
    print("\nGenerating attention heatmaps...")
    visualize_attention_heatmap(
        attn_weights_no_seg,
        segment_ids=None,
        title="Attention Without Segment IDs",
        save_path="attention_no_segments.png",
    )

    visualize_attention_heatmap(
        attn_weights_with_seg,
        segment_ids=segment_ids,
        title="Attention With Segment IDs",
        save_path="attention_with_segments.png",
    )

    # Test with flash attention
    print("\nTesting attention WITHOUT segment_ids...")
    start_time = time.time()
    output_no_seg = attn_module(
        query_states, key_states, value_states, attention_mask=None, segment_ids=None
    )
    xm.mark_step()
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
    xm.mark_step()
    time_with_seg = time.time() - start_time
    print(f"Time with segment_ids: {time_with_seg:.4f}s")
    print(f"Output shape: {output_with_seg.shape}")

    # Compare outputs
    diff = torch.abs(output_no_seg - output_with_seg)
    print(f"\nMax difference between outputs: {diff.max().item():.6f}")
    print(f"Mean difference between outputs: {diff.mean().item():.6f}")

    # The outputs should be different because segment_ids restricts attention
    print(
        f"\nOutputs are {'DIFFERENT' if diff.max().item() > 1e-5 else 'SIMILAR'} (as expected with segment masking)"
    )


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
    query_states = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

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
    xm.mark_step()
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
    xm.mark_step()
    time_with_seg = time.time() - start_time
    print(f"Time with segment_ids: {time_with_seg:.4f}s")
    print(f"Output shape: {output_with_seg.shape}")

    # Compare outputs
    diff = torch.abs(output_no_seg - output_with_seg)
    print(f"\nMax difference between outputs: {diff.max().item():.6f}")
    print(f"Mean difference between outputs: {diff.mean().item():.6f}")

    # The outputs should be different because segment_ids restricts attention
    print(
        f"\nOutputs are {'DIFFERENT' if diff.max().item() > 1e-5 else 'SIMILAR'} (as expected with segment masking)"
    )


def test_model_forward(device):
    """Test full model forward pass with and without segment_ids"""
    print(f"\n=== Testing Qwen2ForCausalLM on device: {device} ===")

    # Create config and model
    config = create_test_config()
    model = Qwen2ForCausalLM(config).to(device)
    model.train()

    # Create test inputs
    batch_size = 2
    seq_len = 256
    input_ids = create_input_with_segments(batch_size, seq_len, config.vocab_size).to(
        device
    )

    print(f"\nInput shape: {input_ids.shape}")

    # Count segments in each sequence
    for i in range(batch_size):
        eos_count = (input_ids[i] == EOS_TOKEN_ID).sum().item()
        print(f"Sequence {i}: {eos_count + 1} segments (EOS tokens: {eos_count})")

    # Test without explicit segment_ids (will be created automatically in pretrain mode)
    print("\nTesting model forward in pretrain mode (auto-generated segment_ids)...")
    start_time = time.time()
    logits_auto, loss_auto = model(input_ids, training_mode="pretrain")
    xm.mark_step()
    time_auto = time.time() - start_time
    print(f"Time with auto segment_ids: {time_auto:.4f}s")
    print(f"Logits shape: {logits_auto.shape}")
    print(f"Loss: {loss_auto.item() if loss_auto is not None else 'None'}")

    # Test with explicit segment_ids = None (no segmentation)
    print("\nTesting model forward with explicit segment_ids=None...")
    # Create a dummy segment_ids of all zeros (single segment)
    dummy_segment_ids = torch.zeros_like(input_ids, dtype=torch.long)
    start_time = time.time()
    logits_none, loss_none = model(
        input_ids, segment_ids=dummy_segment_ids, training_mode="pretrain"
    )
    xm.mark_step()
    time_none = time.time() - start_time
    print(f"Time with no segmentation: {time_none:.4f}s")
    print(f"Logits shape: {logits_none.shape}")
    print(f"Loss: {loss_none.item() if loss_none is not None else 'None'}")

    # Compare losses
    if loss_auto is not None and loss_none is not None:
        loss_diff = abs(loss_auto.item() - loss_none.item())
        print(f"\nLoss difference: {loss_diff:.6f}")
        print(f"Losses are {'DIFFERENT' if loss_diff > 1e-4 else 'SIMILAR'}")


def main():
    device = torch_xla.device()
    test_attention_module(device)
    test_attention_module_with_visualization(device)
    test_model_forward(device)


if __name__ == "__main__":
    main()

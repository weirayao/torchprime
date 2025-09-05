#!/usr/bin/env python3
"""Standalone script to visualize attention patterns with and without segment IDs"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def create_attention_pattern(seq_len, segment_ids=None):
    """Create a simple attention pattern for demonstration"""
    # Create base attention scores
    attention = torch.ones(seq_len, seq_len) * 0.5

    # Add some structure - stronger attention to nearby tokens
    for i in range(seq_len):
        for j in range(seq_len):
            distance = abs(i - j)
            attention[i, j] = 1.0 / (1.0 + distance * 0.1)

    # Apply segment masking if provided
    if segment_ids is not None:
        segment_mask = segment_ids.unsqueeze(1) == segment_ids.unsqueeze(0)
        attention = attention * segment_mask.float()

    # Apply softmax
    attention = torch.softmax(attention, dim=-1)
    return attention


def visualize_attention_comparison():
    """Create side-by-side comparison of attention with and without segments"""
    seq_len = 48

    # Create segment IDs: 3 segments of different sizes
    segment_ids = torch.zeros(seq_len, dtype=torch.long)
    segment_ids[16:32] = 1  # Segment 1
    segment_ids[32:] = 2  # Segment 2

    # Create attention patterns
    attn_no_seg = create_attention_pattern(seq_len, None)
    attn_with_seg = create_attention_pattern(seq_len, segment_ids)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot attention without segments
    im1 = ax1.imshow(
        attn_no_seg.numpy(), cmap="hot", interpolation="nearest", aspect="auto"
    )
    ax1.set_title("Attention Without Segment IDs", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Key Position")
    ax1.set_ylabel("Query Position")
    plt.colorbar(im1, ax=ax1, label="Attention Weight")

    # Plot attention with segments
    im2 = ax2.imshow(
        attn_with_seg.numpy(), cmap="hot", interpolation="nearest", aspect="auto"
    )
    ax2.set_title("Attention With Segment IDs", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Key Position")
    ax2.set_ylabel("Query Position")
    plt.colorbar(im2, ax=ax2, label="Attention Weight")

    # Add segment boundaries to the second plot
    boundaries = [0, 16, 32, seq_len]
    colors = ["red", "green", "blue"]
    labels = ["Segment 0", "Segment 1", "Segment 2"]

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        # Draw rectangle
        rect = patches.Rectangle(
            (start - 0.5, start - 0.5),
            end - start,
            end - start,
            linewidth=3,
            edgecolor=colors[i],
            facecolor="none",
            linestyle="-",
            alpha=0.8,
        )
        ax2.add_patch(rect)

        # Add labels
        ax2.text(
            start + (end - start) / 2,
            -2,
            labels[i],
            ha="center",
            va="top",
            color=colors[i],
            fontweight="bold",
            fontsize=10,
        )
        ax2.text(
            -2,
            start + (end - start) / 2,
            labels[i],
            ha="right",
            va="center",
            color=colors[i],
            fontweight="bold",
            fontsize=10,
            rotation=90,
        )

    # Set ticks
    for ax in [ax1, ax2]:
        ax.set_xticks(range(0, seq_len, 8))
        ax.set_yticks(range(0, seq_len, 8))
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

    plt.suptitle(
        "Attention Pattern Comparison: Effect of Segment IDs",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save the figure
    plt.savefig("attention_comparison.png", dpi=200, bbox_inches="tight")
    print("Saved attention comparison to attention_comparison.png")

    # Create a detailed view of the segment boundaries
    fig2, ax = plt.subplots(figsize=(10, 10))

    # Show attention with segments in more detail
    im = ax.imshow(
        attn_with_seg.numpy(), cmap="hot", interpolation="nearest", aspect="auto"
    )
    plt.colorbar(im, ax=ax, label="Attention Weight")

    # Add text annotations for key observations
    ax.text(
        8,
        40,
        "Cross-segment\nattention blocked\n(dark regions)",
        ha="center",
        va="center",
        color="cyan",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
    )

    ax.text(
        8,
        8,
        "Within-segment\nattention allowed\n(bright blocks)",
        ha="center",
        va="center",
        color="yellow",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
    )

    # Draw segment boundaries
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
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

    ax.set_title(
        "Detailed View: Attention With Segment IDs", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("attention_segments_detailed.png", dpi=200, bbox_inches="tight")
    print("Saved detailed view to attention_segments_detailed.png")

    plt.show()


if __name__ == "__main__":
    print("Generating attention pattern visualizations...")
    visualize_attention_comparison()
    print("\nVisualization complete!")
    print("\nKey observations:")
    print("1. Without segment IDs: All tokens can attend to all other tokens")
    print("2. With segment IDs: Attention is restricted to within-segment only")
    print("3. The diagonal blocks show the allowed attention regions")
    print("4. Dark regions indicate blocked cross-segment attention")

"""
Example script demonstrating how to use Qwen3ForCausalLM for both pre-training and SFT.
"""

import torch
from omegaconf import DictConfig
from modeling_qwen import Qwen3ForCausalLM

def create_sample_config():
    """Create a sample configuration for testing."""
    config = DictConfig({
        "vocab_size": 32000,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "hidden_act": "silu",
        "max_position_embeddings": 2048,
        "rope_theta": 10000.0,
        "use_sliding_window": False,
        "attention_bias": False,
        "rms_norm_eps": 1e-6,
        "initializer_range": 0.02,
        "mask_token_id": 32000,  # Using vocab_size as mask token
    })
    return config

def example_pretrain_mode():
    """Example of using the model in pre-training mode."""
    print("=== Pre-training Mode Example ===")
    
    config = create_sample_config()
    model = Qwen3ForCausalLM(config)
    model.train()
    
    # Sample input for pre-training
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size - 1, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # In pre-training mode, all tokens can be masked
    # No need to provide src_mask, it will be automatically set to all False
    logits, loss = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        training_mode="pretrain"  # Default mode
    )
    
    print(f"Pre-training - Input shape: {input_ids.shape}")
    print(f"Pre-training - Logits shape: {logits.shape}")
    print(f"Pre-training - Loss: {loss.item():.4f}")
    print()

def example_sft_mode():
    """Example of using the model in SFT mode."""
    print("=== SFT Mode Example ===")
    
    config = create_sample_config()
    model = Qwen3ForCausalLM(config)
    model.train()
    
    # Sample input for SFT
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size - 1, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # For SFT, we need to specify which tokens are instruction vs response
    # instruction_lengths: length of instruction/context for each sequence
    instruction_lengths = torch.tensor([4, 6])  # First seq: 4 instruction tokens, Second seq: 6 instruction tokens
    
    # Create src_mask where True = instruction tokens (should not be masked)
    src_mask = model.create_sft_src_mask(input_ids, instruction_lengths)
    
    print(f"SFT - Input shape: {input_ids.shape}")
    print(f"SFT - Instruction lengths: {instruction_lengths}")
    print(f"SFT - Source mask shape: {src_mask.shape}")
    print(f"SFT - Source mask:\n{src_mask}")
    
    # In SFT mode, only response tokens (where src_mask is False) will be masked
    logits, loss = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        src_mask=src_mask,
        training_mode="sft"
    )
    
    print(f"SFT - Logits shape: {logits.shape}")
    print(f"SFT - Loss: {loss.item():.4f}")
    print()

def example_sft_with_custom_mask():
    """Example of using the model in SFT mode with a custom src_mask."""
    print("=== SFT Mode with Custom Mask Example ===")
    
    config = create_sample_config()
    model = Qwen3ForCausalLM(config)
    model.train()
    
    # Sample input for SFT
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size - 1, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Create a custom src_mask manually
    # True = instruction/context tokens (should not be masked)
    # False = response tokens (should be masked for training)
    src_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    
    # Example: First sequence has instruction tokens at positions 0-3
    src_mask[0, :4] = True
    
    # Example: Second sequence has instruction tokens at positions 0-5
    src_mask[1, :6] = True
    
    print(f"SFT Custom - Input shape: {input_ids.shape}")
    print(f"SFT Custom - Source mask:\n{src_mask}")
    
    # In SFT mode, only response tokens (where src_mask is False) will be masked
    logits, loss = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        src_mask=src_mask,
        training_mode="sft"
    )
    
    print(f"SFT Custom - Logits shape: {logits.shape}")
    print(f"SFT Custom - Loss: {loss.item():.4f}")
    print()

if __name__ == "__main__":
    # Run examples
    example_pretrain_mode()
    example_sft_mode()
    example_sft_with_custom_mask()
    
    print("All examples completed successfully!") 
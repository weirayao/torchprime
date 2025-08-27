#!/usr/bin/env python3
"""
Example script showing how to use the custom Qwen3 model implementation
with weights loaded via initialize_model_class function on TPU.
"""

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from omegaconf import DictConfig
from transformers import AutoTokenizer

# Import the initialize_model_class function from train.py
from torchprime.torchprime.torch_xla_models.train_archive import initialize_model_class, set_default_dtype

# Initialize XLA runtime for TPU
xr.use_spmd()


def create_qwen3_config():
    """Create a configuration for Qwen3-8B model based on the YAML config."""
    config = DictConfig({
        # Model architecture configuration from qwen-3-8b.yaml
        "model_class": "qwen.Qwen3ForCausalLM",
        "attention_bias": False,
        "attention_dropout": False,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "pad_token_id": 151643,
        "tokenizer_name": "Qwen/Qwen3-8B",
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 12288,
        "max_position_embeddings": 40960,
        "max_window_layers": 36,
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "rope_theta": 1000000,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": 151936,
        "attention_kernel": "flash_attention"
    })
    return config


def main():
    """Main inference function for TPU."""
    print("Loading Qwen3-8B model on TPU...")
    
    # Get TPU device
    device = xm.xla_device()
    print(f"Using device: {device}")
    
    # Create model configuration
    model_config = create_qwen3_config()
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_name)
    
    # Initialize model with pretrained weights using initialize_model_class
    # Use the same pattern as in train.py for TPU
    with set_default_dtype(torch.bfloat16), torch_xla.device():
        model = initialize_model_class(model_config)
    
    # Set model to evaluation mode
    model.eval()
    
    print("Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Prepare the model input
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # Switches between thinking and non-thinking modes
    )
    # text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    print(f"Input text: {text}")
    
    # Tokenize input and move to TPU device
    model_inputs = tokenizer([text], return_tensors="pt")
    input_ids = model_inputs.input_ids.to(device)
    attention_mask = model_inputs.attention_mask.to(device)
    
    print(f"Input shape: {input_ids.shape}")
    
    # Generate text
    print("Generating response...")
    
    with torch.no_grad():
        # Simple greedy generation loop
        max_new_tokens = 512  # Reduced for faster inference
        generated_ids = input_ids.clone()
        
        for step in range(max_new_tokens):
            # if (step + 1) % 10 == 0:
                # print(f"Generating {step} tokens...")
            print(tokenizer.decode(generated_ids[0]))
            # Forward pass
            logits, _ = model(
                input_ids=generated_ids,
                attention_mask=attention_mask
            )
            
            # Get next token (greedy decoding)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            # Update attention mask
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device)
            ], dim=-1)
            
            # Check for end of sequence
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            
            # Print progress every 50 tokens
            if (step + 1) % 50 == 0:
                print(f"Generated {step + 1} tokens...")
    
    # Wait for TPU operations to complete
    xm.wait_device_ops()
    
    # Move results back to CPU for processing
    output_ids = generated_ids[0][len(input_ids[0]):].cpu().tolist()
    
    # Parse thinking content (if present)
    try:
        # Find the index of </think> token (151668)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    
    # Decode thinking and main content
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    
    print("\n" + "="*50)
    print("RESULTS:")
    print("="*50)
    print(f"Thinking content: {thinking_content}")
    print(f"Content: {content}")
    print("="*50)


if __name__ == "__main__":
    main() 
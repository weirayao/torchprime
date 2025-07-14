#!/usr/bin/env python3
"""
Debug script to test SFT data loading and identify issues with 0 loss.
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

# Add torchprime to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'torchprime'))

from transformers import AutoTokenizer
from torchprime.data.dataset import make_huggingface_sft_dataset
from torchprime.torch_xla_models.sft_data_collator import create_sft_dataset, SFTDataCollator
from torch.utils.data import DataLoader
import torch

def test_sft_data_loading():
    """Test SFT data loading and collation."""
    
    # Load tokenizer
    tokenizer_name = "Qwen/Qwen2.5-1.5B"  # Use the same tokenizer as your model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    print(f"Tokenizer loaded: {tokenizer_name}")
    print(f"Pad token ID: {tokenizer.pad_token_id}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    
    # Load raw dataset
    print("\nLoading Alpaca dataset...")
    raw_data = make_huggingface_sft_dataset(
        name="tatsu-lab/alpaca",
        config_name=None,
        split="train",
        cache_dir="/tmp/",
    )
    
    print(f"Raw dataset size: {len(raw_data)}")
    print(f"Raw dataset features: {list(raw_data.features.keys())}")
    
    # Show a few raw examples
    print("\nRaw examples:")
    for i in range(3):
        example = raw_data[i]
        print(f"Example {i}:")
        print(f"  Instruction: {example.get('instruction', 'N/A')[:100]}...")
        print(f"  Output: {example.get('output', 'N/A')[:100]}...")
        print(f"  System: {example.get('system', 'N/A')}")
        print()
    
    # Process dataset for SFT
    print("Processing dataset for SFT...")
    data = create_sft_dataset(
        dataset=raw_data,
        tokenizer=tokenizer,
        format="alpaca",
        include_system_prompt=True,
        instruction_response_separator="\n\n### Response:\n",
        block_size=8192,
    )
    
    print(f"Processed dataset size: {len(data)}")
    print(f"Processed dataset features: {list(data.features.keys())}")
    
    # Show processed examples
    print("\nProcessed examples:")
    for i in range(3):
        example = data[i]
        print(f"Example {i}:")
        print(f"  Input IDs length: {len(example['input_ids'])}")
        print(f"  Instruction length: {example['instruction_length']}")
        print(f"  Src mask sum: {sum(example['src_mask'])}")
        print(f"  Src mask ratio: {sum(example['src_mask']) / len(example['src_mask']):.3f}")
        
        # Decode some tokens
        instruction_tokens = example['input_ids'][:example['instruction_length']]
        response_tokens = example['input_ids'][example['instruction_length']:]
        
        instruction_text = tokenizer.decode(instruction_tokens, skip_special_tokens=True)
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        print(f"  Instruction text: {instruction_text[:100]}...")
        print(f"  Response text: {response_text[:100]}...")
        print()
    
    # Test data collator
    print("Testing data collator...")
    collator = SFTDataCollator(
        tokenizer=tokenizer,
        format="alpaca",
        include_system_prompt=True,
        instruction_response_separator="\n\n### Response:\n",
    )
    
    # Create a small batch
    batch_examples = [data[i] for i in range(4)]
    batch = collator(batch_examples)
    
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Src mask shape: {batch['src_mask'].shape}")
    print(f"Instruction lengths: {batch['instruction_lengths']}")
    
    # Validate src_mask
    total_instruction_tokens = batch['src_mask'].sum().item()
    total_tokens = batch['src_mask'].numel()
    instruction_ratio = total_instruction_tokens / total_tokens
    
    print(f"Total instruction tokens: {total_instruction_tokens}")
    print(f"Total tokens: {total_tokens}")
    print(f"Instruction ratio: {instruction_ratio:.3f}")
    
    if total_instruction_tokens == 0:
        print("ERROR: No instruction tokens found! This will cause 0 loss.")
        return False
    else:
        print("SUCCESS: Instruction tokens found. Data loading looks correct.")
        return True

def test_distributed_simulation():
    """Simulate distributed data loading to test splitting."""
    
    print("\n" + "="*50)
    print("Testing distributed data simulation...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    
    # Load and process dataset
    raw_data = make_huggingface_sft_dataset(
        name="tatsu-lab/alpaca",
        config_name=None,
        split="train",
        cache_dir="/tmp/",
    )
    
    data = create_sft_dataset(
        dataset=raw_data,
        tokenizer=tokenizer,
        format="alpaca",
        include_system_prompt=True,
        instruction_response_separator="\n\n### Response:\n",
        block_size=8192,
    )
    
    # Simulate 256-core setup
    num_devices = 256
    print(f"Simulating {num_devices} devices...")
    
    # Test manual splitting
    total_size = len(data)
    per_device_size = total_size // num_devices
    
    print(f"Total dataset size: {total_size}")
    print(f"Per device size: {per_device_size}")
    
    # Check a few devices
    for device_id in [0, 1, 255]:
        start_idx = device_id * per_device_size
        end_idx = start_idx + per_device_size if device_id < num_devices - 1 else total_size
        
        device_data = data.select(range(start_idx, end_idx))
        
        # Test collation on this device's data
        collator = SFTDataCollator(
            tokenizer=tokenizer,
            format="alpaca",
            include_system_prompt=True,
            instruction_response_separator="\n\n### Response:\n",
        )
        
        if len(device_data) > 0:
            batch_examples = [device_data[0]]  # Just test first example
            batch = collator(batch_examples)
            
            total_instruction_tokens = batch['src_mask'].sum().item()
            
            print(f"Device {device_id}: data size={len(device_data)}, "
                  f"instruction_tokens={total_instruction_tokens}")
            
            if total_instruction_tokens == 0:
                print(f"  ERROR: Device {device_id} has no instruction tokens!")
                return False
    
    print("SUCCESS: All simulated devices have instruction tokens.")
    return True

if __name__ == "__main__":
    print("SFT Data Loading Debug Script")
    print("="*50)
    
    success1 = test_sft_data_loading()
    success2 = test_distributed_simulation()
    
    if success1 and success2:
        print("\n" + "="*50)
        print("ALL TESTS PASSED! Data loading looks correct.")
        print("If you're still getting 0 loss, the issue might be elsewhere.")
    else:
        print("\n" + "="*50)
        print("TESTS FAILED! There are issues with data loading.")
        print("Check the errors above and fix the data processing pipeline.") 
#!/usr/bin/env python3
"""
Test script to verify the SFT data processing fix.
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

def test_sft_data_processing():
    """Test SFT data processing to verify the fix works."""
    
    print("="*60)
    print("Testing SFT Data Processing Fix")
    print("="*60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    print(f"Tokenizer loaded: Qwen/Qwen3-1.7B")
    
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
    for i in range(2):
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
    for i in range(2):
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
    
    # Test the simple collator on preprocessed data
    print("Testing simple collator on preprocessed data...")
    
    # Create a simple collator function similar to what we added to the trainer
    def simple_sft_collator(features):
        """Simple collator for preprocessed SFT data."""
        # Find the maximum length in the batch
        max_length = max(len(feature['input_ids']) for feature in features)
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_instruction_lengths = []
        batch_src_mask = []
        
        for feature in features:
            input_ids = feature['input_ids']
            instruction_length = feature['instruction_length']
            src_mask = feature['src_mask']
            
            # Pad input_ids
            padding_length = max_length - len(input_ids)
            padded_input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            
            # Pad attention mask
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            
            # Pad src_mask
            padded_src_mask = src_mask + [False] * padding_length
            
            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(attention_mask)
            batch_instruction_lengths.append(instruction_length)
            batch_src_mask.append(padded_src_mask)
        
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "instruction_lengths": torch.tensor(batch_instruction_lengths, dtype=torch.long),
            "src_mask": torch.tensor(batch_src_mask, dtype=torch.bool),
        }
    
    # Test the simple collator
    batch_examples = [data[i] for i in range(4)]
    batch = simple_sft_collator(batch_examples)
    
    print(f"Simple collator test:")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Attention mask shape: {batch['attention_mask'].shape}")
    print(f"  Src mask shape: {batch['src_mask'].shape}")
    print(f"  Instruction lengths: {batch['instruction_lengths']}")
    
    # Validate src_mask
    total_instruction_tokens = batch['src_mask'].sum().item()
    total_tokens = batch['src_mask'].numel()
    instruction_ratio = total_instruction_tokens / total_tokens
    
    print(f"  Total instruction tokens: {total_instruction_tokens}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Instruction ratio: {instruction_ratio:.3f}")
    
    if total_instruction_tokens == 0:
        print("ERROR: No instruction tokens found! This will cause 0 loss.")
        return False
    elif instruction_ratio > 0.9:
        print("WARNING: High instruction ratio detected. This might still cause small loss values.")
        return False
    else:
        print("SUCCESS: Simple collator works correctly with preprocessed data!")
        return True

def test_double_processing_error():
    """Test that the SFT data collator properly detects double processing."""
    
    print("\n" + "="*60)
    print("Testing Double Processing Detection")
    print("="*60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    
    # Create a mock preprocessed example (like what we see in the log)
    mock_preprocessed_example = {
        'input_ids': [271, 14374, 5949, 510],
        'instruction_length': 4,
        'src_mask': [True, True, True, True]
    }
    
    # Try to use the SFT data collator on preprocessed data
    print("Testing SFT data collator on preprocessed data...")
    collator = SFTDataCollator(
        tokenizer=tokenizer,
        format="alpaca",
        include_system_prompt=True,
        instruction_response_separator="\n\n### Response:\n",
    )
    
    try:
        # This should fail because the data is already preprocessed
        batch = collator([mock_preprocessed_example])
        print("ERROR: SFT data collator should have failed on preprocessed data!")
        return False
    except ValueError as e:
        if "already preprocessed" in str(e):
            print("SUCCESS: SFT data collator correctly detected preprocessed data!")
            print(f"Error message: {e}")
            return True
        else:
            print(f"ERROR: Unexpected error: {e}")
            return False

if __name__ == "__main__":
    success1 = test_sft_data_processing()
    success2 = test_double_processing_error()
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Data processing test: {'PASS' if success1 else 'FAIL'}")
    print(f"Double processing detection: {'PASS' if success2 else 'FAIL'}")
    
    if success1 and success2:
        print("\nSUCCESS: All tests passed! The SFT fix should work correctly.")
    else:
        print("\nFAILURE: Some tests failed. The fix may need additional work.") 
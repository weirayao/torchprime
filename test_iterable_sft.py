#!/usr/bin/env python3
"""
Test script to verify IterableDataset approach for SFT data distribution.
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

# Add torchprime to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'torchprime'))

from transformers import AutoTokenizer
from torchprime.data.dataset import make_huggingface_sft_dataset
from torchprime.torch_xla_models.sft_data_collator import create_sft_iterable_dataset, SFTDataCollator
from datasets.distributed import split_dataset_by_node

def test_iterable_sft():
    """Test the IterableDataset approach for SFT."""
    
    print("="*60)
    print("Testing IterableDataset SFT Approach")
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
    
    # Create IterableDataset
    print("\nCreating IterableDataset...")
    data = create_sft_iterable_dataset(
        dataset=raw_data,
        tokenizer=tokenizer,
        format="alpaca",
        include_system_prompt=True,
        instruction_response_separator="\n\n### Response:\n",
        block_size=8192,
        seed=42,
    )
    
    print(f"Created IterableDataset: {type(data)}")
    
    # Test data collator
    print("\nTesting data collator...")
    collator = SFTDataCollator(
        tokenizer=tokenizer,
        format="alpaca",
        include_system_prompt=True,
        instruction_response_separator="\n\n### Response:\n",
    )
    
    # Get a few examples
    examples = []
    for i, example in enumerate(data):
        if i >= 4:  # Get 4 examples
            break
        examples.append(example)
    
    print(f"Got {len(examples)} examples from IterableDataset")
    
    # Test collation
    batch = collator(examples)
    
    src_mask_sum = batch['src_mask'].sum().item()
    total_tokens = batch['src_mask'].numel()
    instruction_ratio = src_mask_sum / total_tokens
    
    print(f"Batch test:")
    print(f"  src_mask sum: {src_mask_sum}")
    print(f"  total tokens: {total_tokens}")
    print(f"  instruction ratio: {instruction_ratio:.3f}")
    print(f"  response ratio: {1-instruction_ratio:.3f}")
    
    # Test splitting simulation (simulate 256 processes)
    print(f"\nTesting dataset splitting simulation...")
    num_processes = 256
    
    # Simulate different process indices
    for process_idx in [0, 1, 255]:
        try:
            split_data = split_dataset_by_node(data, process_idx, num_processes)
            print(f"Process {process_idx}: Split successful")
            
            # Get one example from this split
            example = next(iter(split_data))
            print(f"Process {process_idx}: Got example with {len(example['input_ids'])} tokens")
            
        except Exception as e:
            print(f"Process {process_idx}: Split failed - {e}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    
    if instruction_ratio > 0.9:
        print("WARNING: High instruction ratio detected. This might still cause small loss values.")
    else:
        print("SUCCESS: IterableDataset approach looks good!")
    
    return True

if __name__ == "__main__":
    test_iterable_sft() 
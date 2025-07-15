"""
Data collator for SFT (Supervised Fine-Tuning) that handles instruction-response pairs.
"""

import torch
from typing import Dict, List, Optional, Union
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from datasets import Dataset, IterableDataset


class SFTDataCollator(DataCollatorMixin):
    """
    Data collator for SFT training that creates instruction-response pairs
    and generates src_masks to indicate which tokens are instruction vs response.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        format: str = "alpaca",
        include_system_prompt: bool = True,
        instruction_response_separator: str = "\n\n### Response:\n",
        custom_format: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.format = format
        self.include_system_prompt = include_system_prompt
        self.instruction_response_separator = instruction_response_separator
        self.custom_format = custom_format or {}
        
        # Get separator token IDs
        self.separator_token_ids = self.tokenizer.encode(
            instruction_response_separator, add_special_tokens=False
        )
        
    def _extract_instruction_response(self, example: Dict) -> tuple[str, str]:
        """Extract instruction and response from example based on format."""
        if self.format == "alpaca":
            instruction = example.get("instruction", "")
            response = example.get("output", "")
            
            # Include system prompt if available and enabled
            if self.include_system_prompt and "system" in example:
                system = example["system"]
                if system and system.strip():
                    instruction = f"{system}\n\n{instruction}"
                    
        elif self.format == "sharegpt":
            # ShareGPT format typically has conversations
            conversations = example.get("conversations", [])
            if len(conversations) >= 2:
                # Take the first human message as instruction, first assistant as response
                instruction = conversations[0].get("value", "")
                response = conversations[1].get("value", "")
            else:
                instruction = ""
                response = ""
                
        elif self.format == "custom":
            instruction = example.get(self.custom_format.get("instruction_field", "instruction"), "")
            response = example.get(self.custom_format.get("response_field", "response"), "")
            
            # Include system prompt if available and enabled
            if self.include_system_prompt and "system_field" in self.custom_format:
                system = example.get(self.custom_format["system_field"], "")
                if system and system.strip():
                    instruction = f"{system}\n\n{instruction}"
        else:
            raise ValueError(f"Unsupported format: {self.format}")
            
        return instruction, response
    
    def _create_instruction_response_sequence(
        self, 
        instruction: str, 
        response: str
    ) -> tuple[List[int], int]:
        """
        Create a tokenized sequence with instruction and response,
        and return the length of the instruction part.
        """
        # Tokenize instruction
        instruction_tokens = self.tokenizer.encode(
            instruction, add_special_tokens=False
        )
        
        # Add separator
        full_instruction_tokens = instruction_tokens + self.separator_token_ids
        
        # Tokenize response
        response_tokens = self.tokenizer.encode(
            response, add_special_tokens=False
        )
        
        # Combine instruction + separator + response
        full_sequence = full_instruction_tokens + response_tokens
        
        # Return sequence and instruction length (including separator)
        return full_sequence, len(full_instruction_tokens)
    
    def __call__(
        self, 
        features: List[Dict[str, Union[str, List[int]]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate features into a batch for SFT training.
        
        Args:
            features: List of examples, each containing instruction-response data
            
        Returns:
            Dictionary with:
            - input_ids: Tokenized sequences
            - attention_mask: Attention masks
            - instruction_lengths: Length of instruction part for each sequence
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_instruction_lengths = []
        
        for feature in features:
            # Debug the raw feature first
            if not hasattr(self, '_debug_extraction_count'):
                self._debug_extraction_count = 0
            
            if self._debug_extraction_count < 2:
                print(f"DEBUG Raw Feature {self._debug_extraction_count}:")
                print(f"  feature keys: {list(feature.keys())}")
                for key, value in feature.items():
                    if isinstance(value, str):
                        print(f"  {key}: '{value[:100]}...'")
                    else:
                        print(f"  {key}: {type(value)} - {value}")
            
            # Extract instruction and response
            instruction, response = self._extract_instruction_response(feature)
            
            if self._debug_extraction_count < 2:
                print(f"DEBUG Extraction {self._debug_extraction_count}:")
                print(f"  instruction: '{instruction[:100]}...'")
                print(f"  response: '{response[:100]}...'")
                print(f"  instruction len: {len(instruction)}")
                print(f"  response len: {len(response)}")
                self._debug_extraction_count += 1
            
            # Create sequence and get instruction length
            sequence, instruction_length = self._create_instruction_response_sequence(
                instruction, response
            )
            
            if self._debug_extraction_count <= 2:
                print(f"  sequence len: {len(sequence)}")
                print(f"  instruction_length: {instruction_length}")
                print(f"  sequence: {sequence[:10]}...")
            
            batch_input_ids.append(sequence)
            batch_instruction_lengths.append(instruction_length)
            
            # Create attention mask (all tokens are attended to)
            attention_mask = [1] * len(sequence)
            batch_attention_mask.append(attention_mask)
        
        # Pad sequences to the same length
        max_length = max(len(seq) for seq in batch_input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for input_ids, attention_mask in zip(batch_input_ids, batch_attention_mask):
            # Pad with tokenizer's pad token
            padding_length = max_length - len(input_ids)
            padded_input_ids.append(input_ids + [self.tokenizer.pad_token_id] * padding_length)
            padded_attention_mask.append(attention_mask + [0] * padding_length)
            
            # Ensure we have at least one valid token in each sequence
            if len(input_ids) == 0:
                padded_input_ids[-1][0] = self.tokenizer.eos_token_id or 1
                padded_attention_mask[-1][0] = 1
        
        # Create src_mask for SFT training
        src_mask = torch.zeros(len(padded_input_ids), max_length, dtype=torch.bool)
        for i, length in enumerate(batch_instruction_lengths):
            # Ensure length is valid and within bounds
            valid_length = min(length, max_length)
            if valid_length > 0:
                src_mask[i, :valid_length] = True
        
        # Ensure we don't have completely empty sequences
        # If any sequence has zero instruction length, give it at least 1 token
        for i, length in enumerate(batch_instruction_lengths):
            if length == 0:
                src_mask[i, 0] = True
                batch_instruction_lengths[i] = 1
        
        result = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "instruction_lengths": torch.tensor(batch_instruction_lengths, dtype=torch.long),
            "src_mask": src_mask,
        }
        
        # Add validation and debugging information
        if len(features) > 0:
            # Log some debugging info for the first batch
            total_instruction_tokens = src_mask.sum().item()
            total_tokens = src_mask.numel()
            instruction_ratio = total_instruction_tokens / total_tokens if total_tokens > 0 else 0
            
            # Only log on first call to avoid spam
            if not hasattr(self, '_logged_debug_info'):
                print(f"SFT DataCollator Debug: batch_size={len(features)}, "
                      f"max_length={max_length}, instruction_tokens={total_instruction_tokens}, "
                      f"total_tokens={total_tokens}, instruction_ratio={instruction_ratio:.3f}")
                self._logged_debug_info = True
            
            # Validate that we have instruction tokens
            if total_instruction_tokens == 0:
                raise ValueError("No instruction tokens found in batch - all src_mask values are False!")
        
        return result


def create_sft_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    format: str = "alpaca",
    include_system_prompt: bool = True,
    instruction_response_separator: str = "\n\n### Response:\n",
    custom_format: Optional[Dict[str, str]] = None,
    block_size: int = 8192,
) -> Dataset:
    """
    Create an SFT dataset from a raw dataset.
    
    Args:
        dataset: Raw dataset with instruction-response pairs
        tokenizer: Tokenizer for processing text
        format: Format of the data ("alpaca", "sharegpt", "custom")
        include_system_prompt: Whether to include system prompts
        instruction_response_separator: Separator between instruction and response
        custom_format: Custom format configuration
        block_size: Maximum sequence length
        
    Returns:
        Processed dataset ready for SFT training
    """
    def process_example(example):
        collator = SFTDataCollator(
            tokenizer=tokenizer,
            format=format,
            include_system_prompt=include_system_prompt,
            instruction_response_separator=instruction_response_separator,
            custom_format=custom_format,
        )
        
        # Extract instruction and response
        instruction, response = collator._extract_instruction_response(example)
        
        # Create sequence
        sequence, instruction_length = collator._create_instruction_response_sequence(
            instruction, response
        )
        
        # Truncate if too long
        if len(sequence) > block_size:
            sequence = sequence[:block_size]
            # Adjust instruction length if it was truncated
            instruction_length = min(instruction_length, len(sequence))
        
        # Create src_mask for this example
        src_mask = [True] * instruction_length + [False] * (len(sequence) - instruction_length)
        
        return {
            "input_ids": sequence,
            "instruction_length": instruction_length,
            "src_mask": src_mask,
        }
    
    return dataset.map(process_example, remove_columns=dataset.column_names)


def create_sft_iterable_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    format: str = "alpaca",
    include_system_prompt: bool = True,
    instruction_response_separator: str = "\n\n### Response:\n",
    custom_format: Optional[Dict[str, str]] = None,
    block_size: int = 8192,
    seed: int = 42,
) -> IterableDataset:
    """
    Create an SFT IterableDataset from a raw dataset.
    This enables proper data distribution across multiple processes.
    
    Args:
        dataset: Raw dataset with instruction-response pairs
        tokenizer: Tokenizer for processing text
        format: Format of the data ("alpaca", "sharegpt", "custom")
        include_system_prompt: Whether to include system prompts
        instruction_response_separator: Separator between instruction and response
        custom_format: Custom format configuration
        block_size: Maximum sequence length
        seed: Random seed for shuffling
        
    Returns:
        IterableDataset ready for SFT training with proper distribution
    """
    def process_example(example):
        collator = SFTDataCollator(
            tokenizer=tokenizer,
            format=format,
            include_system_prompt=include_system_prompt,
            instruction_response_separator=instruction_response_separator,
            custom_format=custom_format,
        )
        
        # Extract instruction and response
        instruction, response = collator._extract_instruction_response(example)
        
        # Create sequence
        sequence, instruction_length = collator._create_instruction_response_sequence(
            instruction, response
        )
        
        # Truncate if too long
        if len(sequence) > block_size:
            sequence = sequence[:block_size]
            # Adjust instruction length if it was truncated
            instruction_length = min(instruction_length, len(sequence))
        
        # Create src_mask for this example
        src_mask = [True] * instruction_length + [False] * (len(sequence) - instruction_length)
        
        return {
            "input_ids": sequence,
            "instruction_length": instruction_length,
            "src_mask": src_mask,
        }
    
    try:
        # Convert to IterableDataset and add shuffling
        iterable_dataset = dataset.to_iterable_dataset()
        iterable_dataset = iterable_dataset.map(process_example, remove_columns=dataset.column_names)
        iterable_dataset = iterable_dataset.shuffle(seed=seed, buffer_size=10000)
        
        # Create a wrapper class to ensure the custom flag persists
        class CustomIterableDatasetWrapper(IterableDataset):
            def __init__(self, dataset):
                super().__init__()
                self.dataset = dataset
                self._is_custom = True
            
            def __iter__(self):
                return iter(self.dataset)
            
            def __getattr__(self, name):
                return getattr(self.dataset, name)
        
        return CustomIterableDatasetWrapper(iterable_dataset)
    except Exception as e:
        # Fallback: create a simple IterableDataset manually
        class SimpleIterableDataset(IterableDataset):
            def __init__(self, dataset, process_fn, seed):
                self.dataset = dataset
                self.process_fn = process_fn
                self.seed = seed
                # Flag to indicate this is a custom dataset that doesn't need splitting
                self._is_custom = True
            
            def __iter__(self):
                # Simple shuffling by creating a list and shuffling it
                import random
                random.seed(self.seed)
                indices = list(range(len(self.dataset)))
                random.shuffle(indices)
                
                for idx in indices:
                    example = self.dataset[idx]
                    yield self.process_fn(example)
        
        return SimpleIterableDataset(dataset, process_example, seed) 
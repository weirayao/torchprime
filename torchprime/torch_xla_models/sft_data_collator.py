"""
Data collator for SFT (Supervised Fine-Tuning) that handles instruction-response pairs.
"""

import torch
from typing import Dict, List, Optional, Union
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from datasets import Dataset


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
        # Add logging for debugging (reduced frequency)
        import logging
        logger = logging.getLogger(__name__)
        
        # Minimal logging - only for first batch
        if hasattr(self, '_batch_count'):
            self._batch_count += 1
        else:
            self._batch_count = 0
            
        # Only log the first batch
        if self._batch_count == 1:
            logger.info(f"SFTDataCollator processing {len(features)} features")
            logger.info(f"First feature keys: {list(features[0].keys()) if features else 'No features'}")
        
        # Check if data is already pre-processed (has input_ids and src_mask)
        if features and 'input_ids' in features[0] and 'src_mask' in features[0]:
            if self._batch_count == 1:
                logger.info("Data is pre-processed, using existing tokenization")
            return self._collate_preprocessed_features(features)
        else:
            if self._batch_count == 1:
                logger.info("Data is raw text, processing with instruction/response extraction")
            return self._collate_raw_features(features)
    
    def _collate_preprocessed_features(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate features that are already pre-processed with input_ids and src_mask."""
        import logging
        logger = logging.getLogger(__name__)
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_src_masks = []
        batch_instruction_lengths = []
        
        for i, feature in enumerate(features):
            input_ids = feature['input_ids']
            src_mask = feature['src_mask']
            instruction_length = feature.get('instruction_length', sum(src_mask))
            
            # Only log for first batch, first feature
            if self._batch_count == 1 and i == 0:
                logger.info(f"Pre-processed feature {i}:")
                logger.info(f"  input_ids length: {len(input_ids)}")
                logger.info(f"  instruction_length: {instruction_length}")
                logger.info(f"  src_mask sum: {sum(src_mask)}")
            
            batch_input_ids.append(input_ids)
            batch_instruction_lengths.append(instruction_length)
            batch_src_masks.append(src_mask)
            
            # Create attention mask (all tokens are attended to)
            attention_mask = [1] * len(input_ids)
            batch_attention_mask.append(attention_mask)
        
        # Pad sequences to the same length
        max_length = max(len(seq) for seq in batch_input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_src_masks = []
        
        for input_ids, attention_mask, src_mask in zip(batch_input_ids, batch_attention_mask, batch_src_masks):
            # Pad with tokenizer's pad token
            padding_length = max_length - len(input_ids)
            padded_input_ids.append(input_ids + [self.tokenizer.pad_token_id] * padding_length)
            padded_attention_mask.append(attention_mask + [0] * padding_length)
            padded_src_masks.append(src_mask + [False] * padding_length)  # Pad src_mask with False
            
            # Ensure we have at least one valid token in each sequence
            if len(input_ids) == 0:
                padded_input_ids[-1][0] = self.tokenizer.eos_token_id or 1
                padded_attention_mask[-1][0] = 1
                padded_src_masks[-1][0] = True
        
        # Convert to tensors
        result = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "instruction_lengths": torch.tensor(batch_instruction_lengths, dtype=torch.long),
            "src_mask": torch.tensor(padded_src_masks, dtype=torch.bool),
        }
        
        return result
    
    def _collate_raw_features(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate features that contain raw text instruction/response pairs."""
        import logging
        logger = logging.getLogger(__name__)
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_instruction_lengths = []
        
        for i, feature in enumerate(features):
            # Extract instruction and response
            instruction, response = self._extract_instruction_response(feature)
            
            # Log only for first batch, first feature
            if self._batch_count == 1 and i == 0:
                logger.info(f"Raw feature {i}:")
                logger.info(f"  Extracted instruction: '{instruction[:50]}...'")
                logger.info(f"  Extracted response: '{response[:50]}...'")
            
            # Create sequence and get instruction length
            sequence, instruction_length = self._create_instruction_response_sequence(
                instruction, response
            )
            
            if self._batch_count == 1 and i == 0:
                logger.info(f"  Sequence length: {len(sequence)}")
                logger.info(f"  Instruction length: {instruction_length}")
            
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
        
        # Add validation to ensure src_mask is properly formed
        if self._batch_count == 1:
            logger.info(f"src_mask shape: {src_mask.shape}")
            logger.info(f"src_mask sum per sequence: {src_mask.sum(dim=1)}")
            logger.info(f"instruction_lengths: {batch_instruction_lengths}")
        
        result = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "instruction_lengths": torch.tensor(batch_instruction_lengths, dtype=torch.long),
            "src_mask": src_mask,
        }
        
        # Add validation to ensure all tensors have consistent shapes
        if self._batch_count == 1:
            logger.info(f"Final tensor shapes:")
            logger.info(f"  input_ids: {result['input_ids'].shape}")
            logger.info(f"  attention_mask: {result['attention_mask'].shape}")
            logger.info(f"  src_mask: {result['src_mask'].shape}")
            logger.info(f"  instruction_lengths: {result['instruction_lengths'].shape}")
            
            # Check for potential issues
            if result['input_ids'].shape != result['attention_mask'].shape:
                logger.error(f"Shape mismatch: input_ids {result['input_ids'].shape} vs attention_mask {result['attention_mask'].shape}")
            if result['input_ids'].shape != result['src_mask'].shape:
                logger.error(f"Shape mismatch: input_ids {result['input_ids'].shape} vs src_mask {result['src_mask'].shape}")
        
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
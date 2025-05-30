from datasets import Dataset, DatasetDict, load_dataset
from transformers.tokenization_utils import PreTrainedTokenizerBase


def make_huggingface_dataset(
  name: str,
  config_name: str,
  split: str,
  cache_dir: str,
  tokenizer: PreTrainedTokenizerBase,
  block_size: int,
) -> Dataset:
  # Downloading and loading a dataset from the hub.
  data = load_dataset(
    name,
    config_name,
    cache_dir=cache_dir,
  )
  assert isinstance(data, DatasetDict)
  data = data[split]

  column_names = list(data.features)
  data = data.map(
    lambda samples: tokenizer(samples["text"]),
    batched=True,
    remove_columns=column_names,
  )

  # Taken from run_clm.py. It's important to group texts evenly to avoid recompilations in TPU.
  def group_texts(examples):
    from itertools import chain

    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
      k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
      for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

  data = data.map(group_texts, batched=True)
  return data


def make_gcs_dataset(
  name: str,
  config_name: str,
  split: str,
  cache_dir: str,
  tokenizer: PreTrainedTokenizerBase,
  block_size: int,
) -> Dataset:
  """
  Load dataset from GCS bucket or use HuggingFace datasets with GCS paths.
  
  Args:
    name: Can be a GCS path (gs://bucket/path) or HuggingFace dataset name
    config_name: Dataset configuration name (can be None for GCS files)
    split: Dataset split (train/validation/test)
    cache_dir: Local cache directory
    tokenizer: Tokenizer to use for processing text
    block_size: Block size for grouping texts
    
  Returns:
    Processed Dataset ready for training
  """
  
  # Check if name is a GCS path
  if name.startswith("gs://"):
    # Load dataset directly from GCS
    # This supports various formats: parquet, csv, json, text files
    try:
      data = load_dataset(
        name,
        name=config_name,  # For datasets with multiple configs
        split=split,
        cache_dir=cache_dir,
      )
    except Exception:
      # Fallback: try loading without config_name
      data = load_dataset(
        name,
        split=split,
        cache_dir=cache_dir,
      )
  else:
    # Use standard HuggingFace dataset loading
    data = load_dataset(
      name,
      config_name,
      cache_dir=cache_dir,
    )
    if isinstance(data, DatasetDict):
      data = data[split]

  # Handle different data structures
  if isinstance(data, DatasetDict):
    data = data[split]
  
  # Determine text column name
  column_names = list(data.features)
  text_column = None
  for col in ["text", "content", "document", "sentence"]:
    if col in column_names:
      text_column = col
      break
  
  if text_column is None:
    raise ValueError(f"No text column found in dataset. Available columns: {column_names}")
  
  # Tokenize the dataset
  def tokenize_function(examples):
    return tokenizer(examples[text_column])
  
  data = data.map(
    tokenize_function,
    batched=True,
    remove_columns=column_names,
  )

  # Group texts into blocks
  def group_texts(examples):
    from itertools import chain

    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
      k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
      for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

  data = data.map(group_texts, batched=True)
  return data

import os
import random
from itertools import chain
from typing import Sequence
from glob import glob
from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset, interleave_datasets, Dataset, DatasetDict, IterableDataset, concatenate_datasets
from transformers.tokenization_utils import PreTrainedTokenizerBase

MOUNTED_GCS_DIR = os.environ.get("MOUNTED_GCS_DIR", None)
if MOUNTED_GCS_DIR is None:
  raise ValueError("MOUNTED_GCS_DIR is not set or GCS is not mounted.")


def group_texts(examples, block_size):
  """Group texts into blocks of specified size for training."""
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


DATASET_TYPES = {
    "fineweb_edu_dedup": (".parquet", "parquet"),  # 512.29 GiB - educational web, largest
    "dclm-baseline-1.0-shuffled": (".json", "json"), # 251.32 GiB - web crawl
    "RedPajama_math": (".json", "json"),           # 168.48 GiB - math dataset
    "stackv2_Python_shuffled": (".json", "json"),  # 186.7 GiB - Python code
    "open-web-math": (".json", "json"),            # 45.39 GiB - math problems
    "Redpajama-Arxiv": (".json", "json"),          # 26.37 GiB - academic papers
    "Falcon-refinedweb": (".json", "json"),        # 130.89 GiB - web text
    "the-stack-v2-train-smol": (".json", "json"),  # 135.9 GiB - code
    "cosmopedia_v2_parquet": (".parquet", "parquet"), # 113.96 GiB - synthetic textbook
    "c4_2023-14": (".json", "json"),               # 77.85 GiB - web crawl
    "RedPajama": (".json", "json"),                # 65.53 GiB - mixed content
    "Wikipedia_en": (".json", "json"),             # 19.53 GiB - encyclopedia
    "python_edu": (".jsonl", "json"),              # 13.88 GiB - code
    "Gutenberg": (".json", "json"),                # 10.68 GiB - literature  
    "DM_Mathematics": (".json", "json"),           # 5.69 GiB - math, smallest
}

def make_gcs_pretokenized_dataset(
  path: str,
  seed: int = 42,
) -> IterableDataset:
  """
  Search for all parquet files in the given path and load them into a dataset.
  Shuffle the files first.
  """
  random.seed(seed)
  data_files = glob(f"{path}/**/*.parquet", recursive=True)
  print(f"dataset path: {data_files}")
  random.shuffle(data_files)
  
  data = load_dataset(
    "parquet",
    data_files=data_files,
    streaming=True,
    split="train",
  )
  data = data.shuffle(seed=seed, buffer_size=32768)
  return data

def make_gcs_dataset(
  names: list[str],
  tokenizer: PreTrainedTokenizerBase,
  block_size: int,
  seed: int = 42,
  weights: Sequence[float] = None,
) -> IterableDataset:
  if any(name not in DATASET_TYPES for name in names):
    raise ValueError(f"Dataset {names} not found in {DATASET_TYPES}")
  
  def tokenize_fn(examples):
    texts = [example + tokenizer.eos_token for example in examples["text"]]
    return tokenizer(texts)

  data_mixture = []
  for name in names:
    extension, data_type = DATASET_TYPES[name]
    data_files = glob(f"{MOUNTED_GCS_DIR}/data/xgen_cleaned_data/{name}/*{extension}")
    print(f"Loading dataset {name}, data_files example: {data_files[0]}")
    data = load_dataset(
      data_type,
      data_files=data_files,
      streaming=True,
      split="train",
    )
    # Get columns from first example but convert dict_keys to list immediately
    columns = list(list(data.take(1))[0].keys())

    print(f"Shuffling dataset {name}")
    data = data.shuffle(seed=seed, buffer_size=32768)

    print(f"Pretokenizing dataset {name}")
    data = data.map(
      tokenize_fn,
      batched=True,
      remove_columns=columns,
    )

    print(f"Grouping dataset {name}")
    data = data.map(
      lambda examples: group_texts(examples, block_size),
      batched=True,
    )
    data_mixture.append(data)

  if len(data_mixture) == 1:
    return data_mixture[0]
  else:
    return concatenate_datasets(data_mixture)
    # return interleave_datasets(data_mixture, probabilities=weights, seed=seed)


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

  # Check if this is an SFT dataset (has instruction/output fields)
  if "instruction" in data.features and "output" in data.features:
    # For SFT datasets, create text field from instruction + response
    def create_sft_text(example):
      instruction = example.get("instruction", "")
      output = example.get("output", "")
      system = example.get("system", "")
      
      # Combine system + instruction + separator + output
      if system and system.strip():
        text = f"{system}\n\n{instruction}\n\n### Response:\n{output}"
      else:
        text = f"{instruction}\n\n### Response:\n{output}"
      
      return {"text": text}
    
    data = data.map(create_sft_text, remove_columns=list(data.features))
  elif "text" not in data.features:
    raise ValueError(f"Dataset {name} does not have 'text' field and is not a recognized SFT format")

  column_names = list(data.features)
  data = data.map(
    lambda samples: tokenizer(samples["text"]),
    batched=True,
    remove_columns=column_names,
  )

  # Taken from run_clm.py. It's important to group texts evenly to avoid recompilations in TPU.
  data = data.map(lambda examples: group_texts(examples, block_size), batched=True)
  return data


def make_huggingface_sft_dataset(
  name: str,
  config_name: str,
  split: str,
  cache_dir: str,
) -> Dataset:
  """
  Load a HuggingFace dataset for SFT training without tokenization.
  The SFT data collator will handle tokenization later.
  
  Args:
    name: Dataset name
    config_name: Dataset config name
    split: Dataset split
    cache_dir: Cache directory
    
  Returns:
    Raw dataset ready for SFT processing
  """
  # Downloading and loading a dataset from the hub.
  data = load_dataset(
    name,
    config_name,
    cache_dir=cache_dir,
  )
  assert isinstance(data, DatasetDict)
  data = data[split]
  
  return data


def make_huggingface_sft_iterable_dataset(
  name: str,
  config_name: str,
  split: str,
  cache_dir: str,
  seed: int = 42,
) -> IterableDataset:
  """
  Load a HuggingFace dataset as IterableDataset for SFT training.
  This is more memory efficient for large datasets and better for multi-process training.
  
  Args:
    name: Dataset name
    config_name: Dataset config name
    split: Dataset split
    cache_dir: Cache directory
    seed: Random seed for shuffling
    
  Returns:
    IterableDataset ready for SFT processing
  """
  # Downloading and loading a dataset from the hub as IterableDataset
  data = load_dataset(
    name,
    config_name,
    cache_dir=cache_dir,
    streaming=True,  # Creates IterableDataset directly
    split=split,
  )
  
  # Add shuffling for better distribution across processes
  data = data.shuffle(seed=seed, buffer_size=10000)
  
  return data


def make_mixed_huggingface_datasets(
  hf_datasets: list[dict],
  split: str,
  cache_dir: str,
  seed: int = 42,
) -> Dataset:
  """
  Load and mix multiple HuggingFace datasets for SFT training.
  
  Args:
    hf_datasets: List of dicts with 'name', 'config', and 'weight' keys
    split: Dataset split to load
    cache_dir: Cache directory for datasets
    seed: Random seed for shuffling
    
  Returns:
    Mixed dataset with all datasets concatenated
  """
  if not hf_datasets:
    raise ValueError("No datasets provided in hf_datasets")
  
  print(f"Loading {len(hf_datasets)} datasets for mixing...")
  
  datasets_list = []
  for i, dataset_config in enumerate(hf_datasets):
    name = dataset_config["name"]
    config = dataset_config.get("config")
    weight = dataset_config.get("weight", 1.0)
    
    print(f"Loading dataset {i+1}/{len(hf_datasets)}: {name} (config: {config}, weight: {weight})")
    
    # Load the dataset
    data = load_dataset(
      name,
      config,
      cache_dir=cache_dir,
    )
    assert isinstance(data, DatasetDict)
    data = data[split]
    
    # Shuffle the dataset
    data = data.shuffle(seed=seed + i, buffer_size=10000)
    
    datasets_list.append(data)
  
  # Concatenate all datasets
  print("Concatenating datasets...")
  mixed_dataset = concatenate_datasets(datasets_list)
  
  # Final shuffle of the mixed dataset
  print("Final shuffle of mixed dataset...")
  mixed_dataset = mixed_dataset.shuffle(seed=seed, buffer_size=10000)
  
  print(f"Mixed dataset created with {len(mixed_dataset)} total examples")
  return mixed_dataset

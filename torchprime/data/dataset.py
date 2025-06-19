import os
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
    print(f"tokenized text list length: {len(texts)}")
    return tokenizer(texts)

  data_mixture = []
  for name in names:
    extension, data_type = DATASET_TYPES[name]
    data_files = glob(f"{MOUNTED_GCS_DIR}/data/xgen_cleaned_data/{name}/*{extension}")
    print(f"total number of data files: {len(data_files)}")
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

    print(f"debug: Getting first example from dataset")
    first_example = list(data.take(1))[0]
    print(f"debug: First example keys: {list(first_example.keys())}")
    print(f"debug: First example text type: {type(first_example['text'])}")
    print(f"debug: First example text: {first_example['text'][:200]}...")
      
    data = data.map(
      tokenize_fn,
      batched=True,
      remove_columns=columns,
    )
    # Check tokens after tokenization
    print(f"debug: Checking tokens after tokenization...")
    try:
      tokenized_example = list(data.take(1))[0]
      print(f"debug: Tokenized example keys: {list(tokenized_example.keys())}")
      if 'input_ids' in tokenized_example:
        print(f"debug: input_ids length: {len(tokenized_example['input_ids'])}")
        print(f"debug: First 20 input_ids: {tokenized_example['input_ids'][:20]}")
      if 'attention_mask' in tokenized_example:
        print(f"debug: attention_mask length: {len(tokenized_example['attention_mask'])}")
        print(f"debug: First 20 attention_mask: {tokenized_example['attention_mask'][:20]}")
    except Exception as e:
      print(f"debug: Error checking tokenized data: {e}")

    print(f"Grouping dataset {name}")
    data = data.map(
      lambda examples: group_texts(examples, block_size),
      batched=True,
    )
    # Check tokens after grouping
    print(f"debug: Checking tokens after grouping...")
    try:
      grouped_example = list(data.take(1))[0]
      print(f"debug: Grouped example keys: {list(grouped_example.keys())}")
      if 'input_ids' in grouped_example:
        print(f"debug: Number of input_ids chunks: {len(grouped_example['input_ids'])}")
        print(f"debug: First chunk length: {len(grouped_example['input_ids'][0])}")
        print(f"debug: First chunk first 20 tokens: {grouped_example['input_ids'][0][:20]}")
      if 'labels' in grouped_example:
        print(f"debug: Number of labels chunks: {len(grouped_example['labels'])}")
        print(f"debug: First chunk length: {len(grouped_example['labels'][0])}")
    except Exception as e:
      print(f"debug: Error checking grouped data: {e}")
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

  column_names = list(data.features)
  data = data.map(
    lambda samples: tokenizer(samples["text"]),
    batched=True,
    remove_columns=column_names,
  )

  # Taken from run_clm.py. It's important to group texts evenly to avoid recompilations in TPU.
  data = data.map(lambda examples: group_texts(examples, block_size), batched=True)
  return data

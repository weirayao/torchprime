import os
from itertools import chain
from typing import Sequence
from glob import glob
from datasets import load_dataset, interleave_datasets, Dataset, DatasetDict, IterableDataset
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
  'DM_Mathematics': ('.json', 'json'),
  'Falcon-refinedweb': ('.json', 'json'),
  'Gutenberg': ('.json', 'json'),
  'RedPajama': ('.json', 'json'),
  'RedPajama_math': ('.json', 'json'),
  'Redpajama-Arxiv': ('.json', 'json'),
  'Wikipedia_en': ('.json', 'json'),
  'c4_2023-14': ('.json', 'json'),
  'cosmopedia_v2_parquet': ('.parquet', 'parquet'),
  'dclm-baseline-1.0-shuffled': ('.json', 'json'),
  'fineweb_edu_dedup': ('.parquet', 'parquet'),
  'open-web-math': ('.json', 'json'),
  'python_edu': ('.jsonl', 'json'),
  'stackv2_Python_shuffled': ('.json', 'json'),
  'the-stack-v2-train-smol': ('.json', 'json'),
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
    columns = list(data.take(1))[0].keys()

    print(f"Shuffling dataset {name}")
    data = data.shuffle(seed=seed, buffer_size=32768)

    print(f"Pretokenizing dataset {name}")
    data = data.map(
      lambda examples: tokenizer(examples["text"]),
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
    return interleave_datasets(data_mixture, probabilities=weights, seed=seed)


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

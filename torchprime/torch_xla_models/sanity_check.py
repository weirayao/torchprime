import importlib
import json
import logging
import math
import os
import logging
import sys
from contextlib import contextmanager
from collections import OrderedDict
from functools import partial
from pathlib import Path
from timeit import default_timer as timer

from dotenv import load_dotenv
load_dotenv()

import datasets
import hydra
import torch
import transformers
from omegaconf import DictConfig, OmegaConf
from transformers import (
  AutoTokenizer,
  Qwen2ForCausalLM,
)

HF_MODEL_CLASS_MAPPING = {
  "llama.LlamaForCausalLM": "LlamaForCausalLM",
  "flex.LlamaForCausalLM": "LlamaForCausalLM", 
  "flex.Qwen3ForCausalLM": "Qwen3ForCausalLM",
  "flex.Qwen2ForCausalLM": "Qwen2ForCausalLM",
  "mixtral.MixtralForCausalLM": "MixtralForCausalLM",
  "llama4.Llama4TextForCausalLM": "Llama4ForCausalLM",
  "qwen.Qwen3ForCausalLM": "Qwen3ForCausalLM",
}


logger = logging.getLogger(__name__)

def initialize_model_class(model_config, load_from_hf=True):
  """Import and initialize model_class specified by the config."""
  module_name, model_class_name = model_config.model_class.rsplit(".", 1)
  try:
    module = importlib.import_module(module_name)
  except ModuleNotFoundError as e:
    print(f"Error importing relative module: {e}")
    sys.exit(1)
  if hasattr(module, model_class_name):
    model_class = getattr(module, model_class_name)
  else:
    print(f"Error: Function '{model_class_name}' not found in module '{module_name}'")
    sys.exit(1)
  model = model_class(model_config)
  logger.info(f"model.state_dict().keys() before loading: {model.state_dict().keys()}")
  # Load pretrained weights from HuggingFace model
  if load_from_hf:
    hf_model = load_hf_model(model_config)
    logger.info("Loaded model from HuggingFace. Now loading state dict.")
    model.load_state_dict(hf_model.state_dict())
    del hf_model
  return model

def load_hf_model(model_config):
  # logger.info(f"Loading HuggingFace model from {model_config.tokenizer_name}")
#   hf_model_class_name = HF_MODEL_CLASS_MAPPING.get("flex.Qwen2ForCausalLM")
#   if hf_model_class_name is None:
#     print(f"Error: No HuggingFace model mapping found for flex.Qwen2ForCausalLM")
#     print(f"Available mappings: {list(HF_MODEL_CLASS_MAPPING.keys())}")
#     sys.exit(1)

# # Dynamically import the HuggingFace model class
#   try:
#     transformers_module = importlib.import_module("transformers")
#     hf_model_class = getattr(transformers_module, hf_model_class_name)
#   except (ModuleNotFoundError, AttributeError) as e:
#     print(f"Error importing HuggingFace model class '{hf_model_class_name}': {e}")
#     sys.exit(1)

  hf_model = Qwen2ForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
  )
  return hf_model

@hydra.main(version_base=None, config_path="configs", config_name="sanity")
def main(config: DictConfig):  
  # Configure logging (only on main process to avoid duplicate logs)

  log_level = logging.INFO
  logger.setLevel(log_level)
  logger.setLevel(log_level)
  datasets.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.enable_default_handler()
  transformers.utils.logging.enable_explicit_format()

  # Initialize distributed process group for XLA
  # torch.distributed.init_process_group('gloo', init_method='xla://')

  # Start profiling server (only on main process)

  # TODO(https://github.com/AI-Hypercomputer/torchprime/issues/14): Add tokenizers to torchprime.
  tokenizer_name = config.model.tokenizer_name
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

  # Set the model dtype to bfloat16, and set the default device to the XLA device.
  # This will capture the model constructor into a graph so that we can add
  # sharding annotations to the weights later, and run the constructor on the XLA device.
  # NOTE: read HF model from GCS bucket if checkpoint is not provided, otherwise read from checkpoint_load_dir/checkpoint_load_step in _load_checkpoint()
  load_from_checkpoint = config.checkpoint_load_dir is not None and config.checkpoint_load_step is not None
  model = initialize_model_class(config.model, load_from_hf=not load_from_checkpoint)
  model = model.eval()
  messages = [
    {"role": "user", "content": "Write me a python code to print 'Hello, World!' and loop each character in the string."},
  ]
  inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
  )
  outputs = model.generate(**inputs, max_new_tokens=100)
  logger.info(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
  logger.info(f"model.state_dict().keys() after loading: {model.state_dict().keys()}")


if __name__ == "__main__":
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
  )
  main()

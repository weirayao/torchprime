import logging
import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch.distributed as dist
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass

# Import the initialize_model_class function from train.py
from torchprime.torch_xla_models.train import initialize_model_class, set_default_dtype

# Initialize XLA runtime for TPU
xr.use_spmd()
dist.init_process_group(backend='gloo', init_method='xla://')

def is_main_process():
  """Check if this is the main process (rank 0)."""
  return xr.process_index() == 0


@dataclass
class GenerationConfig:
  diffusion_steps: int = 10
  max_new_tokens: int = 100
  temperature: float = 1.0
  top_p: float = 0.9


def generate(
  model: AutoModelForCausalLM,
  tokenizer: AutoTokenizer,
  inputs: dict[str, torch.Tensor],
  args: GenerationConfig
):
  # Set model to evaluation mode
  model.eval()
  temperature = args.temperature
  top_p = args.top_p

  x = inputs.input_ids


@hydra.main(version_base=None, config_path="configs", config_name="default_inference")
def main(config: DictConfig):
  device = xm.xla_device()
  print(f"Using device: {device}")


  model_config = config.model
  tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_name) # TODO: make our own tokenizer
  with set_default_dtype(torch.bfloat16), torch_xla.device():
    model = initialize_model_class(model_config)



  prompt = "Give me a short introduction to large language model."
  messages = [
    {"role": "user", "content": prompt}
  ]
  # Apply chat template
  text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # Switches between thinking and non-thinking modes
  )
  print(f"Input text: {text}")
    
  # Tokenize input and move to TPU device
  model_inputs = tokenizer([text], return_tensors="pt")
  generation_config = GenerationConfig()
  generation = generate(model, tokenizer, model_inputs, generation_config)
  # Move results back to CPU for processing
  output_ids = generation[0][len(model_inputs.input_ids[0]):].cpu().tolist()
    
  # Parse thinking content (if present)
  try:
    # Find the index of </think> token (151668)
    index = len(output_ids) - output_ids[::-1].index(151668)
  except ValueError:
    index = 0

  # Decode thinking and main content
  thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
  content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
  print("\n" + "="*50)
  print("RESULTS:")
  print("="*50)
  print(f"Thinking content: {thinking_content}")
  print(f"Content: {content}")
  print("="*50)


if __name__ == "__main__":
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
  )
  main()

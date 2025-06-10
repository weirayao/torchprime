import logging
import sys
import torch
import torch_xla
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch.distributed as dist
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding
from dataclasses import dataclass, asdict

# Import the initialize_model_class function from train.py
from torchprime.torch_xla_models.train import initialize_model_class, set_default_dtype

# Initialize XLA runtime for TPU
xr.use_spmd()
dist.init_process_group(backend="gloo", init_method="xla://")


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return xr.process_index() == 0


logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    diffusion_steps: int = 10
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 0.9


def get_anneal_attn_mask(seq_len, bsz, dtype, device, attn_mask_ratio):
    mask = torch.full((seq_len, seq_len), 0, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1)
    causal_mask = mask.to(dtype)

    random_mask = torch.bernoulli(
        torch.full((seq_len, seq_len), 0.0, device=device) + attn_mask_ratio
    )

    anneal_mask = torch.logical_or(causal_mask, random_mask)
    expanded_mask = anneal_mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len)
    inverted_mask = 1.0 - expanded_mask.to(dtype)

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def top_p_logits(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    # import pdb; pdb.set_trace();
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: BatchEncoding,
    args: GenerationConfig,
):
    # Set model to evaluation mode
    model.eval()
    logger.info(f"Start sampling with params: {asdict(args)}")
    temperature = args.temperature
    top_p = args.top_p

    x = inputs.input_ids.to(model.device)


@hydra.main(version_base=None, config_path="configs", config_name="default_inference")
def main(config: DictConfig):
    device = xm.xla_device()
    print(f"Using device: {device}")

    model_config = config.model
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_name
    )
    if not hasattr(tokenizer, "mask_token_id"):
        tokenizer.add_tokens("<|mask|>", special_tokens=True)
        tokenizer.add_special_tokens(
            {"mask_token": "<|mask|>"}, replace_additional_special_tokens=False
        )

    with set_default_dtype(torch.bfloat16), torch_xla.device():
        model = initialize_model_class(model_config)

    prompt = "Give me a short introduction to large language model."
    messages = [{"role": "user", "content": prompt}]
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # Switches between thinking and non-thinking modes
    )
    print(f"Input text: {text}")

    # Tokenize input and move to TPU device
    model_inputs = tokenizer([text], return_tensors="pt")
    generation_config = GenerationConfig(**config.generation)
    generation = generate(model, tokenizer, model_inputs, generation_config)
    # Move results back to CPU for processing
    output_ids = generation[0][len(model_inputs.input_ids[0]) :].cpu().tolist()

    # Parse thinking content (if present)
    try:
        # Find the index of </think> token (151668)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    # Decode thinking and main content
    thinking_content = tokenizer.decode(
        output_ids[:index], skip_special_tokens=True
    ).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    print("\n" + "=" * 50)
    print("RESULTS:")
    print("=" * 50)
    print(f"Thinking content: {thinking_content}")
    print(f"Content: {content}")
    print("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()

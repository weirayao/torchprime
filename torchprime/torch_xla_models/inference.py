import logging
import sys
import torch
import torch_xla
import torch.distributions as dists
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
from torchprime.torch_xla_models.train import initialize_model_class, set_default_dtype, Trainer

# Initialize XLA runtime for TPU
xr.use_spmd()
if not dist.is_initialized():
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


def sample(
    model: PreTrainedModel,
    xt: torch.Tensor,
    x: torch.Tensor,
    attention_mask: torch.Tensor,
    maskable_mask: torch.Tensor,
    temperature: float,
    top_p: float,
    greedy: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample tokens from the model output with top-p filtering and handle shifting.

    Args:
        model: The model to use for inference
        xt: Current input tokens (with masks)
        x: Original input tokens (for shifting)
        attention_mask: Attention mask for the model
        maskable_mask: Mask indicating which positions can be modified
        temperature: Temperature for sampling
        top_p: Top-p threshold for filtering

    Returns:
        Updated x0 tensor with sampled tokens and their scores
    """
    # Get model predictions
    logits, _ = model(xt, attention_mask=attention_mask)
    # logits = top_p_logits(logits / temperature + 1e-5, p=top_p)

    # Convert to probability distribution and sample
    scores = torch.log_softmax(logits, dim=-1)
    if greedy: # TODO: rethink greedy sampling
        _, x0 = scores.max(-1)
    else:
        x0 = dists.Categorical(logits=scores).sample()
        # x0_scores = torch.gather(scores, -1, x0.unsqueeze(-1)).squeeze(-1)
    logger.info(f"x0: {x0}")
    # NOTE: we already cut one token in forward pass, so we don't need to cut x0 here
    # Shift tokens and scores right by 1 position
    x0 = torch.cat([x[:, 0:1], x0], dim=1)
    # x0_scores = torch.cat([x0_scores[:, 0:1], x0_scores[:, :-1]], dim=1)

    # replace output of non-[MASK] positions with xt
    # x0: predicted tokens
    # x0[maskable_mask]: predicted tokens in masked positions
    x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
    return x0


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: dict[str, torch.Tensor],
    args: GenerationConfig,
    verbose: bool = False,
) -> torch.Tensor:
    # Set model to evaluation mode
    model.eval()
    logger.info(f"Start sampling with params: {asdict(args)}")
    temperature = args.temperature
    top_p = args.top_p
    device = xm.xla_device()

    x = inputs["input_ids"].to(device)
    if "src_mask" not in inputs:
        src_mask = (x != tokenizer.mask_token_id).to(device)
    else:
        src_mask = inputs["src_mask"].bool().to(device)

    attention_mask = torch.ones_like(x).to(device) # NOTE: this is actually not used in the model
    maskable_mask = ~src_mask

    # first forward, all position except src is [M]
    # xt: torch.Tensor = x.masked_fill(maskable_mask, tokenizer.mask_token_id)
    xt = x.clone() # NOTE: we already did the masking in prepare_inputs
    if verbose:
        logger.info(f"t={args.diffusion_steps}(in): {tokenizer.decode(xt.tolist()[0])}")
    x0 = sample(
        model, xt, x, attention_mask, maskable_mask, temperature, top_p, greedy=True
    )
    if verbose:
        logger.info(f"t={args.diffusion_steps}(out): {tokenizer.decode(x0.tolist()[0])}")

    for t in range(args.diffusion_steps - 1, 0, -1):
        # select rate% tokens to be still [MASK]
        p_to_x0 = 1 / (t + 1)

        masked_to_x0 = maskable_mask & (
            torch.rand_like(x0, dtype=torch.float) < p_to_x0
        ) # a token is previously [MASK] has probability p_to_x0 to be replaced by x0
        xt.masked_scatter_(masked_to_x0, x0[masked_to_x0])
        maskable_mask = maskable_mask.masked_fill(masked_to_x0, False)
        if verbose:
            logger.info(f"t={t}(in): {tokenizer.decode(xt.tolist()[0])}")

        x0 = sample(
            model, xt, x, attention_mask, maskable_mask, temperature, top_p, greedy=True
        )
        if verbose:
            logger.info(f"t={t}(out): {tokenizer.decode(x0.tolist()[0])}")

    return x0[:, 1:]  # shift left by 1


def prepare_inputs(
    tokenizer: PreTrainedTokenizerBase,
    messages: str | list[dict],
    max_new_tokens: int,
    enable_thinking: bool = True,
) -> tuple[dict[str, torch.Tensor], BatchEncoding]:
    """
    Prepare model inputs by applying chat template, tokenizing, and extending with mask tokens.

    Args:
        tokenizer: The tokenizer to use
        messages: List of chat messages in the format [{"role": "user", "content": "..."}]
        max_new_tokens: Number of new tokens to generate
        enable_thinking: Whether to enable thinking mode in chat template

    Returns:
        BatchEncoding with input_ids and attention_mask extended for generation
    """
    # Apply chat template
    if isinstance(messages, str):
        text_inputs = messages # NOTE: we shift one token and we don't apply chat template
        # if not enable_thinking:
        #     text_inputs += "<think> </think>"
    else:
        text_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    print(f"Input text: {text_inputs}")

    # Tokenize input
    ar_inputs = tokenizer([text_inputs], return_tensors="pt")


    # Extend input_ids with mask tokens for generation
    mask_token_ids = torch.full(
        size=(
            ar_inputs.input_ids.shape[0],
            max_new_tokens,
        ),
        fill_value=tokenizer.mask_token_id,
        dtype=ar_inputs.input_ids.dtype,
    )

    input_ids = torch.cat(
        [
            ar_inputs.input_ids,
            mask_token_ids,
        ],
        dim=1,
    )
    src_mask = torch.where(input_ids == tokenizer.mask_token_id, 0, 1)
    ddlm_inputs = {
        "input_ids": input_ids,
        "src_mask": src_mask,
    }

    return ddlm_inputs, ar_inputs

@hydra.main(version_base=None, config_path="configs", config_name="default_inference")
def main(config: DictConfig):
    if is_main_process():
        logger.info(f"Config: {OmegaConf.to_yaml(config)}")

    model_config = config.model
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_name)
    if tokenizer.mask_token is None:
        tokenizer.add_tokens("<|mask|>", special_tokens=True)
        tokenizer.add_special_tokens(
            {"mask_token": "<|mask|>"}, replace_additional_special_tokens=False
        )

    logger.info("Initializing model...")
    with set_default_dtype(torch.bfloat16), torch_xla.device():
        model = initialize_model_class(model_config)
    xm.wait_device_ops()

    logger.info(f"hf model weights: {model.state_dict()['model.embed_tokens.weight']}")

    trainer = Trainer(model=model, tokenizer=tokenizer, config=config, train_dataset=None)
    trainer._load_checkpoint()

    logger.info(f"ckpt model weights: {trainer.model.state_dict()['model.embed_tokens.weight']}")

    logger.info("Preparing inputs...")
    prompt = "Donald John Trump (born June 14, 1946) is an American <|mask|>, media personality, and businessman who is the 47th <|mask|> of the <|mask|> <|mask|>."
    # messages = [{"role": "user", "content": prompt}]
    messages = prompt
    generation_config = GenerationConfig(**OmegaConf.to_container(config.generation))

    ddlm_inputs, ar_inputs = prepare_inputs(
        tokenizer, messages, generation_config.max_new_tokens, enable_thinking=False
    )

    logger.info("Generating...")
    generation = generate(
        trainer.model, tokenizer, ddlm_inputs, generation_config, verbose=True
    )
    # Move results back to CPU for processing
    output_ids = generation[0][len(ar_inputs.input_ids[0]) :].cpu().tolist()

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

import logging
import torch
import torch_xla
import torch.distributions as dists
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    diffusion_steps: int = 10
    max_tokens: int = 256
    max_new_tokens: int | None = None
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None


def top_p_logits(logits, p: float = 0.9):
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

def top_k_logits(logits, top_k: int):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

@torch.no_grad()
def sample(
    model: PreTrainedModel,
    xt: torch.Tensor,
    x: torch.Tensor,
    attention_mask: torch.Tensor,
    maskable_mask: torch.Tensor,
    temperature: float,
    top_p: float | None = None,
    top_k: int | None = None,
) -> torch.Tensor:
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
        top_k: Top-k threshold for filtering

    Returns:
        Updated x0 tensor with sampled tokens
    """
    print(xt.shape, attention_mask.shape, maskable_mask.shape)
    # Get model predictions
    logits, _ = model(xt, attention_mask=attention_mask)
    if temperature > 0:
        logits = logits / (temperature + 1e-5)
        if top_p is not None and top_p < 1:
            logits = top_p_logits(logits, top_p)
        if top_k is not None:
            logits = top_k_logits(logits, top_k)

    # Convert to probability distribution and sample
    probs = torch.softmax(logits, dim=-1)
    if temperature == 0:
        _, x0 = probs.max(-1)
    else:
        x0 = dists.Categorical(probs=probs).sample()
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
    top_k = args.top_k
    device = torch_xla.device()

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
        logger.info(f"t={args.diffusion_steps}(in): {tokenizer.batch_decode(xt.detach().cpu())}")
    x0 = sample(
        model, xt, x, attention_mask, maskable_mask, temperature, top_p, top_k
    )
    if verbose:
        logger.info(f"t={args.diffusion_steps}(out): {tokenizer.batch_decode(x0.detach().cpu())}")

    for t in range(args.diffusion_steps - 1, 0, -1):
        # select rate% tokens to be still [MASK]
        p_to_x0 = 1 / (t + 1)
        if verbose:
            logger.info(f"p_to_x0: {p_to_x0}")

        masked_to_x0 = maskable_mask & (
            torch.rand_like(x0, dtype=torch.float) < p_to_x0
        ) # a token is previously [MASK] has probability p_to_x0 to be replaced by x0
        if verbose:
            logger.info(f"masked_to_x0: {masked_to_x0}")
            logger.info(f"xt before: {xt}")
        xt.masked_scatter_(masked_to_x0, x0[masked_to_x0])
        if verbose:
            logger.info(f"xt after: {xt}")
        maskable_mask = maskable_mask.masked_fill(masked_to_x0, False)
        if verbose:
            logger.info(f"maskable_mask: {maskable_mask}")
            logger.info(f"t={t}(in): {tokenizer.batch_decode(xt.detach().cpu())}")

        x0 = sample(
            model, xt, x, attention_mask, maskable_mask, temperature, top_p, top_k
        )
        if verbose:
            logger.info(f"t={t}(out): {tokenizer.batch_decode(x0.detach().cpu())}")

    return x0

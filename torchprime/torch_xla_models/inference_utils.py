import logging
import time
from typing import Optional
import torch
import torch_xla
import torch.distributions as dists
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig_:
    diffusion_steps: int = 10
    mask_token_id: int = 151669
    max_tokens: int = 256
    max_new_tokens: int | None = None
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    eps: float = 1e-3
    alg: str = "original"
    alg_temp: float = 0.2
    output_history: bool = False
    return_dict_in_generate: bool = False


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
def sample_(
    logits: torch.Tensor,
    temperature=0.0,
    top_p=None,
    top_k=None,
    neg_entropy=False,
):
    # Timing instrumentation
    total_start_time = time.time()
    logger.info(f"sampling tokens with logits shape: {logits.shape}; temperature: {temperature}; top_p: {top_p}; top_k: {top_k}; neg_entropy: {neg_entropy}")
    
    # Scale by temperature
    logits_processing_start_time = time.time()
    
    # Apply filtering and get probabilities in one go
    topk_time = 0.0
    topp_time = 0.0
    softmax_time = 0.0
    scatter_time = 0.0

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    logits_processing_time = time.time() - logits_processing_start_time
    
    # Sample from the distribution efficiently
    sampling_start_time = time.time()
    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = torch.max(probs, dim=-1)
    else:
        confidence, x0 = torch.max(probs, dim=-1)
    
    sampling_time = time.time() - sampling_start_time
    
    entropy_time = 0.0
    if neg_entropy:
        entropy_start_time = time.time()
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
        entropy_time = time.time() - entropy_start_time
    
    total_time = time.time() - total_start_time
    
    # Log timing results with detailed breakdown
    logger.info(f"sample_ timing - total: {total_time:.4f}s, logits_processing: {logits_processing_time:.4f}s, sampling: {sampling_time:.4f}s, entropy: {entropy_time:.4f}s")
    logger.info(f"sample_ detailed timing - topk: {topk_time:.4f}s, topp: {topp_time:.4f}s, softmax: {softmax_time:.4f}s, scatter: {scatter_time:.4f}s")
    
    return confidence, x0


@torch.no_grad()
def generate_(
    model: PreTrainedModel,
    input_ids: torch.LongTensor,
    generation_config: GenerationConfig_,
) -> dict[str, torch.Tensor | list[torch.Tensor]] | torch.Tensor:
    logger.info(f"Generating with config: {asdict(generation_config)}")
    model.eval()
    device = torch_xla.device()

    # Timing setup
    timing_results = {
        'total_time': 0,
        'model_forward_time': 0,
        'sampling_time': 0,
        'tensor_ops_time': 0,
        'step_times': []
    }
    total_start_time = time.time()

    output_history = generation_config.output_history
    return_dict_in_generate = generation_config.return_dict_in_generate

    mask_token_id = generation_config.mask_token_id
    steps = generation_config.diffusion_steps
    eps = generation_config.eps
    alg = generation_config.alg
    alg_temp = generation_config.alg_temp
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k

    histories = [] if (return_dict_in_generate and output_history) else None

    x = input_ids.to(device)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    
    for i in range(steps):
        step_start_time = time.time()
        tensor_ops_time = 0.0  # Initialize tensor ops time for this step
        sampling_time = 0.0  # Initialize sampling time for this step

        logger.info(f"Diffusion step {i} of {steps}...")
        
        tensor_ops_start = time.time()
        mask_index = x == mask_token_id
        tensor_ops_time += time.time() - tensor_ops_start
        
        # Model forward pass timing
        forward_start_time = time.time()
        logits, _ = model(x, attention_mask=None)  # NOTE: flex model doesn't use attention mask
        forward_time = time.time() - forward_start_time
        timing_results['model_forward_time'] += forward_time
        
        logger.info(f"logits shape: {logits.shape}")
        
        tensor_ops_start = time.time()
        # Optimize logits shifting - avoid cat operation when possible
        shifted_logits = torch.empty_like(logits)
        shifted_logits[:, 0:1] = logits[:, 0:1]  # Copy first token
        shifted_logits[:, 1:] = logits[:, :-1]   # Shift remaining tokens
        
        mask_logits = shifted_logits[mask_index]
        t = timesteps[i]
        s = timesteps[i + 1]
        tensor_ops_time += time.time() - tensor_ops_start

        
        
        if alg == "original":
            logger.info(f"original sampling algorithm...")
            tensor_ops_start = time.time()
            p_transfer = 1 - s / t if i < steps - 1 else 1
            x0 = torch.full_like(x[mask_index], mask_token_id, device=device)
            transfer_index_t_s = torch.rand(*x0.shape, device=device) < p_transfer
            tensor_ops_time += time.time() - tensor_ops_start
            
            sampling_start_time = time.time()
            if transfer_index_t_s.any():
                _, x0[transfer_index_t_s] = sample_(
                    mask_logits[transfer_index_t_s],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            sampling_time = time.time() - sampling_start_time
            timing_results['sampling_time'] += sampling_time

            tensor_ops_start = time.time()
            x[mask_index] = x0.clone()
            tensor_ops_time += time.time() - tensor_ops_start
            
        elif alg == "neg_entropy":
            logger.info(f"negative entropy sampling...")
            sampling_start_time = time.time()
            confidence, x0 = sample_(
                mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True
            )
            sampling_time = time.time() - sampling_start_time
            timing_results['sampling_time'] += sampling_time
            
            tensor_ops_start = time.time()
            num_mask_token = mask_index.sum() / mask_index.shape[0]
            number_transfer_tokens = (
                int(num_mask_token * (1 - s / t))
                if i < steps - 1
                else int(num_mask_token)
            )
            full_confidence = torch.full_like(
                x, -torch.inf, device=device, dtype=logits.dtype
            )
            full_confidence[mask_index] = confidence
            if number_transfer_tokens > 0:
                if alg_temp is None or alg_temp == 0:
                    _, transfer_index = torch.topk(
                        full_confidence, number_transfer_tokens
                    )
                else:
                    full_confidence = full_confidence / alg_temp
                    full_confidence = F.softmax(full_confidence, dim=-1)
                    transfer_index = torch.multinomial(
                        full_confidence, num_samples=number_transfer_tokens
                    )
                x_ = (
                    torch.zeros_like(x, device=device, dtype=torch.long)
                    + mask_token_id
                )
                x_[mask_index] = x0.clone()
                row_indices = (
                    torch.arange(x.size(0), device=device)
                    .unsqueeze(1)
                    .expand_as(transfer_index)
                )
                x[row_indices, transfer_index] = x_[row_indices, transfer_index]
            tensor_ops_time += time.time() - tensor_ops_start
        else:
            raise ValueError(f"Invalid algorithm: {alg}")

        if histories is not None:
            histories.append(x.detach().cpu().clone())

        step_time = time.time() - step_start_time
        timing_results['step_times'].append(step_time)
        timing_results['tensor_ops_time'] += tensor_ops_time
        logger.info(f"Step {i} time: {step_time:.4f}s (forward: {forward_time:.4f}s, sampling: {sampling_time:.4f}s, tensor_ops: {tensor_ops_time:.4f}s)")

    # Single sync at the end instead of every step
    torch_xla.sync()

    timing_results['total_time'] = time.time() - total_start_time
    
    # Print profiling results
    logger.info("=== PROFILING RESULTS ===")
    logger.info(f"Total time: {timing_results['total_time']:.4f}s")
    logger.info(f"Model forward time: {timing_results['model_forward_time']:.4f}s ({timing_results['model_forward_time']/timing_results['total_time']*100:.1f}%)")
    logger.info(f"Sampling time: {timing_results['sampling_time']:.4f}s ({timing_results['sampling_time']/timing_results['total_time']*100:.1f}%)")
    logger.info(f"Tensor ops time: {timing_results['tensor_ops_time']:.4f}s ({timing_results['tensor_ops_time']/timing_results['total_time']*100:.1f}%)")
    logger.info(f"Average step time: {sum(timing_results['step_times'])/len(timing_results['step_times']):.4f}s")
    logger.info(f"Step times: {[f'{t:.4f}' for t in timing_results['step_times']]}")
    logger.info("========================")

    if return_dict_in_generate:
        return {
            "completion": x,
            "history": histories,
            "timing": timing_results,
        }
    else:
        return x


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
    neg_entropy: bool = False,
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
    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    logger.info(f"x0: {x0}")

    # NOTE: we already cut one token in forward pass, so we don't need to cut x0 here
    # Shift tokens and scores right by 1 position
    x0 = torch.cat([x[:, :1], x0], dim=1)
    # x0_scores = torch.cat([x0_scores[:, 0:1], x0_scores[:, :-1]], dim=1)

    # replace output of non-[MASK] positions with xt
    # x0: predicted tokens
    # x0[maskable_mask]: predicted tokens in masked positions
    x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
    return confidence, x0


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
    steps = args.diffusion_steps
    device = torch_xla.device()

    x = inputs["input_ids"].to(device)
    if "src_mask" not in inputs:
        src_mask = (x != tokenizer.mask_token_id).to(device)
    else:
        src_mask = inputs["src_mask"].bool().to(device)

    attention_mask = torch.ones_like(x).to(
        device
    )  # NOTE: this is actually not used in the model
    mask_positions = ~src_mask

    # first forward, all position except src is [M]
    # xt: torch.Tensor = x.masked_fill(maskable_mask, tokenizer.mask_token_id)
    xt = x.clone()  # NOTE: we already did the masking in prepare_inputs
    if verbose:
        logger.info(f"t={steps}(in): {tokenizer.batch_decode(xt.detach().cpu())}")
    confidence, x0 = sample(
        model, xt, x, attention_mask, mask_positions, temperature, top_p, top_k
    )
    if verbose:
        logger.info(f"t={steps}(out): {tokenizer.batch_decode(x0.detach().cpu())}")

    for t in range(steps - 1, 0, -1):
        # select rate% tokens to be still [MASK]
        p_to_x0 = 1 / (t + 1)
        if verbose:
            logger.info(f"p_to_x0: {p_to_x0}")

        masked_to_x0 = mask_positions & (
            torch.rand_like(x0, dtype=torch.float) < p_to_x0
        )  # a token is previously [MASK] has probability p_to_x0 to be replaced by x0
        if verbose:
            logger.info(f"masked_to_x0: {masked_to_x0}")
            logger.info(f"xt before: {xt}")
        xt.masked_scatter_(masked_to_x0, x0[masked_to_x0])
        if verbose:
            logger.info(f"xt after: {xt}")
        mask_positions = mask_positions.masked_fill(masked_to_x0, False)
        if verbose:
            logger.info(f"maskable_mask: {mask_positions}")
            logger.info(f"t={t}(in): {tokenizer.batch_decode(xt.detach().cpu())}")

        confidence, x0 = sample(
            model, xt, x, attention_mask, mask_positions, temperature, top_p, top_k
        )
        if verbose:
            logger.info(f"t={t}(out): {tokenizer.batch_decode(x0.detach().cpu())}")

    return x0

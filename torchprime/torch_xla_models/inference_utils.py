import logging
import time
from typing import Optional
import torch
import torch.distributions as dists
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase, BatchEncoding
from dataclasses import dataclass, asdict

IS_TPU = not torch.cuda.is_available()
if IS_TPU:
    import torch_xla


logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig_:
    noise_level: float = 0.1 # NOTE: for unmasking tasks
    seed: int = 42
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


def prepare_inputs(
    tokenizer: PreTrainedTokenizerBase,
    messages: str | list[dict],
    # args: GenerationConfig,
    args: GenerationConfig_,
    enable_thinking: bool = True,
    noise_ratio: float = 0.0,
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
        text_inputs = (
            messages  # NOTE: we shift one token and we don't apply chat template
        )
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
    ar_inputs = tokenizer(text_inputs, return_tensors="pt")
    input_ids = ar_inputs.input_ids

    # if noise_ratio > 0:
    #     # Randomly replace noise_ratio proportion of tokens with mask token
    #     mask_indices = torch.rand_like(input_ids.float()) < noise_ratio
    #     input_ids = torch.where(mask_indices, tokenizer.mask_token_id, input_ids)

    # Use max_tokens if provided and > 0, otherwise use max_new_tokens
    num_new_tokens = (
        args.max_tokens - ar_inputs.input_ids.shape[1]
        if args.max_tokens > 0
        else args.max_new_tokens if args.max_new_tokens is not None else 0
    )

    # Extend input_ids with mask tokens for generation
    if num_new_tokens > 0:
        mask_token_ids = torch.full(
            size=(
                input_ids.shape[0],
                num_new_tokens,
            ),
            fill_value=tokenizer.mask_token_id,
            dtype=ar_inputs.input_ids.dtype,
        )
        input_ids = torch.cat(
            [
                input_ids,
                mask_token_ids,
            ],
            dim=1,
        )
    if IS_TPU:
        # Right pad input_ids to nearest multiple of 256
        seq_len = input_ids.shape[1]
        pad_len = (256 - seq_len % 256) % 256  # Calculate padding needed
        if pad_len > 0:
            pad_ids = torch.full(
                (input_ids.shape[0], pad_len),
                tokenizer.pad_token_id,
                dtype=input_ids.dtype,
            )
            input_ids = torch.cat([input_ids, pad_ids], dim=1)

    input_ids = input_ids.squeeze(0)
    src_mask = torch.where(input_ids == tokenizer.mask_token_id, 0, 1)

    print(f"input_ids: {input_ids.shape}")
    print(f"src_mask: {src_mask.shape}")
    ddlm_inputs = {
        "input_ids": input_ids,
        "src_mask": src_mask,
    }

    return ddlm_inputs, ar_inputs


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
    logits_processing_time = topp_time = topk_time = 0.0
    total_start_time = time.time()
    logger.info(
        f"sampling tokens with logits shape: {logits.shape}; temperature: {temperature}; top_p: {top_p}; top_k: {top_k}; neg_entropy: {neg_entropy}"
    )

    # Scale by temperature
    logits_processing_start_time = time.time()

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        topp_time = time.time()
        logits = top_p_logits(logits, top_p)
        topp_time = time.time() - topp_time
    if top_k is not None:
        topk_time = time.time()
        logits = top_k_logits(logits, top_k)
        topk_time = time.time() - topk_time
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
    logger.info(
        f"sample_ timing - total: {total_time:.4f}s, logits_processing: {logits_processing_time:.4f}s, sampling: {sampling_time:.4f}s, entropy: {entropy_time:.4f}s"
    )
    logger.info(
        f"sample_ detailed logits processing timing - total: {logits_processing_time:.4f}s, topk: {topk_time:.4f}s, topp: {topp_time:.4f}s"
    )

    return confidence, x0


@torch.no_grad()
def generate_(
    model: PreTrainedModel,
    input_ids: torch.LongTensor,
    generation_config: GenerationConfig_,
) -> dict[str, torch.Tensor | list[torch.Tensor]] | torch.Tensor:
    logger.info(f"Generating with config: {asdict(generation_config)}")
    model.eval()
    device = torch_xla.device() if IS_TPU else torch.device("cuda")

    # Timing setup
    timing_results = {
        "total_time": 0,
        "model_forward_time": 0,
        "sampling_time": 0,
        "tensor_ops_time": 0,
        "step_times": [],
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

    if IS_TPU:
        x = input_ids.to(device)
    elif not input_ids.device.type.startswith("cuda"):
        x = input_ids.to(device)
    else:
        x = input_ids
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
        logits, _ = model(
            x, attention_mask=None
        )  # NOTE: flex model doesn't use attention mask
        forward_time = time.time() - forward_start_time
        timing_results["model_forward_time"] += forward_time

        tensor_ops_start = time.time()
        # Optimize logits shifting - avoid cat operation when possible
        shifted_logits = torch.empty_like(logits, device=device)
        shifted_logits[:, 0:1] = logits[:, 0:1]  # Copy first token
        shifted_logits[:, 1:] = logits[:, :-1]  # Shift remaining tokens

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
            timing_results["sampling_time"] += sampling_time

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
            timing_results["sampling_time"] += sampling_time

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
                    # Add numerical stability for temperature scaling and softmax
                    scaled_confidence = full_confidence / alg_temp
                    
                    # Clamp extreme values to prevent numerical instability
                    scaled_confidence = torch.clamp(scaled_confidence, min=-50, max=50)
                    
                    # Apply softmax
                    full_confidence_probs = F.softmax(scaled_confidence, dim=-1)
                    
                    # Validate probabilities before multinomial sampling
                    if torch.isnan(full_confidence_probs).any() or torch.isinf(full_confidence_probs).any():
                        logger.warning("Invalid probabilities detected, falling back to topk sampling")
                        _, transfer_index = torch.topk(
                            full_confidence, number_transfer_tokens
                        )
                    else:
                        # Ensure probabilities are non-negative and sum to 1
                        full_confidence_probs = torch.clamp(full_confidence_probs, min=1e-8)
                        full_confidence_probs = full_confidence_probs / full_confidence_probs.sum(dim=-1, keepdim=True)
                        
                        try:
                            transfer_index = torch.multinomial(
                                full_confidence_probs, num_samples=number_transfer_tokens
                            )
                        except RuntimeError as e:
                            logger.warning(f"Multinomial sampling failed: {e}, falling back to topk sampling")
                            _, transfer_index = torch.topk(
                                full_confidence, number_transfer_tokens
                            )
                x_ = (
                    torch.zeros_like(x, device=device, dtype=torch.long) + mask_token_id
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
        timing_results["step_times"].append(step_time)
        timing_results["tensor_ops_time"] += tensor_ops_time
        logger.info(
            f"Step {i} time: {step_time:.4f}s (forward: {forward_time:.4f}s, sampling: {sampling_time:.4f}s, tensor_ops: {tensor_ops_time:.4f}s)"
        )

    # Single sync at the end instead of every step
    if IS_TPU:
        torch_xla.sync()

    timing_results["total_time"] = time.time() - total_start_time

    # Print profiling results
    logger.info("=== PROFILING RESULTS ===")
    logger.info(f"Total time: {timing_results['total_time']:.4f}s")
    logger.info(
        f"Model forward time: {timing_results['model_forward_time']:.4f}s ({timing_results['model_forward_time']/timing_results['total_time']*100:.1f}%)"
    )
    logger.info(
        f"Sampling time: {timing_results['sampling_time']:.4f}s ({timing_results['sampling_time']/timing_results['total_time']*100:.1f}%)"
    )
    logger.info(
        f"Tensor ops time: {timing_results['tensor_ops_time']:.4f}s ({timing_results['tensor_ops_time']/timing_results['total_time']*100:.1f}%)"
    )
    logger.info(
        f"Average step time: {sum(timing_results['step_times'])/len(timing_results['step_times']):.4f}s"
    )
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

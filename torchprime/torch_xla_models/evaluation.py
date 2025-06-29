import logging
import sys
import torch
import torch_xla
import json
from pathlib import Path
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch.distributed as dist
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm
from torchprime.torch_xla_models.train import (
    initialize_model_class,
    set_default_dtype,
    Trainer,
)
from torchprime.torch_xla_models.inference_utils import GenerationConfig, generate

# Initialize XLA runtime for TPU
xr.use_spmd()
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", init_method="xla://")


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return xr.process_index() == 0


logger = logging.getLogger(__name__)


def prepare_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    args: GenerationConfig,
) -> Dataset:
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

    if noise_ratio > 0:
        # Randomly replace noise_ratio proportion of tokens with mask token
        mask_indices = torch.rand_like(input_ids.float()) < noise_ratio
        input_ids = torch.where(mask_indices, tokenizer.mask_token_id, input_ids)

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
    # Left pad input_ids to nearest multiple of 256
    seq_len = input_ids.shape[1]
    pad_len = (256 - seq_len % 256) % 256  # Calculate padding needed
    if pad_len > 0:
        pad_ids = torch.full(
            (input_ids.shape[0], pad_len), tokenizer.pad_token_id, dtype=input_ids.dtype
        )
        input_ids = torch.cat([pad_ids, input_ids], dim=1)

    input_ids = input_ids.squeeze(0)
    src_mask = torch.where(input_ids == tokenizer.mask_token_id, 0, 1)

    print(f"input_ids: {input_ids.shape}")
    print(f"src_mask: {src_mask.shape}")
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

    generation_config = GenerationConfig(**OmegaConf.to_container(config.generation))
    logger.info(f"Evaluation with generation_config: {generation_config}")

    logger.info("Loading evaluation dataset...")
    match config.eval_dataset_name_or_path:
        case "loubnabnl/humaneval_infilling":
            dataset = load_dataset(config.eval_dataset_name_or_path, config="HumanEval-RandomSpanInfillingLight", split="test")
        case "openai/openai_humaneval":
            dataset = load_dataset(config.eval_dataset_name_or_path, split="test")
        case _:
            raise ValueError(f"Unsupported dataset: {config.eval_dataset_name_or_path}")
    # TODO: Need to assemble and pretokenize the query.
    dataset = prepare_dataset(tokenizer, dataset)

    logger.info("Loading model checkpoint...")
    trainer = Trainer(
        model=model, tokenizer=tokenizer, config=config, train_dataset=dataset
    )
    trainer._load_checkpoint()
    loader = trainer._get_train_dataloader() # TODO: think about if we need to implement a new dataloader for evaluation
    iterator = iter(loader)


    logger.info("Evaluating...")
    generation_results = []
    for _, batch in tqdm(enumerate(iterator)):
        generation = generate(
            trainer.model, tokenizer, batch, generation_config, verbose=True
        )
        generation = generation.cpu().tolist()
        generation_text = tokenizer.batch_decode(generation, skip_special_tokens=True)
        if is_main_process():
            generation_results.append(generation_text)

    if is_main_process():
        eval_results = eval_func(generation_results)
        save_path = Path(config.eval_results_save_path) / config.eval_dataset_name_or_path / f"{config.checkpoint_path.split('/')[-1]}_{config.resume_from_checkpoint}.json"
        with open(save_path, "w") as f:
            json.dump(eval_results, f)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()

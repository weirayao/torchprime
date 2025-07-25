import logging
import sys
import torch
import torch_xla
import json
from datetime import datetime
from pathlib import Path
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch.distributed as dist
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset, Dataset, concatenate_datasets
from datasets.distributed import split_dataset_by_node
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm
from torchprime.torch_xla_models.train import (
    initialize_model_class,
    set_default_dtype,
    Trainer,
)
from torchprime.torch_xla_models.inference_utils import GenerationConfig_, generate_

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
    args: GenerationConfig_,
) -> Dataset:
    """
    Prepare model inputs by applying chat template, tokenizing, and extending with mask tokens.

    Args:
        tokenizer: The tokenizer to use
        dataset: The dataset to use
        args: GenerationConfig_

    Returns:
        Dataset with input_ids extended for generation
    """

    # Assemble the query:
    match dataset.config_name:
        case "HumanEval-RandomSpanInfillingLight":
            dataset = dataset.map(
                lambda x: {
                    "canonical_solution_length": len(
                        tokenizer(x["canonical_solution"])["input_ids"]
                    )
                }
            )

            def assemble_query(x):
                prompts = x["prompt"]
                suffixes = x["suffix"]
                canonical_solution_lengths = x["canonical_solution_length"]
                queries = []
                for prompt, suffix, length in zip(
                    prompts, suffixes, canonical_solution_lengths
                ):
                    num_infill_tokens = (
                        max(args.max_new_tokens, length)
                        if args.max_new_tokens is not None
                        else length
                    )
                    query = (
                        "You are a helpful assistant. Please fill in the missing code to complete the following python function:\n```python\n"
                        + prompt
                        + tokenizer.mask_token * num_infill_tokens
                        + suffix
                        + "\n```"
                    )
                    queries.append(query)
                return {"query": queries}

            dataset = dataset.map(assemble_query, batched=True)

        case "openai_humaneval":
            def assemble_query(x):
                prompts = x["prompt"]
                queries = [
                    "Please complete the following python function:\n```python\n"
                    + prompt
                    + tokenizer.mask_token * args.max_new_tokens
                    for prompt in prompts
                ]
                return {"query": queries}

            dataset = dataset.map(assemble_query, batched=True)

        case _:
            raise ValueError(f"Unsupported dataset: {dataset.config_name}")
    # TODO: apply chat template when evaluating instructed models
    # text_inputs = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True,
    #     enable_thinking=enable_thinking,
    # )
    # print(f"Input text: {text_inputs}")

    # Tokenize input
    column_names = list(dataset.features)
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["query"]), batched=True, remove_columns=column_names
    )
    # Find the maximum length in the dataset efficiently
    max_length = max(
        tokenized_dataset.map(lambda x: {"length": len(x["input_ids"])})["length"]
    )

    # Round up to nearest multiple of 512 for XLA efficiency
    target_length = ((max_length + 512) // 512) * 512
    logger.info(
        f"Padding all sequences to fixed length: {target_length} (max found: {max_length})"
    )

    # Apply padding as a transform on tokenized dataset
    def pad_to_fixed_length(batch):
        """Pad input_ids to fixed length for consistent batch shapes."""
        input_ids = batch["input_ids"]
        attention_masks = batch["attention_mask"]
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(input_ids, attention_masks):
            current_length = len(ids)
            pad_len = target_length - current_length
            if pad_len > 0:
                # TODO: left-pad or right-pad?
                ids += [tokenizer.pad_token_id] * pad_len
                mask += [1] * pad_len
            padded_ids.append(ids)
            padded_masks.append(mask)
        return {"input_ids": padded_ids, "attention_mask": padded_masks}

    # Remove the lengths column and apply padding
    tokenized_dataset = tokenized_dataset.map(pad_to_fixed_length, batched=True)

    return tokenized_dataset


@hydra.main(version_base=None, config_path="configs", config_name="default_evaluation")
def main(config: DictConfig):
    if is_main_process():
        logger.info(f"Config: {OmegaConf.to_yaml(config)}")

    model_config = config.model
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_config.tokenizer_name)
    if tokenizer.mask_token is None:
        tokenizer.add_tokens("<|mask|>", special_tokens=True)
        tokenizer.add_special_tokens(
            {"mask_token": "<|mask|>"}, replace_additional_special_tokens=False
        )

    logger.info("Initializing model...")
    with set_default_dtype(torch.bfloat16), torch_xla.device():
        model = initialize_model_class(model_config)
    xm.wait_device_ops()

    generation_config = GenerationConfig_(**OmegaConf.to_container(config.generation))
    logger.info(f"Evaluation with generation_config: {generation_config}")

    logger.info("Loading evaluation dataset...")
    match config.eval_dataset_name_or_path:
        case "loubnabnl/humaneval_infilling":
            dataset = load_dataset(
                config.eval_dataset_name_or_path,
                name="HumanEval-RandomSpanInfillingLight",
                split="test",
                trust_remote_code=True,
            )
        case "openai/openai_humaneval":
            dataset = load_dataset(
                config.eval_dataset_name_or_path, split="test", trust_remote_code=True
            )
        case _:
            raise ValueError(f"Unsupported dataset: {config.eval_dataset_name_or_path}")
    tokenized_dataset = prepare_dataset(tokenizer, dataset, generation_config)
    eval_dataset_len = len(tokenized_dataset)
    logger.info(f"Evaluation dataset length: {eval_dataset_len}")
    if (num_dummy_batches := eval_dataset_len % config.global_batch_size) != 0:
        logger.warning(
            f"Evaluation dataset length {eval_dataset_len} is not divisible by global batch size {config.global_batch_size}, will append {num_dummy_batches} dummy batches"
        )
        seq_len = len(tokenized_dataset["input_ids"][0])
        dummy_batches = Dataset.from_list(
            [
                {
                    "input_ids": [tokenizer.pad_token_id] * seq_len,
                    "attention_mask": [1] * seq_len,
                }
                for _ in range(num_dummy_batches)
            ]
        )
        tokenized_dataset = concatenate_datasets([tokenized_dataset, dummy_batches])
    tokenized_dataset = split_dataset_by_node(tokenized_dataset, xr.process_index(), xr.process_count())

    logger.info("Loading model checkpoint...")
    trainer = Trainer(
        model=model, tokenizer=tokenizer, config=config, eval_dataset=tokenized_dataset
    )
    trainer._load_checkpoint()
    loader = trainer._get_eval_dataloader()
    iterator = iter(loader)

    logger.info("Evaluating...")
    if generation_config.return_dict_in_generate:
        generations = []
    completions = []
    raw_text = []
    for _, batch in tqdm(enumerate(iterator)):
        generation = generate_(trainer.model, batch["input_ids"], generation_config)
        if generation_config.return_dict_in_generate:
            completion = generation["completion"].detach().cpu().tolist()
        else:
            completion = generation.detach().cpu().tolist()
        raw_generation_text = tokenizer.batch_decode(completion, skip_special_tokens=False)
        match config.eval_dataset_name_or_path:
            case "loubnabnl/humaneval_infilling":
                parsed_generation_text = [
                    x.split("```python")[1].split("```")[0] for x in raw_generation_text
                ]
            case "openai/openai_humaneval":
                parsed_generation_text = [
                    x.split("```python")[1].split("```")[0] for x in raw_generation_text
                ]
            case _:
                raise ValueError(f"Unsupported dataset: {config.eval_dataset_name_or_path}")
        if generation_config.return_dict_in_generate:
            generations.extend(generation)
        completions.extend(parsed_generation_text)
        raw_text.extend(raw_generation_text)

    completions = completions[:eval_dataset_len]
    raw_text = raw_text[:eval_dataset_len]
    if generation_config.return_dict_in_generate:
        generations = generations[:eval_dataset_len]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = (
        Path(config.eval_results_save_path)
        / config.eval_dataset_name_or_path
        / f"{config.checkpoint_dir.split('/')[-1]}_{config.resume_from_checkpoint}_{timestamp}.json"
    )
    # Create directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = dataset.add_column("completion", completions)
    dataset = dataset.add_column("raw_completion", raw_text)
    dataset.to_json(save_path.with_suffix(".jsonl"))
    if generation_config.return_dict_in_generate:
        with open(save_path, "w") as f:
            json.dump(generations, f, indent=4)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()

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
        dataset: The dataset to use
        args: GenerationConfig

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
                    query = prompt + tokenizer.mask_token * num_infill_tokens + suffix
                    queries.append(query)
                return {"query": queries}

            dataset = dataset.map(assemble_query, batched=True)

        case "openai_humaneval":
            pass
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
    def find_max_length(batch):
        lengths = [len(ids) for ids in batch["input_ids"]]
        batch["lengths"] = lengths
        return batch
    
    dataset_with_lengths = tokenized_dataset.map(find_max_length, batched=True)
    max_length = max(length for batch in dataset_with_lengths for length in batch["lengths"])
    
    # Round up to nearest multiple of 256 for XLA efficiency
    target_length = ((max_length + 255) // 256) * 256
    logger.info(f"Padding all sequences to fixed length: {target_length} (max found: {max_length})")
    
    # Apply padding as a transform on tokenized dataset
    def pad_to_fixed_length(batch):
        """Pad input_ids to fixed length for consistent batch shapes."""
        padded_batch = {"input_ids": []}
        
        for ids in batch["input_ids"]:
            current_length = len(ids)
            if current_length > target_length:
                # Truncate if longer than target (shouldn't happen with current setup)
                ids = ids[:target_length]
                logger.warning(f"Truncating sequence from {current_length} to {target_length}")
            elif current_length < target_length:
                # Pad if shorter
                pad_len = target_length - current_length
                ids = [tokenizer.pad_token_id] * pad_len + ids
            
            padded_batch["input_ids"].append(ids)
        
        return padded_batch
    
    # Remove the lengths column and apply padding
    tokenized_dataset = dataset_with_lengths.map(
        pad_to_fixed_length, batched=True, remove_columns=["lengths"]
    )

    return tokenized_dataset


@hydra.main(version_base=None, config_path="configs", config_name="default_evaluation")
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
    tokenized_dataset = prepare_dataset(tokenizer, dataset)

    logger.info("Loading model checkpoint...")
    trainer = Trainer(
        model=model, tokenizer=tokenizer, config=config, train_dataset=tokenized_dataset
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
        break # NOTE: debug

    if is_main_process():
        save_path = Path(config.eval_results_save_path) / config.eval_dataset_name_or_path / f"{config.checkpoint_path.split('/')[-1]}_{config.resume_from_checkpoint}.json"
        # Create directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "w") as f:
            json.dump(generation_results, f)
        dataset.add_column("generation", generation_results)
        dataset.to_json(save_path.with_suffix(".jsonl"))

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()

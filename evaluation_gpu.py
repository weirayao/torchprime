import os
import sys
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
import torch
import json
import random
from datetime import datetime
from pathlib import Path
import hydra
import accelerate
from accelerate.utils import gather_object
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase, HfArgumentParser, set_seed
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchprime.torch_xla_models.model_utils import (
    set_default_dtype,
    load_hf_model,
    load_safetensors_to_state_dict,
)
from torchprime.torch_xla_models.inference_utils import GenerationConfig_, generate_, prepare_inputs
from torchprime.torch_xla_models.flex.modeling_qwen import Qwen3ForCausalLM
from gpu_utils import download_gcs_checkpoint

logger = logging.getLogger(__name__)


def assemble_query_humaneval_infilling(x, tokenizer, args):
    prompts = x["prompt"]
    suffixes = x["suffix"]
    canonical_solution_lengths = x["canonical_solution_length"]
    queries = []
    for prompt, suffix, length in zip(
        prompts, suffixes, canonical_solution_lengths
    ):
        num_infill_tokens = (
            max(args.generation.max_new_tokens, length)
            if args.generation.max_new_tokens is not None
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


def assemble_query_humaneval(x, tokenizer, noise_level, seed):
    prompts = x["prompt"]
    solutions = x["canonical_solution"]

    # Set fixed seed for reproducible noise generation
    torch.manual_seed(seed)
    
    # Batch tokenize all solutions for efficiency
    solution_encodings = tokenizer(solutions, add_special_tokens=False, padding=False)
    solution_token_ids = solution_encodings["input_ids"]
    mask_token_id = tokenizer.mask_token_id

    queries = []
    prefix = "You are a helpful assistant. Please fill in the missing code to complete the following python function:\n```python\n"
    suffix = "\n```"    
    for prompt, token_ids in zip(prompts, solution_token_ids):
        # Convert to torch tensor for efficient operations
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        
        # Generate noise mask using torch operations
        noise_mask = torch.rand(len(token_ids)) < noise_level
        
        # Apply noise efficiently using torch.where
        noisy_token_ids = torch.where(noise_mask, mask_token_id, token_tensor)
        
        # Decode the noisy tokens back to string
        noisy_solution = tokenizer.decode(noisy_token_ids.tolist(), skip_special_tokens=False)
        
        # Assemble query by concatenating prompt and noisy solution
        query = prefix + prompt + noisy_solution + suffix
        queries.append(query)

    return {"query": queries}


def prepare_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    args: DictConfig,
    noise_level: float = None,
) -> Dataset:
    """
    Prepare model inputs by applying chat template, and extending with mask tokens.

    Args:
        tokenizer: The tokenizer to use
        dataset: The dataset to use
        args: GenerationConfig_
        noise_level: Current noise level to use (for humaneval dataset)

    Returns:
        Dataset with query extended for generation
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
            assemble_query = assemble_query_humaneval_infilling
            dataset = dataset.map(assemble_query, fn_kwargs={"tokenizer": tokenizer, "args": args}, batched=True)
        case "openai_humaneval":
            assemble_query = assemble_query_humaneval
            dataset = dataset.map(assemble_query, fn_kwargs={"tokenizer": tokenizer, "noise_level": noise_level, "seed": args.seed}, batched=True)

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
    return dataset



@hydra.main(version_base=None, config_path="torchprime/torch_xla_models/configs", config_name="default_evaluation_gpu")
def main(config: DictConfig):
    logger.info(f"Config: {OmegaConf.to_yaml(config)}")

    # Initialize model (once, outside loops)
    if config.baseline_model_name_or_path is None:
        gcs_bucket = "gs://sfr-text-diffusion-model-research/consolidated_checkpoints"
        local_checkpoint_path = "/export/agentstudio-family-2/haolin/consolidated_checkpoints"
        checkpoint = os.path.join(config.checkpoint_dir, str(config.resume_from_checkpoint))

        model_config = config.model
        logger.info("\nInitializing model...")
        model_checkpoint_path = download_gcs_checkpoint(gcs_bucket, checkpoint, local_checkpoint_path)
        with set_default_dtype(torch.bfloat16):
            model = Qwen3ForCausalLM(model_config)
            state_dict = load_safetensors_to_state_dict(model_checkpoint_path)
            model.load_state_dict(state_dict)
        model.eval()
    else:
        model = AutoModel.from_pretrained(config.baseline_model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        model.eval()
    logger.info(f"Model initialized successfully!")
    logger.info(f"Model class: {model.__class__.__name__}")
    logger.info(f"Model device: {next(model.parameters()).device}")

    # Initialize tokenizer (once, outside loops)
    if config.baseline_model_name_or_path is None:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_config.tokenizer_name, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.baseline_model_name_or_path, trust_remote_code=True)
    if tokenizer.mask_token is None:
        tokenizer.add_tokens("<|mask|>", special_tokens=True)
        tokenizer.add_special_tokens(
            {"mask_token": "<|mask|>"}, replace_additional_special_tokens=False
        )
    logger.info(f"Tokenizer initialized: {tokenizer.__class__.__name__}")
    logger.info(f"Mask token ID: {tokenizer.mask_token_id}")

    # Initialize generation config (once, outside loops)
    generation_config = GenerationConfig_(**OmegaConf.to_container(config.generation))
    if config.baseline_model_name_or_path is not None:
        generation_config = model.generation_config
        model.generation_config.alg = "entropy" if config.generation.alg == "neg_entropy" else "origin"
        model.generation_config.alg_temp = config.generation.alg_temp
        model.generation_config.eps = config.generation.eps
        model.generation_config.mask_token_id = tokenizer.mask_token_id
        model.generation_config.temperature = config.generation.temperature
        model.generation_config.top_p = config.generation.top_p
        model.generation_config.top_k = config.generation.top_k

    # Load base dataset (once, outside loops)
    match config.eval_dataset_name_or_path:
        case "loubnabnl/humaneval_infilling":
            base_dataset = load_dataset(
                config.eval_dataset_name_or_path,
                name="HumanEval-RandomSpanInfillingLight",
                split="test",
                trust_remote_code=True,
            )
        case "openai/openai_humaneval":
            base_dataset = load_dataset(
                config.eval_dataset_name_or_path, split="test", trust_remote_code=True
            )
        case _:
            raise ValueError(f"Unsupported dataset: {config.eval_dataset_name_or_path}")

    # Initialize accelerator (once, outside loops)
    accelerator = accelerate.Accelerator()
    model = accelerator.prepare(model)
    device = accelerator.device

    # Convert noise_levels to list if it's a single value
    noise_levels = config.noise_levels if isinstance(config.noise_levels, list) else [config.noise_levels]
    
    # Loop over noise levels and repeats
    for noise_level in noise_levels:
        logger.info(f"\n=== Starting evaluation for noise_level: {noise_level} ===")
        
        for repeat in range(config.repeats):
            logger.info(f"\n--- Repeat {repeat + 1}/{config.repeats} for noise_level {noise_level} ---")
            
            # Set seed for this specific run (deterministic but different for each run)
            run_seed = config.seed + repeat + int(noise_level * 1000)
            set_seed(run_seed)
            logger.info(f"Using seed: {run_seed}")
            
            # Prepare dataset for this specific noise level
            dataset = prepare_dataset(tokenizer, base_dataset, config, noise_level)
            logger.info(f"Dataset: {dataset}")

            eval_dataloader = DataLoader(
                dataset, batch_size=config.eval_batch_size, shuffle=False
            )
            eval_dataloader = accelerator.prepare(eval_dataloader)

            logger.info("Evaluating...")
            generations = []
            with torch.no_grad():
                for batch in tqdm(eval_dataloader):
                    input_ids = tokenizer(batch["query"], padding=True, return_tensors="pt")["input_ids"].to(device)

                    # Reset diffusion steps to the number of mask tokens as we are unmasking instead of continuing texts
                    num_mask_tokens = (input_ids == tokenizer.mask_token_id).sum(dim=1).max().item()
                    logger.info(f"Overriding diffusion steps to {num_mask_tokens}")

                    if config.baseline_model_name_or_path is None:
                        generation_config.diffusion_steps = num_mask_tokens
                        generation = generate_(model, input_ids, generation_config)
                    else:
                        model.module.generation_config.steps = num_mask_tokens
                        generation = model.module.diffusion_generate(input_ids, model.module.generation_config)

                    generation_cpu = generation.detach().cpu().tolist()
                    generations.extend(gather_object(tokenizer.batch_decode(generation_cpu, skip_special_tokens=True)))

            generations = generations[:len(dataset)]
            completions = [
                x.split("```python")[1].split("```")[0]
                if "```python" in x and "```" in x.split("```python")[1]
                else x
                for x in generations
            ]
            
            # Save results with noise_level and repeat in filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if config.baseline_model_name_or_path is None:
                save_path = (
                    Path(config.eval_results_save_path)
                    / config.eval_dataset_name_or_path
                    / f"{config.checkpoint_dir.split('/')[-1]}_{config.resume_from_checkpoint}_noise_{noise_level}_repeat_{repeat + 1}_{timestamp}.json"
                )
            else:
                save_path = (
                    Path(config.eval_results_save_path)
                    / config.eval_dataset_name_or_path
                    / f"{config.baseline_model_name_or_path.split('/')[-1]}_noise_{noise_level}_repeat_{repeat + 1}_{timestamp}.json"
                )
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                dataset = dataset.add_column("completion", completions)
                dataset.to_json(save_path.with_suffix(".jsonl"))
                metadata = {
                    "noise_level": noise_level,
                    "repeat": repeat + 1,
                    "run_seed": run_seed,
                    "base_seed": config.seed,
                    "generation_config": OmegaConf.to_container(config.generation),
                    "dataset_name": config.eval_dataset_name_or_path,
                    "run_name": config.checkpoint_dir.split('/')[-1],
                    "resume_from_checkpoint": config.resume_from_checkpoint,
                }
                metadata_path = save_path.with_suffix(".metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=4)
                logger.info(f"Results saved to: {save_path}")
            except Exception as e:
                logger.error(f"Failed to save dataset: {e}")

    logger.info(f"\n=== Evaluation completed for all noise levels and repeats ===")


if __name__ == "__main__":
    main()
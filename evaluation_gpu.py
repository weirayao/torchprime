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
from datetime import datetime
from pathlib import Path
import hydra
import accelerate
from accelerate.utils import gather_object
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, PreTrainedTokenizerBase, HfArgumentParser, set_seed
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

def prepare_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    args: GenerationConfig_,
) -> Dataset:
    """
    Prepare model inputs by applying chat template, and extending with mask tokens.

    Args:
        tokenizer: The tokenizer to use
        dataset: The dataset to use
        args: GenerationConfig_

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

    return dataset



@hydra.main(version_base=None, config_path="torchprime/torch_xla_models/configs", config_name="default_evaluation_gpu")
def main(config: DictConfig):
    logger.info(f"Config: {OmegaConf.to_yaml(config)}")

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

    logger.info(f"Model initialized successfully!")
    logger.info(f"Model class: {model.__class__.__name__}")
    logger.info(f"Model device: {next(model.parameters()).device}")

    generation_config = GenerationConfig_(**OmegaConf.to_container(config.generation))


    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_config.tokenizer_name)
    if tokenizer.mask_token is None:
        tokenizer.add_tokens("<|mask|>", special_tokens=True)
        tokenizer.add_special_tokens(
            {"mask_token": "<|mask|>"}, replace_additional_special_tokens=False
        )
    logger.info(f"Tokenizer initialized: {tokenizer.__class__.__name__}")
    logger.info(f"Mask token ID: {tokenizer.mask_token_id}")


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

    dataset = prepare_dataset(tokenizer, dataset, generation_config)
    logger.info(f"Dataset: {dataset}")

    eval_dataloader = DataLoader(
        dataset, batch_size=config.eval_batch_size, shuffle=False
    )

    accelerator = accelerate.Accelerator()
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    device = accelerator.device

    logger.info("Evaluating...")
    generations = []
    completions = []
    raw_text = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            input_ids = tokenizer(batch["query"], padding=True, return_tensors="pt")["input_ids"].to(device)
            generation = generate_(model, input_ids, generation_config)
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
                generations.append(gather_object(generation))
            completions.extend(gather_object(parsed_generation_text))
            raw_text.extend(gather_object(raw_generation_text))
            break

    completions = completions[:len(dataset)]
    raw_text = raw_text[:len(dataset)]
    if generation_config.return_dict_in_generate:
        generations = generations[:len(dataset)]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = (
        Path(config.eval_results_save_path)
        / config.eval_dataset_name_or_path
        / f"{config.checkpoint_dir.split('/')[-1]}_{config.resume_from_checkpoint}_{timestamp}.json"
    )
    # Create directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # dataset = dataset.add_column("completion", completions)
    # dataset = dataset.add_column("raw_completion", raw_text)
    # dataset.to_json(save_path.with_suffix(".jsonl"))
    # if generation_config.return_dict_in_generate:
    #     with open(save_path, "w") as f:
    #         json.dump(generations, f, indent=4)
    with open(save_path, "w") as f:
        json.dump(raw_text, f, indent=4)

if __name__ == "__main__":
    main()
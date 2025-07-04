import logging
import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch.distributed as dist
import hydra
import copy
from omegaconf import DictConfig, OmegaConf
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding

# Import the initialize_model_class function from train.py
from torchprime.torch_xla_models.train import (
    initialize_model_class,
    set_default_dtype,
    Trainer,
)
from torchprime.torch_xla_models.inference_utils import (
    GenerationConfig,
    generate,
    GenerationConfig_,
    generate_,
)

# Initialize XLA runtime for TPU
xr.use_spmd()
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", init_method="xla://")


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return xr.process_index() == 0


logger = logging.getLogger(__name__)


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
    # Right pad input_ids to nearest multiple of 256
    seq_len = input_ids.shape[1]
    pad_len = (256 - seq_len % 256) % 256  # Calculate padding needed
    if pad_len > 0:
        pad_ids = torch.full(
            (input_ids.shape[0], pad_len), tokenizer.pad_token_id, dtype=input_ids.dtype
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

    logger.info("Preparing inputs...")
    # prompt = "Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who is the 47th president of the United States. A member of the Republican Party, he served as the 45th president from 2017 to 2021."
    # prompt = "<|im_start|>" + "<|mask|>"*255
    prompt = """#coding utf-8
'''
斐波那契数列-循环法
'''
def Fib_circle():
    while True:   # 去掉while循环，只用for循环
        num_1 = 0
        num_2 = 1
        fib_array = [0] # 用于存储计算出的FB数列值
        m = input('你想要查找的起始项：')
        n = input('你想要查找的结束项：')
        if m.isdigit() and n.isdigit():   # 在这个实现函数中，不要进行检验。每个函数只做一个事情
            m = int(m) # 将输入化为整数型
            n = int(n)
            for i in range(n):
                num_1, num_2 = num_2, num_1 + num_2
                fib_array.append(num_1)
            print(f'你要查找的数列为{list(enumerate(fib_array[m:], m))}')
            break
        else:
            print('请输入有效的正整数')

if __name__ == '__main__':
    Fib_circle()
"""
    # messages = [{"role": "user", "content": prompt}]
    messages = prompt
    # generation_config = GenerationConfig(**OmegaConf.to_container(config.generation))
    generation_config = GenerationConfig_(**OmegaConf.to_container(config.generation))

    ddlm_inputs, _ = prepare_inputs(
        tokenizer, messages, generation_config, enable_thinking=False, noise_ratio=0.1
    )
    generation_config.diffusion_steps = (ddlm_inputs["input_ids"] == tokenizer.mask_token_id).sum()
    print(f"setting diffusion_steps to number of mask tokens: {generation_config.diffusion_steps}")

    dataset = Dataset.from_list(
        [copy.deepcopy(ddlm_inputs) for _ in range(config.global_batch_size)]
    )  # Create a single-element dataset with ddlm_inputs

    trainer = Trainer(
        model=model, tokenizer=tokenizer, config=config, eval_dataset=dataset
    )
    trainer._load_checkpoint()
    logger.info(
        f"ckpt model weights: {trainer.model.state_dict()['model.embed_tokens.weight']}"
    )

    logger.info("Generating...")
    loader = trainer._get_eval_dataloader()
    iterator = iter(loader)
    try:
        batch = next(iterator)
        logger.info(f"batch: {batch}")
    except StopIteration:
        logger.info("No more batches, reset iterator")
        iterator = iter(loader)
        batch = next(iterator)

    # generation = generate(
    #     trainer.model, tokenizer, batch, generation_config, verbose=True
    # )
    generation = generate_(
        trainer.model, batch["input_ids"], generation_config
    )
    xm.wait_device_ops()
    if generation_config.return_dict_in_generate:
        completion = generation["completion"].cpu().tolist()
        history = generation["history"]
    else:
        completion = generation.cpu().tolist()
        history = None
    if is_main_process():
        if history is not None:
            for i in range(len(history)):
                print("=" * 50 + f"HISTORY at step {i}" + "=" * 50)
                for j in range(len(history[i])):
                    print(
                        f"Completion {j} at step {i}: {tokenizer.decode(history[i][j], skip_special_tokens=True)}"
                    )
                print("=" * 50)
        print("=" * 50 + "GENERATION" + "=" * 50)
        for i in range(len(completion)):
            print(
                f"Completion {i}: {tokenizer.decode(completion[i], skip_special_tokens=True)}"
            )
            print("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()

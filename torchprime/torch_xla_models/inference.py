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
from datasets.distributed import split_dataset_by_node
from torchprime.torch_xla_models.train import Trainer
from torchprime.torch_xla_models.inference_utils import (
    GenerationConfig,
    generate,
    GenerationConfig_,
    generate_,
    prepare_inputs,
)
from torchprime.torch_xla_models.model_utils import initialize_model_class, set_default_dtype

# Initialize XLA runtime for TPU
xr.use_spmd()
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", init_method="xla://")


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return xr.process_index() == 0


logger = logging.getLogger(__name__)



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


    logger.info("Preparing inputs...")
#     prompt = """#coding utf-8
# '''
# 斐波那契数列-循环法
# '''
# def Fib_circle():
#     while True:   # 去掉while循环，只用for循环
#         num_1 = 0
#         num_2 = 1
#         fib_array = [0] # 用于存储计算出的FB数列值
#         m = input('你想要查找的起始项：')
#         n = input('你想要查找的结束项：')
#         if m.isdigit() and n.isdigit():   # 在这个实现函数中，不要进行检验。每个函数只做一个事情
#             m = int(m) # 将输入化为整数型
#             n = int(n)
#             for i in range(n):
#                 num_1, num_2 = num_2, num_1 + num_2
#                 fib_array.append(num_1)
#             print(f'你要查找的数列为{list(enumerate(fib_array[m:], m))}')
#             break
#         else:
#             print('请输入有效的正整数')

# if __name__ == '__main__':
#     Fib_circle()
# """
    prompt = """#coding utf-8
'''
斐波那契数<|mask|>-循环法
'''
def Fib_circle<|mask|>    while True:   # 去掉while循环，只用for<|mask|>
<|mask|> num_1 = 0
        num_2 = 1
        fib<|mask|> = [0]<|mask|> 用于存储计算出的FB数<|mask|>值
        m = input('你想要查找的起<|mask|><|mask|>：')
       <|mask|> = input('你想要查找的结束项<|mask|>')
        if m.isdigit() and n.isdigit():   #<|mask|>这个实现<|mask|>中，不要进行检验。每个函数只做一个事情
           <|mask|> = int(m)<|mask|> 将输入<|mask|>为整数型
            n = int(n)
            for i in range(n):
                num_1,<|mask|>_2 = num_2, num_1 + num_<|mask|>
                fib_array.append(num_<|mask|>)
            print(f'你要查找的数列为{list(enumerate(fib_array[m<|mask|><|mask|>))}')
            break
        else:
            print('请输入有效的正整数')

if __name__ ==<|mask|><|mask|>__':
    Fib_circle()
"""
    # generation_config = GenerationConfig(**OmegaConf.to_container(config.generation))
    generation_config = GenerationConfig_(**OmegaConf.to_container(config.generation))

    ddlm_inputs, _ = prepare_inputs(tokenizer, prompt, generation_config)
    generation_config.diffusion_steps = (ddlm_inputs["input_ids"] == tokenizer.mask_token_id).sum()
    print(f"setting diffusion_steps to number of mask tokens: {generation_config.diffusion_steps}")

    dataset = Dataset.from_list(
        [ddlm_inputs for _ in range(config.global_batch_size)]
    )  # Create a single-element dataset with ddlm_inputs
    dataset = split_dataset_by_node(dataset, xr.process_index(), xr.process_count())

    trainer = Trainer(
        model=model, tokenizer=tokenizer, config=config, eval_dataset=dataset
    )
    trainer._load_checkpoint()

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
        trainer.model, batch["input_ids"], generation_config, output_hidden_states=True
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

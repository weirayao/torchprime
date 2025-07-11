import os
import importlib

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
import sys

# Configure logging to show INFO messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
import torch
import torch.distributed as dist
import hydra
import copy
import shutil
from pathlib import Path
from time import time
from omegaconf import DictConfig, OmegaConf
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding
from datasets.distributed import split_dataset_by_node

from torchprime.torch_xla_models.inference_utils import (
    GenerationConfig,
    generate,
    GenerationConfig_,
    generate_,
    prepare_inputs,
)
from torchprime.torch_xla_models.model_utils import (
    set_default_dtype,
    load_hf_model,
    load_safetensors_to_state_dict,
)
from torchprime.torch_xla_models.flex.modeling_qwen import Qwen3ForCausalLM
from gpu_utils import download_gcs_checkpoint


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    gcs_bucket = "gs://sfr-text-diffusion-model-research/"
    local_checkpoint_path = "/export/agentstudio-family-2/haolin/"
    checkpoint = (
        "consolidated_checkpoints/flex_processed_v1_qw1_7b_512_split_datafix/16000/"
    )

    logger.info("Download the checkpoint from GCS to local path if needed")
    checkpoint_dir = download_gcs_checkpoint(
        gcs_bucket, checkpoint, local_checkpoint_path
    )

    checkpoint_dir = os.path.join(local_checkpoint_path, checkpoint)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Load the model configuration
    config_path = "torchprime/torch_xla_models/configs/model/flex-qwen-1b.yaml"
    model_config = OmegaConf.load(config_path)

    logger.info("Model config loaded:")
    logger.info(OmegaConf.to_yaml(model_config))
    # Initialize the model class
    logger.info("\nInitializing model...")
    with set_default_dtype(torch.bfloat16):
        # For GPU, we don't need torch_xla.device() context
        model = Qwen3ForCausalLM(model_config)
        state_dict = load_safetensors_to_state_dict(checkpoint_dir)
        model.load_state_dict(state_dict)
        model = model.cuda()  # Move to GPU

    logger.info(f"Model initialized successfully!")
    logger.info(f"Model class: {model.__class__.__name__}")
    logger.info(f"Model device: {next(model.parameters()).device}")
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.mask_token is None:
        tokenizer.add_tokens("<|mask|>", special_tokens=True)
        tokenizer.add_special_tokens(
            {"mask_token": "<|mask|>"}, replace_additional_special_tokens=False
        )

    logger.info(f"Tokenizer initialized: {tokenizer.__class__.__name__}")
    logger.info(f"Mask token ID: {tokenizer.mask_token_id}")
    logger.info(f"Vocab size: {len(tokenizer)}")

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

    generation_config = GenerationConfig_(
        diffusion_steps=23,
        max_tokens=0,
        temperature=0.2,
        top_p=None,
        top_k=10000,
        alg="neg_entropy",
        alg_temp=0.2,
        output_history=True,
        return_dict_in_generate=True,
    )

    inputs, _ = prepare_inputs(tokenizer, prompt, generation_config)
    input_ids = inputs["input_ids"].unsqueeze(0)

    generation_time = time()
    generation = generate_(
        model=model,
        input_ids=input_ids,
        generation_config=generation_config,
        output_hidden_states=True,
    )
    generation_time = time() - generation_time
    logger.info(f"Generation time: {generation_time:.2f} seconds")

    competions = tokenizer.batch_decode(generation["completion"])
    for i, completion in enumerate(competions):
        logger.info(f"Competion {i}: {completion}")

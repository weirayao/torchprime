import importlib
import logging
import math
import os
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from timeit import default_timer as timer
from dotenv import load_dotenv
load_dotenv()

import datasets
import hydra
import torch
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import transformers
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from datasets.distributed import split_dataset_by_node
from torch_xla._internal.jax_workarounds import jax_env_context
from torch_xla.distributed.fsdp import checkpoint_module
from torch_xla.distributed.spmd.xla_sharding import apply_xla_patch_to_nn_linear
from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding
from transformers import (
  AutoTokenizer,
  default_data_collator,
  get_scheduler,
  set_seed,
)
from torch_xla.experimental.distributed_checkpoint import CheckpointManager, prime_optimizer
from transformers.optimization import Adafactor
from transformers.trainer_pt_utils import get_module_class_from_name
from transformers.utils import check_min_version
from transformers import PreTrainedTokenizerBase

from torchprime.data.dataset import make_huggingface_dataset, make_gcs_dataset, make_gcs_pretokenized_dataset
from torchprime.layers.sequential import HomogeneousSequential
from torchprime.metrics.metrics import MetricsLogger
from torchprime.metrics.mfu import compute_mfu
from torchprime.metrics.step_duration import step_duration_from_latest_profile
from torchprime.sharding.shard_model import (
  shard_torch_xla_model_from_config,
  wrap_module,
)
from torchprime.torch_xla_models import offloading, remat_all, scan_layers
from torchprime.torch_xla_models.topology import (
  get_mesh,
  get_num_slices,
  is_1d_sharding,
)
from torchprime.utils.retry import retry

check_min_version("4.39.3")
logger = logging.getLogger(__name__)

xr.use_spmd()
assert xr.is_spmd() is True

if not dist.is_initialized():
  dist.init_process_group(backend='gloo', init_method='xla://')

def is_main_process():
  """Check if this is the main process (rank 0)."""
  return xr.process_index() == 0

# Map torchprime model classes to their corresponding HuggingFace model classes
HF_MODEL_CLASS_MAPPING = {
  "llama.LlamaForCausalLM": "LlamaForCausalLM",
  "flex.LlamaForCausalLM": "LlamaForCausalLM", 
  "flex.Qwen3ForCausalLM": "Qwen3ForCausalLM",
  "mixtral.MixtralForCausalLM": "MixtralForCausalLM",
  "llama4.Llama4TextForCausalLM": "Llama4ForCausalLM",
  "qwen.Qwen3ForCausalLM": "Qwen3ForCausalLM",
}

MOUNTED_GCS_DIR = os.environ.get("MOUNTED_GCS_DIR", None)

class Trainer:
  """The trainer."""

  minibatch: bool

  def __init__(
    self,
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    config: DictConfig,
    train_dataset: Dataset | IterableDataset | None = None,
    eval_dataset: Dataset | IterableDataset | None = None,
  ):
    self.config = config
    self.device = torch_xla.device()
    self.global_batch_size = self.config.global_batch_size
    self.train_dataset = train_dataset
    self.eval_dataset = eval_dataset
    self.tokenizer = tokenizer

    # Set up SPMD mesh and shard the model
    mesh = get_mesh(self.config)
    xs.set_global_mesh(mesh)
    logger.info(f"Logical mesh shape: {mesh.shape()}")
    logger.info(f"Logical mesh device assignments: {mesh.device_ids}")

    # TODO(https://github.com/pytorch/xla/issues/8696): Minibatch only works in 1D sharding.
    minibatch = is_1d_sharding(tuple(config.ici_mesh.values()))
    self.minibatch = minibatch
    logger.info(f"Minibatch dataloading: {minibatch}")

    # TODO(https://github.com/AI-Hypercomputer/torchprime/issues/66): Test this for multislice
    self.input_sharding_spec = xs.ShardingSpec(
      mesh, (("data", "fsdp"), None), minibatch=minibatch
    )

    # Recursively replace `nn.Linear` layers with einsum operations in the model.
    # Without this patch, an `nn.Linear` module will flatten non-contracting dimensions
    # (e.g. batch and sequence), thus destroying the sharding constraints on those dimensions.
    model = apply_xla_patch_to_nn_linear(model)

    # Annotate model weights and activations with sharding constraints to distribute
    # the training across devices following the SPMD paradigm.
    sharding_config = OmegaConf.to_container(self.config.model.sharding, resolve=True)
    assert isinstance(sharding_config, dict), (
      f"Sharding config {sharding_config} must be a dict"
    )
    model = shard_torch_xla_model_from_config(model, config=sharding_config)

    # Rematerialize forward computation during the backward pass if requested.
    model = self._add_checkpoint_offload_scan_model(model)
    model = self._add_optimization_barrier_model(model)
    self.model = model
    self.hf_model = load_hf_model(self.config.model)

    # Set up optimizers
    self.optimizer = Adafactor(
      params=model.parameters(),
      lr=self.config.optimizer.learning_rate,
      relative_step=False,
      scale_parameter=False,
    )

    # TODO: this OOMs the TPU.
    # self._prime_optimizer()

    self.lr_scheduler = get_scheduler(
      name=self.config.lr_scheduler.type,
      optimizer=self.optimizer,
      num_warmup_steps=self.config.lr_scheduler.warmup_steps,
      num_training_steps=self.config.max_steps,
    )

    # Initialize checkpoint manager
    # Use GCS for checkpoints with proper path handling
    self.ckpt_dir = config.checkpoint_dir
    self.ckpt_mgr = CheckpointManager(path=self.ckpt_dir, save_interval=config.save_steps)
    self.start_step = 0

    # Execute all initialization work queued so far before starting training.
    torch_xla.sync()

  def _prime_optimizer(self):
    for group in self.optimizer.param_groups:
      for p in group["params"]:
        p.grad = torch.zeros_like(p)
        p.grad.requires_grad_(False)
    self.optimizer.step()
    torch_xla.sync()

  def _load_checkpoint(self):
    """Load optimizer, scheduler, and training state from checkpoint."""
    tracked_steps = self.ckpt_mgr.all_steps()
    if not tracked_steps:
      logger.warning("No checkpoint steps found. Starting from scratch.")
      return
    self.optimizer = prime_optimizer(self.optimizer) # NOTE: needed to create the dummy state dict for the optimizer
    state_dict = {
      "model": self.model.state_dict(),
      "optimizer": self.optimizer.state_dict(),
      "scheduler": self.lr_scheduler.state_dict(),
      "step": self.start_step,
    }
    if self.config.resume_from_checkpoint in tracked_steps:
      logger.info(f"Loading checkpoint from step {self.config.resume_from_checkpoint}")
      self.ckpt_mgr.restore(self.config.resume_from_checkpoint, state_dict)
    elif self.config.resume_from_checkpoint == "latest":
      last_step = max(tracked_steps)
      logger.warning(f"Checkpoint step {self.config.resume_from_checkpoint} not found in tracked steps {tracked_steps}. Loading from latest checkpoint {last_step}.")
      self.ckpt_mgr.restore(last_step, state_dict)
    else:
      raise ValueError(f"Invalid checkpoint step: {self.config.resume_from_checkpoint}. Must be one of {tracked_steps} or 'latest'.")

    self.model.load_state_dict(state_dict["model"])
    self.optimizer.load_state_dict(state_dict["optimizer"])
    self.lr_scheduler.load_state_dict(state_dict["scheduler"])
    self.start_step = state_dict["step"]

  def _instrument_dataloader(self, dataloader, name): 
    """Instrument the shapes from the dataloader with a name for logging.""" 
  
    class WrapperDataLoader: 
      def __init__(self, dataloader, name): 
        self.dataloader = dataloader 
        self.name = name 
        self.step = 0
    
      def __iter__(self): 
        for batch in self.dataloader: 
          self.step += 1
          # Log the shapes of the inputs and targets. 
          if self.step % 100 == 0:
            self._log_shapes(batch) 
          yield batch 
    
      def _log_shapes(self, batch): 
        import torch.utils._pytree as pytree 
    
        shapes = pytree.tree_map( 
          lambda x: x.shape if isinstance(x, torch.Tensor) else None, batch 
        ) 
        logger.info(f"[{self.name}] step: {self.step}, data shapes: {shapes}, data: {batch}") 
    
      #   # Visualize one example tensor. 
      #   t = next(iter(pytree.tree_iter(batch))) 
      #   if t.device.type == "xla": 
      #     import click 
      #     from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding 
    
      #     generated_table = visualize_tensor_sharding(t, use_color=False) 
      #     click.echo(generated_table) 
    
      # def __len__(self): 
      #   return len(self.dataloader) 
  
    return WrapperDataLoader(dataloader, name) 

  def _get_train_dataloader(self):
    if self.train_dataset is None:
      raise ValueError("Trainer: training requires a train_dataset.")

    num_replicas = xr.process_count()
    logger.info(f"Num replicas: {num_replicas}")
    if isinstance(self.train_dataset, IterableDataset):
      # For IterableDataset, don't use DistributedSampler as it doesn't have len()
      sampler = None
      logger.info("Using IterableDataset without DistributedSampler")
    else:
      # For regular Dataset, use DistributedSampler
      if self.minibatch:
        sampler = torch.utils.data.DistributedSampler(
          self.train_dataset,
          num_replicas=num_replicas,
          rank=xr.process_index(),
        )
      else:
        # Without minibatch, every process loads the global batch the same way.
        sampler = torch.utils.data.DistributedSampler(
          self.train_dataset,
          num_replicas=1,
          rank=0,
        )
      
    assert self.global_batch_size is not None
    if self.minibatch:
      # Each process loads the per-host batch size.
      batch_size = self.global_batch_size // num_replicas
    else:
      # Each process will load the global batch, then discard the unneeded parts.
      # For IterableDataset, use global batch size as distributed sampling is handled upstream
      batch_size = self.global_batch_size
    dataloader = DataLoader(
      self.train_dataset,
      # Data collator will default to DataCollatorWithPadding, so we change it.
      collate_fn=default_data_collator,
      batch_size=batch_size,
      sampler=sampler,
      drop_last=True,
    )
    print(f"Global batch size: {self.global_batch_size}, per-process batch size: {batch_size}, device: {self.device}, process: {xr.process_index()}")
    dataloader = self._instrument_dataloader(dataloader, f"base_{xr.process_index()}") 
    loader = pl.MpDeviceLoader(
      dataloader, self.device, input_sharding=self.input_sharding_spec
    )
    loader = self._instrument_dataloader(loader, f"mp_device_loader") 
    return loader

  def _add_checkpoint_offload_scan_model(self, model: nn.Module):
    remat_classes = self._get_classes_by_names(
      model, self.config.model.remat.get("activation_checkpoint_layers", [])
    )
    layers_to_scan = self.config.model.remat.get("scan_layers", None)
    offload_tensors = self.config.model.remat.get("offload_tensors", [])

    # Checking preconditions and logging.
    if remat_classes:
      logger.info(f"Enabling activation checkpointing on {remat_classes}")
    if layers_to_scan:
      assert isinstance(layers_to_scan, str)
      logger.info(f"Compiling module `{layers_to_scan}` with scan")
    if len(offload_tensors):
      logger.info(f"Will offload these tensors to host RAM: {offload_tensors}")
      if layers_to_scan is None:
        raise NotImplementedError("Host offloading requires scan")
      if len(remat_classes) != 1:
        raise NotImplementedError(
          "Host offloading requires checkpointing exactly one layer"
        )

    def maybe_checkpoint(mod, _name):
      if isinstance(mod, tuple(remat_classes)):
        return checkpoint_module(mod)
      return mod

    if layers_to_scan is None:
      # Implement activation checkpointing without scan by wrapping modules.
      if not remat_classes:
        return model
      return wrap_module(model, maybe_checkpoint)

    if not remat_classes:
      # Scan without activation checkpointing.
      return scan_layers.compile(model, layers_to_scan)

    # Implement activation checkpointing and host offloading under scan via
    # a graph partitioner instead of `checkpoint_module`.
    seq = model.get_submodule(layers_to_scan)
    assert isinstance(seq, HomogeneousSequential)
    if len(remat_classes) != 1 or list(remat_classes)[0] != seq.repeated_layer:
      raise NotImplementedError(
        f"When compiling decoder layers with scan and \
          activation checkpointing is also requested, we only support \
          checkpointing {seq.repeated_layer} i.e. the layer being scanned."
      )
    if not len(offload_tensors):
      partition_fn = remat_all.remat_all_partition_fn
    else:
      partition_fn = partial(
        offloading.remat_all_and_offload_these_inputs,
        names_to_offload=offload_tensors,
      )
    return scan_layers.compile(model, layers_to_scan, partition_fn=partition_fn)

  def _add_optimization_barrier_model(self, model: nn.Module):
    classes = self._get_classes_by_names(
      model, self.config.model.remat.get("optimization_barrier_layers", [])
    )
    if not classes:
      return model

    logger.info(f"Adding backward optimization barriers to {classes}")

    def maybe_add_barrier(mod, _name):
      if isinstance(mod, tuple(classes)):
        # Register a backward hook to place optimization barrier to prevent
        # gigantic fusions on syncing the gradients.
        xs.apply_backward_optimization_barrier(mod)
        return mod
      return mod

    return wrap_module(model, maybe_add_barrier)

  def _get_classes_by_names(self, model, activation_checkpoint_layers: list[str]):
    classes_to_checkpoint = set()
    for layer_class in activation_checkpoint_layers:
      cls = get_module_class_from_name(model, layer_class)
      if cls is None:
        raise Exception(
          f"Could not find the transformer layer class {layer_class} in the model."
        )
      else:
        classes_to_checkpoint.add(cls)
    return tuple(classes_to_checkpoint)

  def data_loop(self):
    # For now we assume that we wil never train for mor than one epoch
    train_loader = self._get_train_dataloader()
    train_iterator = iter(train_loader)
    max_step = self.config.max_steps
    epoch = 0
    token_count = 0
    for step in range(max_step):
      try:
        batch = next(train_iterator)
        token_count += batch["input_ids"].shape[0] * self.config.data.block_size
      except StopIteration:
        logger.warning(f"DataLoader exhausted at step {step}, reset iterator")
        logger.info(f"Token count in an epoch: {token_count} ({token_count/1e9:.3f}B)")
        epoch += 1
        break
      # visualize_tensor_sharding(batch['input_ids'], use_color=False)
      if step % 100 == 0:
        logger.info(f"Step {step}, Token count: {token_count}, Device: {xr.process_index()}")
      assert batch["input_ids"].shape == (self.global_batch_size, self.config.data.block_size), f"Batch shape mismatch: {batch['input_ids'].shape} != {(self.global_batch_size, self.config.data.block_size)}"

  def train_loop(self):
    if self.config.resume_from_checkpoint is not None:
      self._load_checkpoint()
    self.model.train()
    self.model.zero_grad()

    # For now we assume that we wil never train for mor than one epoch
    max_step = self.config.max_steps
    train_loader = self._get_train_dataloader()
    train_iterator = iter(train_loader)

    metrics_logger = MetricsLogger(self.config.model)
    logger.info("Starting training")
    logger.info(f"    Max step: {max_step}")
    logger.info(f"    Global batch size: {self.global_batch_size}")
    if hasattr(self, 'start_step') and self.start_step > 0:
      logger.info(f"    Resuming from step: {self.start_step}")
    if is_main_process():
      wandb.login(key=os.environ.get("WANDB_API_KEY"), host="https://salesforceairesearch.wandb.io")
      wandb.init(project="text-diffusion-model-research", name=self.config.model.model_class)
      # Log the configuration to wandb
      wandb.config.update(OmegaConf.to_container(self.config, resolve=True))
      # Set wandb step to start_step if resuming from checkpoint
      if self.start_step > 0:
        wandb.log({}, step=self.start_step-1)  # Set the initial step counter

    # Initialize epoch and step counters, accounting for checkpoint loading
    epoch = 0
    start_step = self.start_step

    # Skip batches if we're resuming from a checkpoint
    if start_step > 0:
      logger.info(f"Skipping {start_step} batches to resume from checkpoint...")
      for _ in range(start_step):
        try:
          next(train_iterator)
        except StopIteration:
          epoch += 1
          train_iterator = iter(train_loader)
          next(train_iterator)

    for step in range(start_step, max_step):
      try:
        batch = next(train_iterator)
      except StopIteration:
        logger.warning(f"DataLoader exhausted at step {step}, reset iterator")
        epoch += 1
        train_iterator = iter(train_loader)
        batch = next(train_iterator)

      trace_start_time = timer()
      loss = self.train_step(batch)
      trace_end_time = timer()

      if step % self.config.logging_steps == 0:

        def step_closure(epoch, step, loss, trace_start_time, trace_end_time):
          loss = loss.detach().item()
          logger.info(
            f"Epoch: {epoch}, step: {step}, loss: {loss:0.4f}, "
            f"trace time: {(trace_end_time - trace_start_time) * 1000:0.2f} ms"
          )
          if math.isnan(loss):
            raise ValueError(f"Loss is NaN at step {step}")
          if is_main_process():
            wandb.log(
              {
                "train/loss": loss,
                "train/ppl": math.exp(loss),
                "train/step_time": (trace_end_time - trace_start_time) * 1000,
                "train/epoch": epoch,
                "train/step": step,
                "train/lr": self.lr_scheduler.get_last_lr()[0],
                "train/total_tokens": self.config.data.block_size * (step + 1) * self.global_batch_size,
              },
              step=step  # Explicitly set the wandb global step
            )

        xm.add_step_closure(
          step_closure,
          args=(epoch, step, loss, trace_start_time, trace_end_time),
          run_async=True,
        )

      if step > self.start_step and step % self.config.save_steps == 0:
        # NOTE: currently we save the checkpoint synchronously
        xm.wait_device_ops()  # Wait for all XLA operations to complete
        state_dict = {
          "model": self.model.state_dict(),
          "optimizer": self.optimizer.state_dict(),
          "scheduler": self.lr_scheduler.state_dict(),
          "step": step,
        }
        try:
          self.ckpt_mgr.save(step, state_dict, force=True)
          logger.info(f"Checkpoint saved at step {step} to {self.ckpt_dir}")
        except Exception as e:
          logger.error(f"Failed to save checkpoint at step with ckpt_mgr {step}: {e}")
        xm.wait_device_ops()  # Ensure save is complete before logging

      # Capture profile at the prefer step
      if step == self.config.profile_step:
        # Wait until device execution catches up to tracing before triggering the profile. This will
        # interrupt training slightly on the hosts which are capturing, but by waiting after tracing
        # for the step, the interruption will be minimal.
        xm.wait_device_ops()
        xp.trace_detached(
          "127.0.0.1:9012",
          self.config.profile_dir,
          self.config.profile_duration,
        )

    xm.wait_device_ops()
    logger.info("Finished training run")

    if self.config.profile_step >= 0:
      # Analyze the step duration from the latest profile
      step_duration = step_duration_from_latest_profile(self.config.profile_dir)
      metrics_logger.log_step_execution_time(step_duration)

      tpu_name = os.environ.get("TORCHPRIME_TPU_TYPE", None)
      if tpu_name:
        # Add "torch_dtype" in model config
        model_config_for_mfu = OmegaConf.to_container(self.config.model, resolve=True)
        model_config_for_mfu["torch_dtype"] = str(
          get_model_dtype(self.model)
        ).removeprefix("torch.")

        # Compute MFU
        mfu = compute_mfu(
          config=model_config_for_mfu,
          batch_size=self.config.global_batch_size,
          step_duration=step_duration,
          tpu_name=tpu_name,
          num_slices=get_num_slices(),
          sequence_length=self.config.block_size,
        )
        metrics_logger.log_mfu(mfu.mfu)

    # Print and save metrics
    metrics = metrics_logger.finalize()
    logger.info("***** train metrics *****\n%s", metrics)
    metrics.save(Path(self.config.output_dir) / "train_metrics.json")

  @torch_xla.compile(full_graph=True)
  def train_step(self, batch):
    _logits, loss = self.model(**batch)
    loss.backward()
    self.optimizer.step()
    self.lr_scheduler.step()
    self.model.zero_grad()
    return loss

def load_hf_model(model_config):
  logger.info(f"Loading HuggingFace model from {model_config.tokenizer_name}")
  hf_model_class_name = HF_MODEL_CLASS_MAPPING.get(model_config.model_class)
  if hf_model_class_name is None:
    print(f"Error: No HuggingFace model mapping found for '{model_config.model_class}'")
    print(f"Available mappings: {list(HF_MODEL_CLASS_MAPPING.keys())}")
    sys.exit(1)

  # Dynamically import the HuggingFace model class
  try:
    transformers_module = importlib.import_module("transformers")
    hf_model_class = getattr(transformers_module, hf_model_class_name)
  except (ModuleNotFoundError, AttributeError) as e:
    print(f"Error importing HuggingFace model class '{hf_model_class_name}': {e}")
    sys.exit(1)

  hf_model = hf_model_class.from_pretrained(
    model_config.tokenizer_name,
    torch_dtype=torch.bfloat16,
  )
  return hf_model

def initialize_model_class(model_config, load_from_hf=True):
  """Import and initialize model_class specified by the config."""
  module_name, model_class_name = model_config.model_class.rsplit(".", 1)
  try:
    module = importlib.import_module(module_name)
  except ModuleNotFoundError as e:
    print(f"Error importing relative module: {e}")
    sys.exit(1)
  if hasattr(module, model_class_name):
    model_class = getattr(module, model_class_name)
  else:
    print(f"Error: Function '{model_class_name}' not found in module '{module_name}'")
    sys.exit(1)
  model = model_class(model_config)
  # Load pretrained weights from HuggingFace model
  if load_from_hf:
    hf_model = load_hf_model(model_config)
    logger.info("Loaded model from HuggingFace. Now loading state dict.")
    model.load_state_dict(hf_model.state_dict())
    del hf_model
  return model


@contextmanager
def set_default_dtype(dtype):
  # Get the current default dtype
  previous_dtype = torch.get_default_dtype()
  # Set the new default dtype
  torch.set_default_dtype(dtype)
  try:
    yield
  finally:
    # Revert to the original default dtype
    torch.set_default_dtype(previous_dtype)


def get_model_dtype(module):
  dtypes = {param.dtype for param in module.parameters()}
  if len(dtypes) != 1:
    raise ValueError(f"Inconsistent dtypes found: {dtypes}")
  return dtypes.pop()


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: DictConfig):  
  # Configure logging (only on main process to avoid duplicate logs)
  if is_main_process():
    print(OmegaConf.to_yaml(config))  # Print the config for debugging
  
  log_level = logging.INFO
  logger.setLevel(log_level)
  logger.setLevel(log_level)
  datasets.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.enable_default_handler()
  transformers.utils.logging.enable_explicit_format()

  # Initialize distributed process group for XLA
  # torch.distributed.init_process_group('gloo', init_method='xla://')

  set_seed(config.seed)
  torch_xla.manual_seed(config.seed)
  
  # Start profiling server (only on main process)
  server = xp.start_server(9012)
  logger.info(f"Profiling server started: {str(server)}")

  # TODO(https://github.com/AI-Hypercomputer/torchprime/issues/14): Add tokenizers to torchprime.
  tokenizer_name = config.model.tokenizer_name
  tokenizer = retry(lambda: AutoTokenizer.from_pretrained(tokenizer_name))

  # Set the model dtype to bfloat16, and set the default device to the XLA device.
  # This will capture the model constructor into a graph so that we can add
  # sharding annotations to the weights later, and run the constructor on the XLA device.
  # NOTE: read HF model from GCS bucket if resume_from_checkpoint is not provided, otherwise read from checkpoint_dir in _load_checkpoint()
  load_from_checkpoint = hasattr(config, 'resume_from_checkpoint')
  with set_default_dtype(torch.bfloat16), torch_xla.device():
    model = initialize_model_class(config.model, load_from_hf=not load_from_checkpoint)

  n_params = sum([p.numel() for p in model.parameters()])
  if is_main_process():
    if load_from_checkpoint:
      logger.info(f"Continuing training on previous model checkpoint - Total size={n_params} params")
    else:
      logger.info(f"Training from scratch on pretrained model - Total size={n_params} params")

  if config.data.dataset_name:
    # Downloading and loading a dataset from the hub.
    dataset_name = config.data.dataset_name
    gcs_prefix = "gs://sfr-text-diffusion-model-research/"
    if dataset_name.startswith(gcs_prefix):
      dataset_name = os.path.join(MOUNTED_GCS_DIR, dataset_name.split(gcs_prefix)[1])
      logger.info(f"Loading dataset from {dataset_name}")
      data = retry(
        lambda: make_gcs_pretokenized_dataset(dataset_name, seed=config.seed)
      )
    else:
      data = retry(
        lambda: make_huggingface_dataset(
          name=config.data.dataset_name,
          config_name=config.data.dataset_config_name,
          split="train",
          cache_dir=config.data.cache_dir,
          tokenizer=tokenizer,
          block_size=config.data.block_size,
        )
      )
  elif config.data.gcs_dataset_names:
    # Downloading and loading a dataset from GCS bucket.
    data = retry(
      lambda: make_gcs_dataset(
        names=config.data.gcs_dataset_names,
        weights=config.data.weights,
        tokenizer=tokenizer,
        seed=config.seed,
        block_size=config.data.block_size,
      )
    )
  
  else:
    raise ValueError("No dataset provided")
  
  # Add debugging for SPMD data loading
  if is_main_process():
    logger.info(f"Dataset type: {type(data)}")
    logger.info(f"SPMD mode enabled: {xr.is_spmd()}")
    logger.info(f"Number of devices: {xr.process_count()}")
    logger.info(f"Current device rank: {xr.process_index()}")
    
    if isinstance(data, IterableDataset):
      logger.info("Using IterableDataset - need to ensure proper data distribution")
      logger.info("In SPMD mode, each device should get a different portion of the data")

  # For IterableDataset in SPMD mode, apply dataset splitting
  if isinstance(data, IterableDataset):
    try:
      logger.info(f"Applying split_dataset_by_node for device {xr.process_index()}/{xr.process_count()}")
      data = split_dataset_by_node(data, xr.process_index(), xr.process_count())
      logger.info(f"Dataset split successful for device {xr.process_index()}")
    except Exception as e:
      logger.warning(f"Dataset splitting failed: {e}. This may cause data duplication across devices.")

  trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
    train_dataset=data,
  )

  # Synchronize all processes before starting training
  xm.wait_device_ops()  # Wait for all XLA operations to complete
  if is_main_process():
    logger.info("All processes synchronized, starting training")

  # TODO(https://github.com/pytorch/xla/issues/8954): Remove `jax_env_context`.
  with jax_env_context():
    trainer.data_loop()


if __name__ == "__main__":
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
  )
  main()

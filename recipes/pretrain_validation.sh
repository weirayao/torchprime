export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=98304 --xla_enable_async_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export TPU_PREMAPPED_BUFFER_SIZE=40000000000
export WDS_GOPEN="fsspec" # gcsfs
export GCSFS_DEFAULT_BLOCK_SIZE=$((3210241024))   # try 32 MiB; use 16 MiB if memory is tight
export GCSFS_DEFAULT_RETRIES=12
export GCSFS_DEFAULT_CACHE_TYPE=readahead
export GCSFS_DEFAULT_FILL_CACHE=false

# export PT_XLA_DEBUG_LEVEL=2
# export HYDRA_FULL_ERROR=1
python torchprime/torch_xla_models/pretrain_validation.py \
    run_name=pretrain_validation \
    training_mode=pretrain \
    reshape_context=false \
    seg_attn=false \
    data=validation_v2 \
    model=flex-qwen2-1b \
    global_batch_size=512 \
    max_steps=10 \
    checkpoint_load_dir=gs://sfr-text-diffusion-model-research/checkpoints/pretrain_qwen25_coder_1b_flex_v2_context_8192_full_attn_bsz_512_lr_3e-4/ \
    checkpoint_load_step=5 \
    resume_from_checkpoint=false \
    logging_steps=1 \
    ici_mesh.fsdp=32 \
    ici_mesh.tensor=2 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen2-scan
# fsdp * tensor * data * expert == num_devices

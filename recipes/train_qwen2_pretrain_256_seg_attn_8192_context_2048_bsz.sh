export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=98304 --xla_enable_async_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export HYDRA_FULL_ERROR=1
export TPU_PREMAPPED_BUFFER_SIZE=40000000000
export WDS_GOPEN="fsspec" # gcsfs
export GCSFS_DEFAULT_BLOCK_SIZE=$((3210241024))   # try 32 MiB; use 16 MiB if memory is tight
export GCSFS_DEFAULT_RETRIES=12
export GCSFS_DEFAULT_CACHE_TYPE=readahead
export GCSFS_DEFAULT_FILL_CACHE=false

# export PT_XLA_DEBUG_LEVEL=2
# export HYDRA_FULL_ERROR=1
python torchprime/torch_xla_models/train.py \
    run_name=reshape_tpu256_mask0_01_context_8192_segment_attn_large_batch_2048_lr_3e-4 \
    training_mode=pretrain \
    reshape_context=false \
    seg_attn=true \
    data=flex_v2_webdataset \
    model=flex-qwen2-1b \
    model.block_masking_probability=0.01 \
    model.mask_block_sizes=[[2,4,8],[4,8,16],[8,16,32],[16,32,64]] \
    model.truncate_probability=0.01 \
    model.prefix_probability=0.01 \
    model.masking_scheduler.schedule_type=linear \
    model.masking_scheduler.max_schedule_steps=10000 \
    optimizer.learning_rate=3e-4 \
    lr_scheduler.type=cosine \
    lr_scheduler.warmup_steps=2000 \
    global_batch_size=2048 \
    max_steps=55000 \
    checkpoint_load_dir=null \
    checkpoint_load_step=null \
    resume_from_checkpoint=false \
    checkpoint_save_dir=gs://sfr-text-diffusion-model-research/checkpoints/pretrain_qwen25_coder_1b_flex_v2_mask0_01_256_context_8192_seg_attn_bsz_2048_lr_3e-4/ \
    save_steps=2500 \
    logging_steps=1 \
    ici_mesh.fsdp=64 \
    ici_mesh.tensor=2 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen2-scan
# fsdp * tensor * data * expert == num_devices

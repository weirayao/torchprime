export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=98304 --xla_enable_async_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export TPU_PREMAPPED_BUFFER_SIZE=40000000000
export GCSFS_DEFAULT_BLOCK_SIZE=$((3210241024))   # try 32 MiB; use 16 MiB if memory is tight
export GCSFS_DEFAULT_RETRIES=12
export GCSFS_DEFAULT_CACHE_TYPE=readahead
export GCSFS_DEFAULT_FILL_CACHE=false

# export PT_XLA_DEBUG_LEVEL=2
# export HYDRA_FULL_ERROR=1
python torchprime/torch_xla_models/train.py \
    run_name=sft_qwen25_coder_1b_opc_stage1_context_1024_bsz_512_lr_2e-5 \
    training_mode=sft \
    progress_src_mask=true \
    progress_src_mask_ratio=0.1 \
    seg_attn=true \
    data=sft_stage1 \
    model=flex-qwen2-1b \
    model.block_masking_probability=0.1 \
    model.mask_block_sizes=[[2,4,8],[4,8,16],[8,16,32],[16,32,64]] \
    model.truncate_probability=0 \
    model.prefix_probability=0 \
    model.masking_scheduler.schedule_type=constant \
    model.masking_scheduler.max_schedule_steps=null \
    optimizer.learning_rate=2e-5 \
    lr_scheduler.type=cosine \
    lr_scheduler.warmup_steps=1000 \
    global_batch_size=512 \
    max_steps=75625 \
    checkpoint_load_dir=gs://sfr-text-diffusion-model-research/checkpoints/pretrain_qwen25_coder_1b_flex_v2_mask0_01_256_context_8192_seg_attn_bsz_2048_lr_3e-4/ \
    checkpoint_load_step=15000 \
    resume_from_checkpoint=false \
    checkpoint_save_dir=gs://sfr-text-diffusion-model-research/checkpoints/sft_qwen25_coder_1b_opc_stage1_context_1024_bsz_512_lr_2e-5/ \
    save_steps=5000 \
    logging_steps=1 \
    ici_mesh.fsdp=32 \
    ici_mesh.tensor=2 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen2-scan
# fsdp * tensor * data * expert == num_devices

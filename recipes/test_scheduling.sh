export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=98304 --xla_enable_async_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
# export PT_XLA_DEBUG_LEVEL=2
# export HYDRA_FULL_ERROR=1
python torchprime/torch_xla_models/train.py \
    training_mode=pretrain \
    data=flex_v2 \
    model=flex-qwen2-1b \
    model.block_masking_probability=0.05 \
    model.mask_block_sizes=[[2,4,8],[4,8,16],[8,16,32],[16,32,64]] \
    model.truncate_probability=0.05 \
    model.prefix_probability=0.05 \
    model.masking_scheduler.schedule_type=linear \
    model.masking_scheduler.max_schedule_steps=11000 \
    optimizer.learning_rate=4e-4 \
    lr_scheduler.warmup_steps=60 \
    global_batch_size=8192 \
    max_steps=38000 \
    checkpoint_load_dir=null \
    checkpoint_load_step=null \
    resume_from_checkpoint=false \
    checkpoint_save_dir=gs://sfr-text-diffusion-model-research/checkpoints/pretrain_qwen2_5_Coder_1_5b/ \
    save_steps=500 \
    logging_steps=1 \
    ici_mesh.fsdp=128 \
    ici_mesh.tensor=2 \
    ici_mesh.data=1 \
    ici_mesh.expert=1
    model/remat=qwen2-scan
# fsdp * tensor * data * expert == num_devices

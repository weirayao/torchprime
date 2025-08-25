export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=98304 --xla_enable_async_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
python torchprime/torch_xla_models/train_sanity_check.py \
    training_mode=pretrain \
    data=flex_v2 \
    model=test_data \
    model.block_masking_probability=1 \
    model.mask_block_sizes=[[2,4,8],[16,32,64]] \
    model.truncate_probability=0.5 \
    model.prefix_probability=0.5 \
    model.masking_scheduler.schedule_type=linear \
    model.masking_scheduler.max_schedule_steps=50 \
    optimizer.learning_rate=2e-4 \
    global_batch_size=32 \
    max_steps=30 \
    checkpoint_load_dir=null \
    checkpoint_load_step=null \
    resume_from_checkpoint=false \
    checkpoint_save_dir=gs://sfr-text-diffusion-model-research/checkpoints/pretrain_2nd_run/ \
    save_steps=10 \
    logging_steps=1 \
    ici_mesh.fsdp=4 \
    ici_mesh.tensor=2 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen2-scan
# fsdp * tensor * data * expert == num_devices

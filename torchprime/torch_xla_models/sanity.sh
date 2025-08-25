python sanity_check.py \
    training_mode=pretrain \
    data=test_data \
    model=flex-qwen2-1b \
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

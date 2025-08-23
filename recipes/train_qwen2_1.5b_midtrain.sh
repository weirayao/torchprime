XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/train_mid.py \
    training_mode=pretrain \
    data=mid_train_dataset_v2 \
    model=flex-qwen2-1b \
    model.block_masking_probability=0.3 \
    model.mask_block_sizes=[2,4,8] \
    model.truncate_probability=0.25 \
    model.prefix_probability=0.25 \
    optimizer.learning_rate=3e-4 \
    lr_scheduler.warmup_steps=1 \
    global_batch_size=8 \
    max_steps=36000 \
    checkpoint_dir=gs://sfr-text-diffusion-model-research/checkpoints/midtrain_allv2 \
    checkpoint_dir_for_midtrain=gs://sfr-text-diffusion-model-research/checkpoints/midtrain_allv2_this_is_a_test \
    save_steps=500 \
    steps_to_skip=0 \
    logging_steps=1 \
    ici_mesh.fsdp=4 \
    resume_from_checkpoint=False \
    resume_for_midtrain=False \
    ici_mesh.tensor=2 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen2-scan
# fsdp * tensor * data * expert == num_devices
# global_batch_size mod num_devices == 0
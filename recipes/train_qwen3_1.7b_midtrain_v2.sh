XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/train_mid.py \
    training_mode=pretrain \
    data=mid_train_dataset \
    model=flex-qwen-1b \
    model.block_masking_probability=0.05 \
    model.mask_block_sizes=[2,4,8] \
    model.truncate_probability=0.05 \
    model.prefix_probability=0.05 \
    optimizer.learning_rate=2e-4 \
    global_batch_size=256 \
    max_steps=8000 \
    checkpoint_dir=gs://sfr-text-diffusion-model-research/checkpoints/flex-qwen3-1b-v2 \
    checkpoint_dir_for_midtrain=gs://sfr-text-diffusion-model-research/checkpoints/flex_processed_v2_midtrain \
    resume_from_checkpoint=null \
    save_steps=500 \
    logging_steps=1 \
    ici_mesh.fsdp=256 \
    resume_from_checkpoint=42500 \
    resume_for_midtrain=True \
    ici_mesh.tensor=1 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen-scan
# fsdp * tensor * data * expert == num_devices
# global_batch_size mod num_devices == 0
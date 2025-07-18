XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/train.py \
    training_mode=pretrain \
    data=mixed_train_dataset \
    model=flex-qwen-1b \
    model.block_masking_probability=0 \
    model.truncate_probability=0 \
    model.prefix_probability=0 \
    optimizer.learning_rate=3e-4 \
    global_batch_size=512 \
    max_steps=100 \
    checkpoint_dir=gs://sfr-text-diffusion-model-research/checkpoints/test-masking-flex-qwen3-1b-v2 \
    resume_from_checkpoint=50 \
    save_steps=50 \
    logging_steps=1 \
    ici_mesh.fsdp=512 \
    ici_mesh.tensor=1 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen-scan
# fsdp * tensor * data * expert == num_devices
# global_batch_size mod num_devices == 0